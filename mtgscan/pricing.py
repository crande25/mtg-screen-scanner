"""Price providers, all looked up at scan time (never baked into the index).

Sources, selectable in the UI:

* TcgcsvProvider (default): TCGplayer's official daily price points (lowPrice, marketPrice
  per Normal/Foil printing) republished free by tcgcsv.com, keyed by the TCGplayer
  product id that Scryfall gives every card. "Low" here is TCGplayer's own lowest
  listing figure across conditions; its FAQ calls it the closest approximation of the
  lowest listing available without SKU-level API access. Refreshed daily by tcgcsv.
* ScryfallMarketProvider: the TCGplayer *market* price Scryfall syncs every 24 hours
  (usd / usd_foil / usd_etched). Official, no keys, market only.
* TcgplayerOfficialProvider: lowest Near Mint listing per finish via the documented
  TCGplayer API (catalog SKUs + /pricing/sku). Needs a key pair, which TCGplayer no
  longer hands out freely.
* TcgplayerSiteProvider: lowest Near Mint listing from the marketplace JSON the
  tcgplayer.com product page itself loads. Unofficial and unverified here; opt-in.
* JustTcgProvider: the Near Mint price per printing from justtcg.com, which tracks
  TCGplayer and updates about every six hours, the freshest condition-level figure
  available without a TCGplayer key. Needs a free JustTCG key (1,000 requests a month,
  100 a day, 10 a minute), so quotes from it are cached for six hours and the quota
  left for the day is shown with each one.

PriceService picks the provider for the chosen mode, falls back per card when a product
is unknown and for a while when a provider is down (to tcgcsv first, since it has a low
figure, then to Scryfall's market), caches results and runs lookups on a background
thread so scanning never waits on the network.

Check any card from the command line:  python -m mtgscan.pricing "Lightning Bolt" --set m10
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import requests

from . import scryfall
from .config import USER_AGENT, data_dir
from .scryfall import CardRecord, RateLimiter

log = logging.getLogger(__name__)

MODE_TCGCSV = "tcgcsv_daily"
MODE_SCRYFALL = "scryfall_market"
MODE_TCG_API = "tcgplayer_api_lowest"
MODE_TCG_SITE = "tcgplayer_site_lowest"
MODE_JUSTTCG = "justtcg_nm"
MODES = {
    MODE_TCGCSV: "TCGplayer low + market (tcgcsv.com, daily)",
    MODE_JUSTTCG: "JustTCG Near Mint (6-hourly, free key, 100/day)",
    MODE_SCRYFALL: "TCGplayer market only (Scryfall, daily)",
    MODE_TCG_API: "TCGplayer lowest Near Mint (official API, needs keys)",
    MODE_TCG_SITE: "TCGplayer lowest Near Mint (website data, unofficial)",
}
MODE_SHORT = {
    MODE_TCGCSV: "tcgcsv daily",
    MODE_JUSTTCG: "JustTCG NM",
    MODE_SCRYFALL: "Scryfall market",
    MODE_TCG_API: "TCG API NM",
    MODE_TCG_SITE: "TCG site NM",
}
# How long a quote stays fresh, per mode. JustTCG moves every six hours and its free
# quota is 100 calls a day, so asking again sooner only spends quota.
MODE_TTL_S = {MODE_JUSTTCG: 6 * 3600.0}


class PriceUnavailable(Exception):
    """The provider is down or refused (blocked, bad keys, rate limited): back off.

    retry_s says how long to leave it alone; a daily quota that is spent stays spent
    until midnight UTC, a per-minute limit clears in a minute.
    """

    def __init__(self, msg: str, retry_s: float = 300.0):
        super().__init__(msg)
        self.retry_s = retry_s


class PriceNotFound(Exception):
    """This provider has nothing for this card: fall back for this card only."""


@dataclass
class PriceQuote:
    low: Optional[float] = None            # lowest listing, normal printing
    low_foil: Optional[float] = None
    low_etched: Optional[float] = None
    market: Optional[float] = None         # market price, normal printing
    market_foil: Optional[float] = None
    market_etched: Optional[float] = None
    low_kind: str = ""                     # "any condition" | "Near Mint" | "" when no low figure
    source: str = ""
    url: str = ""
    fetched_at: float = field(default_factory=time.time)
    note: str = ""

    @staticmethod
    def fmt(v: Optional[float]) -> str:
        return "-" if v is None else f"${v:,.2f}"

    def summary(self) -> str:
        f = PriceQuote.fmt
        parts = []
        if self.low_kind:
            parts.append(f"low ({self.low_kind}): {f(self.low)} / foil {f(self.low_foil)}")
        parts.append(f"market: {f(self.market)} / foil {f(self.market_foil)}")
        if self.low_etched is not None or self.market_etched is not None:
            parts.append(f"etched: low {f(self.low_etched)} / market {f(self.market_etched)}")
        s = f"{self.source}: " + "; ".join(parts)
        return s + (f" [{self.note}]" if self.note else "")


def describe_quote(card: CardRecord, quote: Optional[PriceQuote], age_s: Optional[float]) -> dict:
    """Everything known about one card's price, laid out for display.

    Kept out of the window code so the wording can be tested without building a UI.
    """
    finishes = card.finishes.split(",") if card.finishes else []
    rows = []
    for label, finish, low_attr, market_attr in (
            ("Normal", "nonfoil", "low", "market"),
            ("Foil", "foil", "low_foil", "market_foil"),
            ("Etched", "etched", "low_etched", "market_etched")):
        if finish not in finishes:
            continue
        low = getattr(quote, low_attr) if quote else None
        market = getattr(quote, market_attr) if quote else None
        if market is None and quote is None:
            market = card.usd_foil if finish == "foil" else (card.usd_etched if finish == "etched" else card.usd)
        rows.append({"finish": label,
                     "low": PriceQuote.fmt(low),
                     "market": PriceQuote.fmt(market)})
    if not rows:
        rows.append({"finish": "Normal", "low": "-", "market": PriceQuote.fmt(card.usd)})

    if quote is None:
        source, freshness = "waiting for a price...", ""
    elif quote.source == "unavailable":
        source, freshness = "no price source could answer", ""
    else:
        source = quote.source
        if quote.low_kind:
            source += f" (low is {quote.low_kind})"
        if age_s is None:
            freshness = ""
        elif age_s < 90:
            freshness = f"fetched {int(age_s)}s ago"
        else:
            freshness = f"fetched {int(age_s / 60)} min ago"
    return {"title": f"{card.name}  -  {card.set_label}  #{card.collector_number}",
            "rows": rows, "source": source, "freshness": freshness,
            "note": quote.note if quote else "",
            "url": card.tcgplayer_listing_url() or card.tcgplayer_product_url}


def _f(v) -> Optional[float]:
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def keys_from_env() -> tuple[str, str]:
    return os.environ.get("TCGPLAYER_PUBLIC_KEY", ""), os.environ.get("TCGPLAYER_PRIVATE_KEY", "")


def justtcg_key_from_env() -> str:
    return os.environ.get("JUSTTCG_API_KEY", "")


def seconds_until_utc_midnight(now: float | None = None) -> float:
    """JustTCG's daily quota resets at midnight UTC."""
    t = time.time() if now is None else now
    return 86400.0 - (t % 86400.0)


def _json_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return s


# --- Scryfall (TCGplayer market, daily) ---------------------------------------------

class ScryfallMarketProvider:
    name = "TCGplayer market (Scryfall)"

    def get(self, card: CardRecord) -> PriceQuote:
        data = scryfall.fetch_card(card.id)
        p = data.get("prices") or {}
        url = card.tcgplayer_product_url or (data.get("purchase_uris") or {}).get("tcgplayer", "")
        return PriceQuote(market=_f(p.get("usd")), market_foil=_f(p.get("usd_foil")),
                          market_etched=_f(p.get("usd_etched")), source=self.name, url=url)


# --- tcgcsv.com (TCGplayer official price points, daily) ------------------------------

class TcgcsvProvider:
    """TCGplayer low/market per product and finish from tcgcsv.com.

    Files are per TCGplayer *group* (set). The group of a product is found by matching the
    set abbreviation first (M10, MH3, ...) and otherwise from a product->group map that is
    built once in the background (one request per group, ~450 for Magic) and kept on disk,
    refreshed only for groups whose modifiedOn changed.
    """

    name = "TCGplayer low/market (tcgcsv)"
    BASE = "https://tcgcsv.com/tcgplayer"
    CATEGORY = 1  # Magic: The Gathering

    def __init__(self, cache_dir: Path | None = None, per_second: float = 4.0, prices_ttl_s: float = 3600.0):
        self.cache_dir = Path(cache_dir) if cache_dir else data_dir()
        self.limiter = RateLimiter(per_second)
        self.s = _json_session()
        self.prices_ttl_s = prices_ttl_s
        self._groups: list[dict] = []
        self._groups_at = 0.0
        self._prices: dict[int, tuple[float, dict[tuple[int, str], dict]]] = {}
        self._lock = threading.RLock()
        self._map: dict[str, int] = {}          # productId -> groupId
        self._map_groups: dict[str, str] = {}   # groupId -> modifiedOn when its products were read
        self._map_thread: Optional[threading.Thread] = None
        self._load_map()

    # -- http --------------------------------------------------------------------------
    def _get(self, path: str) -> list:
        self.limiter.wait()
        r = self.s.get(f"{self.BASE}/{path}", timeout=20)
        if r.status_code == 404:
            raise PriceNotFound(f"tcgcsv has no {path}")
        if r.status_code in (403, 429, 503):
            raise PriceUnavailable(f"tcgcsv answered HTTP {r.status_code}")
        r.raise_for_status()
        return r.json().get("results") or []

    def groups(self) -> list[dict]:
        with self._lock:
            if self._groups and time.time() - self._groups_at < 86400:
                return self._groups
            self._groups = self._get(f"{self.CATEGORY}/groups")
            self._groups_at = time.time()
            return self._groups

    def prices(self, group_id: int) -> dict[tuple[int, str], dict]:
        with self._lock:
            cached = self._prices.get(group_id)
            if cached and time.time() - cached[0] < self.prices_ttl_s:
                return cached[1]
        rows = self._get(f"{self.CATEGORY}/{group_id}/prices")
        table = {(int(r["productId"]), str(r.get("subTypeName") or "Normal")): r for r in rows if "productId" in r}
        with self._lock:
            self._prices[group_id] = (time.time(), table)
        return table

    # -- product -> group map -----------------------------------------------------------
    @property
    def _map_path(self) -> Path:
        return self.cache_dir / "tcgcsv_products.json"

    def _load_map(self) -> None:
        try:
            d = json.loads(self._map_path.read_text())
            self._map = {str(k): int(v) for k, v in d.get("products", {}).items()}
            self._map_groups = dict(d.get("groups", {}))
        except (OSError, ValueError):
            self._map, self._map_groups = {}, {}

    def _save_map(self) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._map_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"products": self._map, "groups": self._map_groups}))
            tmp.replace(self._map_path)
        except OSError as e:
            log.warning("could not save tcgcsv map: %s", e)

    @property
    def map_ready(self) -> bool:
        return bool(self._map_groups) and (self._map_thread is None or not self._map_thread.is_alive())

    def ensure_map_async(self) -> None:
        with self._lock:
            if self._map_thread is not None and self._map_thread.is_alive():
                return
            self._map_thread = threading.Thread(target=self._build_map, name="tcgcsv-map", daemon=True)
            self._map_thread.start()

    def _build_map(self) -> None:
        try:
            groups = self.groups()
        except Exception as e:
            log.warning("tcgcsv groups unavailable: %s", e)
            return
        changed = 0
        for g in groups:
            gid = int(g["groupId"])
            if self._map_groups.get(str(gid)) == g.get("modifiedOn"):
                continue
            try:
                products = self._get(f"{self.CATEGORY}/{gid}/products")
            except Exception as e:
                log.warning("tcgcsv products for group %s failed: %s", gid, e)
                continue
            with self._lock:
                for p in products:
                    if "productId" in p:
                        self._map[str(p["productId"])] = gid
                self._map_groups[str(gid)] = g.get("modifiedOn", "")
            changed += 1
            if changed % 25 == 0:
                with self._lock:
                    self._save_map()
        with self._lock:
            self._save_map()
        log.info("tcgcsv product map ready: %d products, %d groups updated", len(self._map), changed)

    def group_for(self, product_id: int, set_code: str) -> Optional[int]:
        with self._lock:
            gid = self._map.get(str(product_id))
        if gid is not None:
            return gid
        code = (set_code or "").upper()
        if code:
            for g in self.groups():
                if (g.get("abbreviation") or "").upper() != code:
                    continue
                gid = int(g["groupId"])
                table = self.prices(gid)
                if any(pid == product_id for pid, _ in table):
                    with self._lock:
                        self._map[str(product_id)] = gid
                    return gid
        self.ensure_map_async()
        return None

    # -- quote -------------------------------------------------------------------------
    def get(self, card: CardRecord) -> PriceQuote:
        if not card.tcgplayer_id:
            raise PriceNotFound("card has no TCGplayer product id")
        gid = self.group_for(card.tcgplayer_id, card.set_code)
        if gid is None:
            raise PriceNotFound("TCGplayer set not mapped yet (building product map in background)")
        table = self.prices(gid)
        n = table.get((card.tcgplayer_id, "Normal"))
        f = table.get((card.tcgplayer_id, "Foil"))
        if n is None and f is None:
            raise PriceNotFound("product missing from tcgcsv price file")
        q = PriceQuote(source=self.name, url=card.tcgplayer_product_url, low_kind="any condition")
        if n:
            q.low, q.market = _f(n.get("lowPrice")), _f(n.get("marketPrice"))
        if f:
            q.low_foil, q.market_foil = _f(f.get("lowPrice")), _f(f.get("marketPrice"))
        if card.tcgplayer_etched_id:
            try:
                gid2 = self.group_for(card.tcgplayer_etched_id, card.set_code)
                if gid2 is not None:
                    t2 = self.prices(gid2)
                    e = t2.get((card.tcgplayer_etched_id, "Foil")) or t2.get((card.tcgplayer_etched_id, "Normal"))
                    if e:
                        q.low_etched, q.market_etched = _f(e.get("lowPrice")), _f(e.get("marketPrice"))
            except (PriceNotFound, PriceUnavailable):
                pass
        return q


# --- Official TCGplayer API (needs keys) ---------------------------------------------

class TcgplayerOfficialProvider:
    """Lowest Near Mint listing per finish via the documented TCGplayer API.

    Flow (docs.tcgplayer.com): POST /token (client credentials) -> bearer token;
    GET /catalog/categories/1/{conditions,printings,languages} to map names to ids;
    GET /catalog/products/{id}/skus -> the SKU for (English, Near Mint, Normal|Foil);
    GET /pricing/sku/{skuIds} -> lowPrice / marketPrice per SKU.
    """

    name = "TCGplayer lowest NM (API)"
    BASE = "https://api.tcgplayer.com"
    MAGIC_CATEGORY = 1

    def __init__(self, public_key: str, private_key: str, per_second: float = 5.0):
        self.public_key, self.private_key = public_key.strip(), private_key.strip()
        self.limiter = RateLimiter(per_second)
        self.s = _json_session()
        self._token_exp = 0.0
        self._cond_id: Optional[int] = None
        self._lang_id: Optional[int] = None
        self._print_ids: dict[str, int] = {}
        self._sku_cache: dict[int, list[dict]] = {}
        self._lock = threading.Lock()

    @property
    def configured(self) -> bool:
        return bool(self.public_key and self.private_key)

    def _auth(self) -> None:
        if time.time() < self._token_exp:
            return
        if not self.configured:
            raise PriceUnavailable("no TCGplayer API keys configured")
        self.limiter.wait()
        r = self.s.post(f"{self.BASE}/token", data={
            "grant_type": "client_credentials", "client_id": self.public_key, "client_secret": self.private_key,
        }, timeout=20)
        if r.status_code in (400, 401, 403):
            raise PriceUnavailable(f"TCGplayer API rejected the keys (HTTP {r.status_code})")
        r.raise_for_status()
        j = r.json()
        self.s.headers["Authorization"] = f"bearer {j['access_token']}"
        self._token_exp = time.time() + float(j.get("expires_in", 3600)) - 120

    def _get(self, path: str, **params) -> list:
        self._auth()
        self.limiter.wait()
        r = self.s.get(f"{self.BASE}{path}", params=params or None, timeout=20)
        if r.status_code == 401:
            self._token_exp = 0.0
            self._auth()
            r = self.s.get(f"{self.BASE}{path}", params=params or None, timeout=20)
        if r.status_code in (403, 429, 503):
            raise PriceUnavailable(f"TCGplayer API answered HTTP {r.status_code}")
        if r.status_code == 404:
            return []
        r.raise_for_status()
        j = r.json()
        if not j.get("success", True):
            raise PriceUnavailable("; ".join(j.get("errors") or ["TCGplayer API error"]))
        return j.get("results") or []

    def _load_maps(self) -> None:
        if self._cond_id is not None:
            return
        conds = self._get(f"/catalog/categories/{self.MAGIC_CATEGORY}/conditions")
        prints = self._get(f"/catalog/categories/{self.MAGIC_CATEGORY}/printings")
        langs = self._get(f"/catalog/categories/{self.MAGIC_CATEGORY}/languages")
        self._cond_id = next((c["conditionId"] for c in conds if c.get("name") == "Near Mint"), None)
        self._lang_id = next((l["languageId"] for l in langs if l.get("name") == "English"), None)
        self._print_ids = {p["name"]: p["printingId"] for p in prints if "name" in p}
        if self._cond_id is None or self._lang_id is None or "Normal" not in self._print_ids:
            raise PriceUnavailable("TCGplayer catalog did not return the expected condition/printing names")

    def _skus(self, product_id: int) -> list[dict]:
        if product_id not in self._sku_cache:
            self._sku_cache[product_id] = self._get(f"/catalog/products/{product_id}/skus")
        return self._sku_cache[product_id]

    def _nm_sku(self, product_id: int, printing: str) -> Optional[int]:
        pid = self._print_ids.get(printing)
        for s in self._skus(product_id):
            if s.get("conditionId") == self._cond_id and s.get("languageId") == self._lang_id and s.get("printingId") == pid:
                return s.get("skuId")
        return None

    def get(self, card: CardRecord) -> PriceQuote:
        if not card.tcgplayer_id:
            raise PriceNotFound("card has no TCGplayer product id")
        with self._lock:
            self._load_maps()
            finishes = card.finishes.split(",")
            wanted: dict[str, int] = {}
            if "nonfoil" in finishes:
                sid = self._nm_sku(card.tcgplayer_id, "Normal")
                if sid:
                    wanted["normal"] = sid
            if "foil" in finishes:
                sid = self._nm_sku(card.tcgplayer_id, "Foil")
                if sid:
                    wanted["foil"] = sid
            if "etched" in finishes and card.tcgplayer_etched_id:
                sid = self._nm_sku(card.tcgplayer_etched_id, "Foil") or self._nm_sku(card.tcgplayer_etched_id, "Normal")
                if sid:
                    wanted["etched"] = sid
            if not wanted:
                raise PriceNotFound("no Near Mint SKU for this product")
            q = PriceQuote(source=self.name, url=card.tcgplayer_product_url, low_kind="Near Mint")
            rows = self._get("/pricing/sku/" + ",".join(str(v) for v in wanted.values()))
            by_sku = {r.get("skuId"): r for r in rows}
            for finish, sid in wanted.items():
                row = by_sku.get(sid) or {}
                low = _f(row.get("lowPrice"))
                if low is None:
                    low = _f(row.get("lowestListingPrice"))
                market = _f(row.get("marketPrice"))
                if finish == "normal":
                    q.low, q.market = low, market
                elif finish == "foil":
                    q.low_foil, q.market_foil = low, market
                else:
                    q.low_etched, q.market_etched = low, market
            return q


# --- Unofficial website data --------------------------------------------------------

class TcgplayerSiteProvider:
    """Lowest NM listing per printing from the marketplace listings JSON (unofficial)."""

    name = "TCGplayer lowest NM (site)"
    LISTINGS_URL = "https://mp-search-api.tcgplayer.com/v1/product/{pid}/listings"

    def __init__(self, country: str = "US", per_second: float = 1.0):
        self.country = country
        self.limiter = RateLimiter(per_second)
        self.s = _json_session()
        self.s.headers.update({"Origin": "https://www.tcgplayer.com", "Referer": "https://www.tcgplayer.com/"})

    def _body(self, printing: str, size: int = 5) -> dict:
        return {
            "filters": {
                "term": {
                    "sellerStatus": "Live",
                    "channelId": 0,
                    "language": ["English"],
                    "condition": ["Near Mint"],
                    "printing": [printing],
                    "listingType": "standard",
                },
                "range": {"quantity": {"gte": 1}},
                "exclude": {"channelExclusion": 0, "listingType": "custom"},
            },
            "from": 0,
            "size": size,
            "sort": {"field": "price", "order": "asc"},
            "context": {"shippingCountry": self.country, "cart": {}},
            "aggregations": ["listingType"],
        }

    def lowest(self, product_id: int, printing: str) -> Optional[float]:
        self.limiter.wait()
        r = self.s.post(self.LISTINGS_URL.format(pid=product_id), json=self._body(printing), timeout=15)
        if r.status_code in (401, 403, 429, 503):
            raise PriceUnavailable(f"TCGplayer site answered HTTP {r.status_code}")
        r.raise_for_status()
        data = r.json()
        results = data.get("results") or []
        listings = (results[0].get("results") if results else None) or []
        prices = [p for p in (_f(x.get("price")) for x in listings) if p is not None]
        return min(prices) if prices else None

    def get(self, card: CardRecord) -> PriceQuote:
        if not card.tcgplayer_id:
            raise PriceNotFound("card has no TCGplayer product id")
        q = PriceQuote(source=self.name, url=card.tcgplayer_product_url, low_kind="Near Mint")
        finishes = card.finishes.split(",")
        if "nonfoil" in finishes:
            q.low = self.lowest(card.tcgplayer_id, "Normal")
        if "foil" in finishes:
            q.low_foil = self.lowest(card.tcgplayer_id, "Foil")
        if "etched" in finishes:
            q.low_etched = self.lowest(card.tcgplayer_etched_id or card.tcgplayer_id, "Foil")
        return q


# --- JustTCG (tracks TCGplayer, condition-level, about six-hourly, free key) ------------

class JustTcgProvider:
    """Near Mint price per printing from api.justtcg.com, looked up by TCGplayer product id.

    One GET /v1/cards?tcgplayerId=... per product, filtered to English Near Mint, with the
    price history left out to keep the answer small. JustTCG reports one figure per
    (condition, printing), "the last price the market reported for this condition", so it
    goes in the low column labelled Near Mint; the market columns keep Scryfall's daily
    TCGplayer market price from the index, and the quote says so.

    Errors follow docs.justtcg.com: 401 bad key, 403 no active plan, 429 with a code that
    says whether the minute, the day or the month ran out.
    """

    name = "JustTCG Near Mint"
    BASE = "https://api.justtcg.com/v1"

    def __init__(self, api_key: str, per_minute: float = 10.0):
        self.api_key = (api_key or "").strip()
        self.limiter = RateLimiter(per_minute / 60.0)
        self.s = _json_session()
        self.requests_left_today: Optional[int] = None
        self.requests_left_month: Optional[int] = None
        self.plan: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _get(self, **params) -> dict:
        if not self.configured:
            raise PriceUnavailable("no JustTCG key configured (API keys...)", retry_s=3600.0)
        self.limiter.wait()
        r = self.s.get(f"{self.BASE}/cards", params=params, headers={"x-api-key": self.api_key}, timeout=20)
        if r.status_code in (400, 401, 403, 429):
            try:
                body = r.json()
            except ValueError:
                body = {}
            code = str(body.get("code") or "")
            msg = str(body.get("error") or f"HTTP {r.status_code}")
            if r.status_code == 401:
                raise PriceUnavailable(f"JustTCG rejected the key ({msg})", retry_s=3600.0)
            if r.status_code == 403:
                raise PriceUnavailable(f"JustTCG plan not active ({msg})", retry_s=3600.0)
            if code == "DAILY_LIMIT_EXCEEDED":
                self.requests_left_today = 0
                raise PriceUnavailable("JustTCG daily quota used up; resets at midnight UTC",
                                       retry_s=seconds_until_utc_midnight())
            if code == "REQUEST_LIMIT_EXCEEDED":
                self.requests_left_month = 0
                raise PriceUnavailable("JustTCG monthly quota used up", retry_s=86400.0)
            if r.status_code == 429:
                raise PriceUnavailable(f"JustTCG rate limit ({msg})", retry_s=65.0)
            raise PriceUnavailable(f"JustTCG refused the request ({msg})", retry_s=300.0)
        r.raise_for_status()
        body = r.json()
        meta = body.get("_metadata") or {}
        if "apiDailyRequestsRemaining" in meta:
            self.requests_left_today = _int(meta.get("apiDailyRequestsRemaining"))
        if "apiRequestsRemaining" in meta:
            self.requests_left_month = _int(meta.get("apiRequestsRemaining"))
        self.plan = str(meta.get("apiPlan") or self.plan)
        return body

    def quota_note(self) -> str:
        parts = []
        if self.requests_left_today is not None:
            parts.append(f"{self.requests_left_today} JustTCG requests left today")
        if self.requests_left_month is not None:
            parts.append(f"{self.requests_left_month} this month")
        return ", ".join(parts)

    @staticmethod
    def nm_prices(body: dict, product_id: int) -> dict[str, float]:
        """{printing: price} for the Near Mint variants of one product in a /cards answer.

        Takes the card whose tcgplayerId matches (the filter should return just one, but
        a search can bring back several). A variant with no current price falls back to
        its last known one; one with neither is skipped.
        """
        cards = body.get("data") or []
        card = next((c for c in cards if str(c.get("tcgplayerId")) == str(product_id)), None)
        if card is None:
            card = cards[0] if len(cards) == 1 else None
        out: dict[str, float] = {}
        for v in (card or {}).get("variants") or []:
            cond = str(v.get("condition") or "")
            if cond not in ("Near Mint", "NM"):
                continue
            price = _f(v.get("price"))
            if price is None:
                price = _f(v.get("lastKnownPrice"))
            if price is None:
                continue
            out[str(v.get("printing") or "Normal")] = price
        return out

    def _lookup(self, product_id: int) -> dict[str, float]:
        body = self._get(tcgplayerId=str(product_id), condition="NM", language="English",
                         include_price_history="false", limit=5)
        return self.nm_prices(body, product_id)

    def get(self, card: CardRecord) -> PriceQuote:
        if not card.tcgplayer_id:
            raise PriceNotFound("card has no TCGplayer product id")
        prices = self._lookup(card.tcgplayer_id)
        finishes = card.finishes.split(",") if card.finishes else []
        if card.tcgplayer_etched_id and "etched" in finishes and card.tcgplayer_etched_id != card.tcgplayer_id:
            # Etched foils are a product of their own on TCGplayer; its foil printing is the etched price.
            try:
                etched = self._lookup(card.tcgplayer_etched_id)
                price = next((p for k, p in etched.items() if "Etched" in k), None)
                if price is None:
                    price = etched.get("Foil", next(iter(etched.values()), None))
                if price is not None:
                    prices["Etched"] = price
            except PriceNotFound:
                pass
        if not prices:
            raise PriceNotFound("JustTCG has no Near Mint price for this product")
        q = PriceQuote(source=self.name, url=card.tcgplayer_listing_url() or card.tcgplayer_product_url,
                       low_kind="Near Mint")
        q.low = prices.get("Normal")
        q.low_foil = prices.get("Foil")
        q.low_etched = next((p for k, p in prices.items() if "Etched" in k), None)
        # JustTCG has no separate market figure; Scryfall's daily TCGplayer market fills in.
        q.market, q.market_foil, q.market_etched = card.usd, card.usd_foil, card.usd_etched
        notes = ["market is Scryfall's daily figure"]
        quota = self.quota_note()
        if quota:
            notes.append(quota)
        q.note = "; ".join(notes)
        return q


def _int(v) -> Optional[int]:
    try:
        return int(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


# --- Service ---------------------------------------------------------------------------

PriceCallback = Callable[[CardRecord, PriceQuote], None]


class PriceService:
    """Caching, background price lookups with provider fallback."""

    def __init__(self, mode: str = MODE_TCGCSV, keys: tuple[str, str] | None = None, ttl_s: float = 900.0,
                 cache_dir: Path | None = None, justtcg_key: str = ""):
        self.mode = mode if mode in MODES else MODE_TCGCSV
        self.ttl_s = ttl_s
        pub, priv = keys if keys and (keys[0] or keys[1]) else keys_from_env()
        self._cache: dict[tuple[str, str], PriceQuote] = {}
        self._lock = threading.Lock()
        self._q: "queue.Queue[tuple[CardRecord, PriceCallback]]" = queue.Queue()
        self._inflight: set[str] = set()
        self._scry = ScryfallMarketProvider()
        self._csv = TcgcsvProvider(cache_dir=cache_dir)
        self._api = TcgplayerOfficialProvider(pub, priv)
        self._site = TcgplayerSiteProvider()
        self._just = JustTcgProvider(justtcg_key or justtcg_key_from_env())
        self._disabled_until: dict[str, float] = {}
        self._disabled_why: dict[str, str] = {}
        self._thread = threading.Thread(target=self._worker, name="prices", daemon=True)
        self._thread.start()

    # -- configuration ---------------------------------------------------------------
    def set_mode(self, mode: str) -> bool:
        if mode in MODES and mode != self.mode:
            self.mode = mode
            return True
        return False

    def set_keys(self, public_key: str, private_key: str) -> None:
        self._api = TcgplayerOfficialProvider(public_key, private_key)
        self._disabled_until.pop(MODE_TCG_API, None)
        with self._lock:
            self._cache = {k: v for k, v in self._cache.items() if k[0] != MODE_TCG_API}

    def set_justtcg_key(self, api_key: str) -> None:
        self._just = JustTcgProvider(api_key or justtcg_key_from_env())
        self._disabled_until.pop(MODE_JUSTTCG, None)
        with self._lock:
            self._cache = {k: v for k, v in self._cache.items() if k[0] != MODE_JUSTTCG}

    def provider_for(self, mode: str):
        return {MODE_TCGCSV: self._csv, MODE_TCG_API: self._api, MODE_TCG_SITE: self._site,
                MODE_JUSTTCG: self._just}.get(mode)

    @property
    def justtcg(self) -> JustTcgProvider:
        return self._just

    def ttl_for(self, mode: str) -> float:
        return MODE_TTL_S.get(mode, self.ttl_s)

    def paused_reason(self, mode: str | None = None) -> str:
        """Why the provider for this mode is not being asked right now, or ''."""
        mode = mode or self.mode
        if time.time() < self._disabled_until.get(mode, 0.0):
            return self._disabled_why.get(mode, "paused after an error")
        return ""

    # -- synchronous -----------------------------------------------------------------
    def get(self, card: CardRecord) -> PriceQuote:
        key = (self.mode, card.id)
        with self._lock:
            q = self._cache.get(key)
        if q and time.time() - q.fetched_at < self.ttl_for(self.mode):
            return q
        q = self._fetch(card)
        with self._lock:
            self._cache[key] = q
        return q

    def refresh(self, card: CardRecord) -> PriceQuote:
        """Fetch this card's price again now, ignoring the cache.

        Quotes are cached for fifteen minutes so scanning does not hammer a provider, but
        when someone is watching one card and asks for its price, they mean the price now.
        """
        with self._lock:
            self._cache.pop((self.mode, card.id), None)
            self._disabled_until.pop(self.mode, None)   # give a paused provider another go
        return self.get(card)

    def age_of(self, card: CardRecord) -> Optional[float]:
        """Seconds since this card's cached quote was fetched."""
        with self._lock:
            q = self._cache.get((self.mode, card.id))
        return None if q is None else time.time() - q.fetched_at

    def cached(self, card: CardRecord) -> Optional[PriceQuote]:
        with self._lock:
            q = self._cache.get((self.mode, card.id))
        return q if q and time.time() - q.fetched_at < self.ttl_for(self.mode) else None

    def _try(self, provider, mode: str, card: CardRecord) -> tuple[Optional[PriceQuote], str]:
        """Ask one provider; (quote, '') on success, (None, why) otherwise."""
        if time.time() < self._disabled_until.get(mode, 0.0):
            return None, f"fallback: {self._disabled_why.get(mode, 'provider paused after an error')}"
        try:
            return provider.get(card), ""
        except PriceNotFound as e:
            return None, f"fallback: {e}"
        except PriceUnavailable as e:
            log.warning("%s unavailable (%s); leaving it alone for %d s", provider.name, e, e.retry_s)
            self._disabled_until[mode] = time.time() + e.retry_s
            self._disabled_why[mode] = str(e)
            return None, f"fallback: {e}"
        except Exception as e:
            log.warning("%s lookup failed for %s: %s", provider.name, card.name, e)
            return None, "fallback after lookup error"

    def _fetch(self, card: CardRecord) -> PriceQuote:
        note = ""
        primary = self.provider_for(self.mode)
        if primary is not None:
            q, note = self._try(primary, self.mode, card)
            if q is not None:
                return q
        # tcgcsv still has a low figure, so it comes before Scryfall's market-only answer.
        if self.mode not in (MODE_TCGCSV, MODE_SCRYFALL):
            q, why = self._try(self._csv, MODE_TCGCSV, card)
            if q is not None:
                q.note = note
                return q
        try:
            q = self._scry.get(card)
            q.note = note
            return q
        except Exception as e:
            log.warning("Scryfall price lookup failed for %s: %s", card.name, e)
            return PriceQuote(source="unavailable", url=card.tcgplayer_product_url, note=note or str(e))

    # -- asynchronous ----------------------------------------------------------------
    def request(self, card: CardRecord, callback: PriceCallback) -> None:
        q = self.cached(card)
        if q is not None:
            callback(card, q)
            return
        with self._lock:
            if card.id in self._inflight:
                return
            self._inflight.add(card.id)
        self._q.put((card, callback))

    def _worker(self) -> None:
        while True:
            card, cb = self._q.get()
            try:
                quote = self.get(card)
            except Exception as e:  # pragma: no cover
                quote = PriceQuote(source="unavailable", note=str(e))
            finally:
                with self._lock:
                    self._inflight.discard(card.id)
            try:
                cb(card, quote)
            except Exception:
                log.exception("price callback failed")


# --- CLI: check what each source returns for one card ---------------------------------

def cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mtgscan.pricing",
                                 description="Show what every price source returns for a card.")
    ap.add_argument("name", help="exact card name, e.g. \"Lightning Bolt\"")
    ap.add_argument("--set", default=None, help="set code to pick a printing, e.g. m10")
    ap.add_argument("--modes", default="all", help="comma separated: " + ",".join(MODES) + " (default all)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    params = {"exact": args.name}
    if args.set:
        params["set"] = args.set
    try:
        card = scryfall.api_get("/cards/named", **params)
    except Exception as e:
        print(f"Scryfall lookup failed: {e}", file=sys.stderr)
        return 1
    recs = scryfall.records_from_card(card)
    if not recs:
        print("card has no image/record", file=sys.stderr)
        return 1
    rec = recs[0]
    print(f"{rec.name} - {rec.set_label} #{rec.collector_number}  tcgplayer_id={rec.tcgplayer_id}  finishes={rec.finishes}")
    svc = PriceService(mode=MODE_SCRYFALL)
    modes = list(MODES) if args.modes == "all" else [m.strip() for m in args.modes.split(",")]
    if args.modes == "all" and not svc.justtcg.configured:
        modes.remove(MODE_JUSTTCG)       # no key in JUSTTCG_API_KEY: nothing to show
    for mode in modes:
        provider = svc.provider_for(mode) or svc._scry
        t0 = time.perf_counter()
        try:
            q = provider.get(rec)
            print(f"[{MODE_SHORT.get(mode, mode)}] {q.summary()}  ({(time.perf_counter() - t0) * 1000:.0f} ms)")
        except (PriceNotFound, PriceUnavailable) as e:
            print(f"[{MODE_SHORT.get(mode, mode)}] not available: {e}")
        except Exception as e:
            print(f"[{MODE_SHORT.get(mode, mode)}] error: {e}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli())
