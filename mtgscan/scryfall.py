"""Scryfall access: bulk data download, compact card records, live card lookups.

Scryfall is the de-facto card database for every hobbyist scanner; it exposes
every printing with images, TCGplayer product ids and daily TCGplayer prices.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable, Iterator, Optional

import requests

from .config import USER_AGENT

API = "https://api.scryfall.com"
log = logging.getLogger(__name__)


class RateLimiter:
    """Minimum spacing between requests, shared across threads."""

    def __init__(self, per_second: float):
        self.min_interval = 1.0 / per_second
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
                now = time.monotonic()
            self._next = max(now, self._next) + self.min_interval


# Scryfall asks clients to stay under 10 requests/second on api.scryfall.com and to send
# UA + Accept. Images come from the cards.scryfall.io CDN, which is not under that
# limit; the image pace is configurable (see set_image_rate) and backs off on 429.
API_LIMITER = RateLimiter(8.0)
IMAGE_LIMITER = RateLimiter(10.0)


def set_image_rate(per_second: float) -> None:
    IMAGE_LIMITER.min_interval = 1.0 / max(0.5, float(per_second))

_session: Optional[requests.Session] = None
_session_lock = threading.Lock()


def session() -> requests.Session:
    global _session
    with _session_lock:
        if _session is None:
            s = requests.Session()
            s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json;q=0.9,*/*;q=0.8"})
            _session = s
        return _session


def api_get(path_or_url: str, **params) -> dict:
    url = path_or_url if path_or_url.startswith("http") else API + path_or_url
    r = None
    for attempt in range(3):
        API_LIMITER.wait()
        r = session().get(url, params=params or None, timeout=30)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(1.5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    assert r is not None
    r.raise_for_status()
    return r.json()


def fetch_card(card_id: str) -> dict:
    return api_get(f"/cards/{card_id}")


def search_cards(query: str, unique: str = "prints") -> Iterator[dict]:
    """Iterate all results of a Scryfall search (used for small test indexes)."""
    page = api_get("/cards/search", q=query, unique=unique)
    while True:
        yield from page.get("data", [])
        if not page.get("has_more"):
            return
        page = api_get(page["next_page"])


def download_image(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(3):
        IMAGE_LIMITER.wait()
        try:
            r = session().get(url, timeout=30)
            if r.status_code == 429:
                # The CDN pushed back: halve the global image pace for the rest of the run.
                IMAGE_LIMITER.min_interval = min(1.0, IMAGE_LIMITER.min_interval * 2)
                log.warning("image CDN rate limited; slowing to %.1f/s", 1.0 / IMAGE_LIMITER.min_interval)
                time.sleep(3.0 * (attempt + 1))
                continue
            if r.status_code >= 500:
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:  # pragma: no cover - network
            last = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"image download failed: {url}: {last}")


# --- Bulk data ----------------------------------------------------------------

def get_bulk_info(bulk_type: str = "default_cards") -> dict:
    data = api_get("/bulk-data")
    for item in data["data"]:
        if item["type"] == bulk_type:
            return item
    raise KeyError(bulk_type)


def _bulk_meta_path(dest_dir: Path, bulk_type: str) -> Path:
    return dest_dir / f"{bulk_type}.meta.json"


def download_bulk(bulk_type: str, dest_dir: Path, progress: Callable[[int, int], None] | None = None) -> Path:
    """Download a bulk dump unless the local copy already matches the updated_at Scryfall reports.

    Prefers the gzipped JSON-lines export (a fraction of the size of the JSON array and
    streamable line by line); falls back to the JSON array when a bulk type lacks it.
    Returns the local file path.
    """
    info = get_bulk_info(bulk_type)
    uri = info.get("jsonl_download_uri") or info["download_uri"]
    if uri.endswith(".gz"):
        suffix = ".jsonl.gz"
    elif uri.endswith(".jsonl"):
        suffix = ".jsonl"
    else:
        suffix = ".json"
    dest_dir = Path(dest_dir)
    dest = dest_dir / f"{bulk_type}{suffix}"
    meta_path = _bulk_meta_path(dest_dir, bulk_type)
    if dest.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if meta["updated_at"] == info["updated_at"] and meta.get("file") == dest.name:
                log.info("bulk %s is up to date (%s)", bulk_type, info["updated_at"])
                return dest
        except (ValueError, KeyError):
            pass
    tmp = dest.with_suffix(".part")
    dest_dir.mkdir(parents=True, exist_ok=True)
    API_LIMITER.wait()
    with session().get(uri, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or info.get("size") or 0)
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    tmp.replace(dest)
    meta_path.write_text(json.dumps({"updated_at": info["updated_at"], "type": bulk_type, "file": dest.name}))
    return dest


def bulk_updated_at(dest: Path) -> str:
    bulk_type = dest.name.split(".")[0]
    try:
        return json.loads(_bulk_meta_path(dest.parent, bulk_type).read_text())["updated_at"]
    except (OSError, ValueError, KeyError):
        return ""


def iter_bulk_cards(path: Path) -> Iterator[dict]:
    """Stream card objects from a bulk dump (.jsonl.gz, .jsonl or .json) without loading it in memory."""
    name = path.name
    if name.endswith(".jsonl.gz") or name.endswith(".jsonl"):
        import gzip

        opener = gzip.open if name.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    import ijson

    with open(path, "rb") as f:
        yield from ijson.items(f, "item")


# --- Compact record -------------------------------------------------------------

@dataclass
class CardRecord:
    id: str
    face: int
    name: str
    face_name: str
    set_code: str
    set_name: str
    collector_number: str
    rarity: str
    released_at: str
    illustration_id: str
    tcgplayer_id: Optional[int]
    tcgplayer_etched_id: Optional[int]
    finishes: str          # comma separated: nonfoil,foil,etched
    lang: str
    layout: str
    frame: str
    border_color: str
    promo: int
    digital: int
    set_type: str
    image_small: str
    image_normal: str
    usd: Optional[float]
    usd_foil: Optional[float]
    usd_etched: Optional[float]
    prices_at: str
    scryfall_uri: str
    tcgplayer_uri: str

    @classmethod
    def columns(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    @property
    def set_label(self) -> str:
        return f"{self.set_code.upper()} - {self.set_name}"

    @property
    def has_foil(self) -> bool:
        return "foil" in self.finishes.split(",")

    @property
    def has_nonfoil(self) -> bool:
        return "nonfoil" in self.finishes.split(",")

    @property
    def tcgplayer_product_url(self) -> str:
        if self.tcgplayer_id:
            return f"https://www.tcgplayer.com/product/{self.tcgplayer_id}"
        return self.tcgplayer_uri or ""


def _price(v) -> Optional[float]:
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def records_from_card(card: dict, prices_at: str = "") -> list[CardRecord]:
    """One record per face image. Cards without images yield none."""
    faces: list[tuple[int, str, dict, str]] = []
    if "image_uris" in card:
        faces.append((0, card.get("name", ""), card["image_uris"], card.get("illustration_id") or ""))
    for i, face in enumerate(card.get("card_faces") or []):
        if "image_uris" in face:
            faces.append((i, face.get("name", card.get("name", "")), face["image_uris"],
                          face.get("illustration_id") or card.get("illustration_id") or ""))
    if not faces:
        return []
    prices = card.get("prices") or {}
    out = []
    for face_idx, face_name, uris, illus in faces:
        if "small" not in uris:
            continue
        out.append(CardRecord(
            id=card["id"],
            face=face_idx,
            name=card.get("name", ""),
            face_name=face_name,
            set_code=card.get("set", ""),
            set_name=card.get("set_name", ""),
            collector_number=str(card.get("collector_number", "")),
            rarity=card.get("rarity", ""),
            released_at=card.get("released_at", ""),
            illustration_id=illus,
            tcgplayer_id=card.get("tcgplayer_id"),
            tcgplayer_etched_id=card.get("tcgplayer_etched_id"),
            finishes=",".join(card.get("finishes") or []),
            lang=card.get("lang", ""),
            layout=card.get("layout", ""),
            frame=str(card.get("frame", "")),
            border_color=card.get("border_color", ""),
            promo=1 if card.get("promo") else 0,
            digital=1 if card.get("digital") else 0,
            set_type=card.get("set_type", ""),
            image_small=uris.get("small", ""),
            image_normal=uris.get("normal", ""),
            usd=_price(prices.get("usd")),
            usd_foil=_price(prices.get("usd_foil")),
            usd_etched=_price(prices.get("usd_etched")),
            prices_at=prices_at,
            scryfall_uri=card.get("scryfall_uri", ""),
            tcgplayer_uri=(card.get("purchase_uris") or {}).get("tcgplayer", ""),
        ))
    return out
