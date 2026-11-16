"""Card index: SQLite metadata plus a numpy matrix of perceptual hashes.

Built from Scryfall bulk data (every printing, every face image), hashed once from
the 146x204 "small" images and kept in %LOCALAPPDATA%/mtg-screen-scanner. The build
is resumable: rows whose hash is already stored are skipped, so an interrupted run
continues where it stopped and a later run only hashes new printings.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import cv2
import numpy as np

from . import scryfall
from .config import HASH_BYTES, data_dir as default_data_dir
from .hashing import card_hash, hamming_many
from .scryfall import CardRecord, records_from_card

log = logging.getLogger(__name__)

COLUMNS = CardRecord.columns()
_INT_COLS = {"face", "tcgplayer_id", "tcgplayer_etched_id", "promo", "digital"}
_REAL_COLS = {"usd", "usd_foil", "usd_etched"}
DB_NAME = "cards.sqlite"
HASHES_NAME = "hashes.npy"
ROWIDS_NAME = "rowids.npy"


def _sqltype(col: str) -> str:
    if col in _INT_COLS:
        return "INTEGER"
    if col in _REAL_COLS:
        return "REAL"
    return "TEXT"


def _connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path), check_same_thread=False, timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def _init_schema(con: sqlite3.Connection) -> None:
    cols = ", ".join(f"{c} {_sqltype(c)}" for c in COLUMNS)
    con.execute(
        f"CREATE TABLE IF NOT EXISTS cards (rowid INTEGER PRIMARY KEY, {cols}, hash BLOB, UNIQUE(id, face))"
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_cards_name ON cards(name)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_cards_face_name ON cards(face_name)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_cards_illus ON cards(illustration_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_cards_hash_null ON cards(hash) WHERE hash IS NULL")
    con.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    con.commit()


def _set_meta(con: sqlite3.Connection, key: str, value: str) -> None:
    con.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))
    con.commit()


def _get_meta(con: sqlite3.Connection, key: str, default: str = "") -> str:
    row = con.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


# --- Build ----------------------------------------------------------------------

@dataclass
class BuildOptions:
    bulk_type: str = "default_cards"
    sets: Optional[set[str]] = None          # lowercase set codes to keep
    since: Optional[str] = None              # keep printings released on/after YYYY[-MM-DD]
    langs: tuple[str, ...] = ("en",)
    include_digital: bool = False            # Arena/MTGO-only objects have no TCGplayer listing
    include_tokens: bool = True
    include_art_series: bool = False
    workers: int = 8
    image_rate: float = 20.0                 # image downloads per second from the Scryfall CDN
    limit: Optional[int] = None              # cap on cards (for tests)
    query: Optional[str] = None              # Scryfall search instead of bulk data (small indexes)
    skip_hashing: bool = False


@dataclass
class Progress:
    phase: str          # download | cards | hashes | export | done
    done: int = 0
    total: int = 0
    message: str = ""
    extra: dict = field(default_factory=dict)


ProgressCb = Callable[[Progress], None]


def _keep(card: dict, opts: BuildOptions) -> bool:
    if opts.langs and card.get("lang") not in opts.langs:
        return False
    if not opts.include_digital and card.get("digital"):
        return False
    layout = card.get("layout", "")
    set_type = card.get("set_type", "")
    if not opts.include_tokens and (set_type == "token" or layout in ("token", "double_faced_token", "emblem")):
        return False
    if not opts.include_art_series and layout == "art_series":
        return False
    if opts.sets and card.get("set", "").lower() not in opts.sets:
        return False
    if opts.since and (card.get("released_at") or "0000") < opts.since:
        return False
    return True


_UPSERT_SQL = None


def _upsert_sql() -> str:
    global _UPSERT_SQL
    if _UPSERT_SQL is None:
        cols = ", ".join(COLUMNS)
        qs = ", ".join("?" for _ in COLUMNS)
        updates = ", ".join(f"{c} = excluded.{c}" for c in COLUMNS if c not in ("id", "face"))
        # A changed image URL means Scryfall rescanned the card: drop the stale hash.
        _UPSERT_SQL = (
            f"INSERT INTO cards ({cols}) VALUES ({qs}) ON CONFLICT(id, face) DO UPDATE SET {updates}, "
            f"hash = CASE WHEN excluded.image_small IS NOT cards.image_small THEN NULL ELSE cards.hash END"
        )
    return _UPSERT_SQL


def _upsert(con: sqlite3.Connection, recs: list[CardRecord]) -> None:
    if not recs:
        return
    con.executemany(_upsert_sql(), [tuple(getattr(r, c) for c in COLUMNS) for r in recs])
    con.commit()


def _hash_image_bytes(data: bytes) -> Optional[bytes]:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return None
    return card_hash(img).tobytes()


def _fetch_and_hash(url: str) -> Optional[bytes]:
    try:
        return _hash_image_bytes(scryfall.download_image(url))
    except Exception as e:  # network or decode problems: leave NULL, retried on the next build
        log.warning("hash failed for %s: %s", url, e)
        return None


def _chunks(items: list, n: int) -> Iterator[list]:
    for i in range(0, len(items), n):
        yield items[i:i + n]


def build_index(data_dir: Path, opts: BuildOptions, progress: ProgressCb | None = None,
                stop: threading.Event | None = None) -> None:
    """Build or update the index. Safe to re-run; only new/changed printings are hashed."""
    def report(p: Progress) -> None:
        if progress:
            progress(p)

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    con = _connect(data_dir / DB_NAME)
    _init_schema(con)

    # Phase 1: card metadata --------------------------------------------------
    if opts.query:
        cards: Iterable[dict] = scryfall.search_cards(opts.query)
        prices_at = dt.date.today().isoformat()
        report(Progress("cards", 0, 0, f"searching Scryfall: {opts.query}"))
    else:
        def dl_progress(done: int, total: int) -> None:
            report(Progress("download", done, total, "downloading Scryfall bulk data"))

        bulk_path = scryfall.download_bulk(opts.bulk_type, data_dir / "bulk", progress=dl_progress)
        prices_at = scryfall.bulk_updated_at(bulk_path)
        cards = scryfall.iter_bulk_cards(bulk_path)
        report(Progress("cards", 0, 0, "reading card data"))

    n_cards = 0
    batch: list[CardRecord] = []
    for card in cards:
        if stop is not None and stop.is_set():
            return
        if not _keep(card, opts):
            continue
        batch.extend(records_from_card(card, prices_at))
        n_cards += 1
        if opts.limit and n_cards >= opts.limit:
            break
        if len(batch) >= 2000:
            _upsert(con, batch)
            batch.clear()
            report(Progress("cards", n_cards, 0, f"{n_cards} printings read"))
    _upsert(con, batch)
    _set_meta(con, "prices_at", prices_at)
    _set_meta(con, "cards_updated_at", dt.datetime.now().isoformat(timespec="seconds"))
    report(Progress("cards", n_cards, n_cards, f"{n_cards} printings read"))

    # Phase 2: hashes -----------------------------------------------------------
    if not opts.skip_hashing:
        scryfall.set_image_rate(opts.image_rate)
        todo = con.execute("SELECT rowid, image_small FROM cards WHERE hash IS NULL AND image_small != ''").fetchall()
        total = len(todo)
        done = 0
        t0 = time.monotonic()
        report(Progress("hashes", 0, total, f"hashing {total} card images"))
        with ThreadPoolExecutor(max_workers=max(1, opts.workers)) as ex:
            for chunk in _chunks(todo, 200):
                if stop is not None and stop.is_set():
                    break
                results = list(ex.map(lambda row: (row[0], _fetch_and_hash(row[1])), chunk))
                updates = [(h, rowid) for rowid, h in results if h is not None]
                if updates:
                    con.executemany("UPDATE cards SET hash = ? WHERE rowid = ?", updates)
                    con.commit()
                done += len(chunk)
                elapsed = time.monotonic() - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (total - done) / rate if rate > 0 else 0.0
                report(Progress("hashes", done, total, f"{done}/{total} hashed, {rate:.1f}/s, ~{eta / 60:.0f} min left",
                                {"rate": rate, "eta_s": eta}))

    # Phase 3: export arrays ------------------------------------------------------
    report(Progress("export", 0, 0, "writing hash arrays"))
    export_arrays(con, data_dir)
    _set_meta(con, "built_at", dt.datetime.now().isoformat(timespec="seconds"))
    con.close()
    report(Progress("done", 1, 1, "index ready"))


def export_arrays(con: sqlite3.Connection, data_dir: Path) -> int:
    rows = con.execute("SELECT rowid, hash FROM cards WHERE hash IS NOT NULL ORDER BY rowid").fetchall()
    if rows:
        rowids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
        hashes = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.uint8).reshape(len(rows), HASH_BYTES)
    else:
        rowids = np.zeros(0, dtype=np.int64)
        hashes = np.zeros((0, HASH_BYTES), dtype=np.uint8)
    tmp_h, tmp_r = data_dir / (HASHES_NAME + ".tmp"), data_dir / (ROWIDS_NAME + ".tmp")
    np.save(tmp_h, hashes)
    np.save(tmp_r, rowids)
    # np.save appends .npy to names without that suffix
    Path(str(tmp_h) + ".npy").replace(data_dir / HASHES_NAME)
    Path(str(tmp_r) + ".npy").replace(data_dir / ROWIDS_NAME)
    return len(rows)


# --- Query ----------------------------------------------------------------------

class HashIndex:
    """Read-only view of a built index, usable from several threads."""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else default_data_dir()
        self.con = _connect(self.data_dir / DB_NAME)
        self._lock = threading.Lock()
        self.hashes = np.load(self.data_dir / HASHES_NAME)
        self.rowids = np.load(self.data_dir / ROWIDS_NAME)
        self._cache: dict[int, CardRecord] = {}
        self._names: Optional[list[str]] = None

    @staticmethod
    def exists(data_dir: Path | None = None) -> bool:
        d = Path(data_dir) if data_dir else default_data_dir()
        return (d / DB_NAME).exists() and (d / HASHES_NAME).exists() and (d / ROWIDS_NAME).exists()

    def __len__(self) -> int:
        return int(self.hashes.shape[0])

    def stats(self) -> dict:
        with self._lock:
            total = self.con.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
            return {
                "cards": total,
                "hashed": len(self),
                "built_at": _get_meta(self.con, "built_at"),
                "prices_at": _get_meta(self.con, "prices_at"),
            }

    def query(self, h: np.ndarray, k: int = 12) -> list[tuple[int, int]]:
        """Return up to k (rowid, hamming_distance) pairs, best first."""
        if len(self) == 0:
            return []
        d = hamming_many(h, self.hashes)
        k = min(k, d.shape[0])
        idx = np.argpartition(d, k - 1)[:k]
        idx = idx[np.argsort(d[idx], kind="stable")]
        return [(int(self.rowids[i]), int(d[i])) for i in idx]

    def _row_to_record(self, row: tuple) -> CardRecord:
        return CardRecord(**dict(zip(COLUMNS, row)))

    def card(self, rowid: int) -> CardRecord:
        rec = self._cache.get(rowid)
        if rec is None:
            with self._lock:
                row = self.con.execute(f"SELECT {', '.join(COLUMNS)} FROM cards WHERE rowid = ?", (rowid,)).fetchone()
            if row is None:
                raise KeyError(rowid)
            rec = self._row_to_record(row)
            self._cache[rowid] = rec
        return rec

    def _select(self, where: str, params: tuple) -> list[CardRecord]:
        with self._lock:
            rows = self.con.execute(f"SELECT {', '.join(COLUMNS)} FROM cards WHERE {where}", params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def cards_by_name(self, name: str) -> list[CardRecord]:
        return self._select("name = ? OR face_name = ?", (name, name))

    def cards_by_illustration(self, illustration_id: str) -> list[CardRecord]:
        if not illustration_id:
            return []
        return self._select("illustration_id = ?", (illustration_id,))

    def cards_by_scryfall_id(self, card_id: str) -> list[CardRecord]:
        return self._select("id = ?", (card_id,))

    def unique_names(self) -> list[str]:
        """Distinct card names for title matching, excluding tokens and emblems.

        A token often carries the short form of a real card's name ("Ragavan" for
        "Ragavan, Nimble Pilferer"), so a partially read title matches the token exactly
        and the wrong card is reported with full confidence.
        """
        if self._names is None:
            with self._lock:
                rows = self.con.execute(
                    "SELECT DISTINCT face_name FROM cards WHERE face_name != '' "
                    "AND layout NOT IN ('token', 'double_faced_token', 'emblem') "
                    "AND set_type != 'token'").fetchall()
            self._names = [r[0] for r in rows]
        return self._names


# --- CLI ----------------------------------------------------------------------------

def _console_progress(p: Progress) -> None:
    if p.phase == "download" and p.total:
        sys.stderr.write(f"\r[download] {p.done / 1e6:,.0f} / {p.total / 1e6:,.0f} MB")
    elif p.phase == "hashes" and p.total:
        sys.stderr.write(f"\r[hashes] {p.message}          ")
    else:
        sys.stderr.write(f"\r[{p.phase}] {p.message}          ")
    if p.phase == "done":
        sys.stderr.write("\n")
    sys.stderr.flush()


def cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mtgscan-build-index",
                                 description="Build or update the card hash index from Scryfall.")
    ap.add_argument("--data-dir", type=Path, default=None, help="where to keep the index (default: %%LOCALAPPDATA%%/mtg-screen-scanner)")
    ap.add_argument("--sets", default=None, help="comma separated set codes to include, e.g. mh3,blb,dsk")
    ap.add_argument("--since", default=None, help="only printings released on/after this date (YYYY or YYYY-MM-DD)")
    ap.add_argument("--query", default=None, help="Scryfall search query instead of bulk data (small/test indexes)")
    ap.add_argument("--include-digital", action="store_true", help="also index Arena/MTGO-only cards")
    ap.add_argument("--skip-tokens", action="store_true")
    ap.add_argument("--include-art-series", action="store_true")
    ap.add_argument("--workers", type=int, default=8, help="parallel image downloads")
    ap.add_argument("--rate", type=float, default=20.0, help="image downloads per second (Scryfall CDN)")
    ap.add_argument("--limit", type=int, default=None, help="stop after this many printings (testing)")
    ap.add_argument("--skip-hashing", action="store_true", help="only refresh card metadata and prices")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    opts = BuildOptions(
        sets={s.strip().lower() for s in args.sets.split(",")} if args.sets else None,
        since=args.since,
        include_digital=args.include_digital,
        include_tokens=not args.skip_tokens,
        include_art_series=args.include_art_series,
        workers=args.workers,
        image_rate=args.rate,
        limit=args.limit,
        query=args.query,
        skip_hashing=args.skip_hashing,
    )
    build_index(args.data_dir or default_data_dir(), opts, progress=_console_progress)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli())
