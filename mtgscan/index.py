"""Card index: SQLite metadata plus a numpy matrix of perceptual hashes.

Built from Scryfall bulk data (every printing, every face image), hashed once from
the 146x204 "small" images and kept in %LOCALAPPDATA%/mtg-screen-scanner. The build
is resumable: rows whose hash is already stored are skipped, so an interrupted run
continues where it stopped and a later run only hashes new printings.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
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


def _migrate(con: sqlite3.Connection) -> None:
    """Add columns introduced after a database was first created.

    Without this an older index would be unreadable, or worse, silently rebuilt from
    scratch. `image_key` is backfilled from the stored URL so existing hashes survive.
    """
    existing = {row[1] for row in con.execute("PRAGMA table_info(cards)")}
    if not existing:
        return
    if "image_key" in existing and "hash_key" in existing:
        return                                   # already migrated: do no work, take no locks
    # Backfilling rewrites every row, which on a full index means a write-ahead log of well
    # over 100 MB. Done as one transaction and checkpointed afterwards, so interrupting it
    # leaves the database exactly as it was rather than part-converted.
    con.execute("BEGIN IMMEDIATE")
    for col in COLUMNS:
        if col not in existing:
            con.execute(f"ALTER TABLE cards ADD COLUMN {col} {_sqltype(col)}")
    if "hash_key" not in existing:
        con.execute("ALTER TABLE cards ADD COLUMN hash_key TEXT")
    if "image_key" not in existing:
        # substr up to '?', or the whole string when there is no query
        con.execute("UPDATE cards SET image_key = CASE WHEN instr(image_small, '?') > 0 "
                    "THEN substr(image_small, 1, instr(image_small, '?') - 1) ELSE image_small END")
        # Hashes already in this database were computed from that same picture, so mark
        # them valid rather than making an upgrade re-download every image.
        con.execute("UPDATE cards SET hash_key = image_key WHERE hash IS NOT NULL")
        log.info("migrated: image_key and hash_key backfilled for existing rows")
    con.commit()
    try:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.Error as e:                   # a busy reader just means it waits for later
        log.debug("post-migration checkpoint skipped: %s", e)


def _init_schema(con: sqlite3.Connection) -> None:
    cols = ", ".join(f"{c} {_sqltype(c)}" for c in COLUMNS)
    con.execute(
        f"CREATE TABLE IF NOT EXISTS cards (rowid INTEGER PRIMARY KEY, {cols}, "
        f"hash BLOB, hash_key TEXT, UNIQUE(id, face))"
    )
    _migrate(con)
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
        # Refreshing card data never touches a stored hash. Whether a hash is still good is
        # decided by comparing hash_key (the picture it was computed from) against
        # image_key, so only a card whose picture actually changed is re-hashed.
        _UPSERT_SQL = (f"INSERT INTO cards ({cols}) VALUES ({qs}) "
                       f"ON CONFLICT(id, face) DO UPDATE SET {updates}")
    return _UPSERT_SQL


# A hash counts as current only while it matches the picture now on record.
STALE_HASH_SQL = "hash IS NULL OR hash_key IS NOT image_key"


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
    stopped = False
    if not opts.skip_hashing:
        scryfall.set_image_rate(opts.image_rate)
        # Anything already hashed in a previous run is in the exported arrays. Putting it
        # back costs a second and saves re-downloading images we have already seen.
        restored = restore_hashes_from_arrays(con, data_dir)
        if restored:
            report(Progress("hashes", 0, 0, f"restored {restored} hashes from the existing index"))
        todo = con.execute(f"SELECT rowid, image_small, image_key FROM cards "
                           f"WHERE image_small != '' AND ({STALE_HASH_SQL})").fetchall()
        total = len(todo)
        done = 0
        t0 = time.monotonic()
        report(Progress("hashes", 0, total, f"hashing {total} card images"))
        with ThreadPoolExecutor(max_workers=max(1, opts.workers)) as ex:
            for chunk in _chunks(todo, 200):
                if stop is not None and stop.is_set():
                    stopped = True
                    break
                results = list(ex.map(lambda row: (row[0], row[2], _fetch_and_hash(row[1])), chunk))
                updates = [(h, key, rowid) for rowid, key, h in results if h is not None]
                if updates:
                    con.executemany("UPDATE cards SET hash = ?, hash_key = ? WHERE rowid = ?", updates)
                    con.commit()
                done += len(chunk)
                elapsed = time.monotonic() - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (total - done) / rate if rate > 0 else 0.0
                report(Progress("hashes", done, total, f"{done}/{total} hashed, {rate:.1f}/s, ~{eta / 60:.0f} min left",
                                {"rate": rate, "eta_s": eta}))

    # Phase 3: export arrays ------------------------------------------------------
    # A stopped build must not publish a half-filled set of arrays over a complete index:
    # the hashes it had already nulled would be gone and the app would start with a
    # near-empty index. The database keeps the partial work, so resuming loses nothing.
    if stopped:
        con.close()
        report(Progress("stopped", 0, 0, "stopped; the existing index was left in place"))
        return
    report(Progress("export", 0, 0, "writing hash arrays"))
    exported = export_arrays(con, data_dir)
    _set_meta(con, "built_at", dt.datetime.now().isoformat(timespec="seconds"))
    con.close()
    report(Progress("done", 1, 1, f"index ready: {exported:,} printings"))


def restore_hashes_from_arrays(con: sqlite3.Connection, data_dir: Path) -> int:
    """Put hashes from the exported arrays back into rows that have lost theirs.

    A repair path. The arrays were written by a build of this same database and row ids
    are stable across refreshes, so a row missing its hash can take it back instead of the
    image being downloaded again. Rows that already have a current hash are left alone.
    """
    data_dir = Path(data_dir)
    try:
        hashes = np.load(data_dir / HASHES_NAME)
        rowids = np.load(data_dir / ROWIDS_NAME)
    except (OSError, ValueError):
        return 0
    if hashes.shape[0] == 0 or hashes.shape[0] != rowids.shape[0]:
        return 0
    missing = {r[0]: r[1] for r in con.execute(
        f"SELECT rowid, image_key FROM cards WHERE {STALE_HASH_SQL}")}
    if not missing:
        return 0
    updates = [(hashes[i].tobytes(), missing[int(rowid)], int(rowid))
               for i, rowid in enumerate(rowids.tolist()) if int(rowid) in missing]
    if not updates:
        return 0
    con.executemany("UPDATE cards SET hash = ?, hash_key = ? WHERE rowid = ?", updates)
    con.commit()
    return len(updates)


def plan_build(data_dir: Path) -> dict:
    """What a build would do, without downloading anything."""
    data_dir = Path(data_dir)
    if not (data_dir / DB_NAME).exists():
        return {"rows": 0, "current": 0, "to_hash": 0, "exists": False, "data_dir": str(data_dir)}
    con = _connect(data_dir / DB_NAME)
    _init_schema(con)
    rows = con.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
    stale = con.execute(f"SELECT COUNT(*) FROM cards WHERE image_small != '' AND ({STALE_HASH_SQL})").fetchone()[0]
    recoverable = 0
    try:
        arrays = np.load(data_dir / ROWIDS_NAME).shape[0]
    except (OSError, ValueError):
        arrays = 0
    con.close()
    return {"rows": rows, "current": rows - stale, "to_hash": stale, "exists": True,
            "exported": arrays, "recoverable": recoverable, "data_dir": str(data_dir)}


# --- Prebuilt index bundles ---------------------------------------------------------
# Hashing every printing means downloading ~107k thumbnails at Scryfall's pace, which takes
# about an hour and a half and repeats that traffic for every user. The result is small
# enough to publish instead: the hashes are 6.8 MB and the card data compresses to about
# 8 MB, so a bundle replaces the whole download. Only hashes and card data travel, never
# card images.

BUNDLE_FORMAT = 1
BUNDLE_MANIFEST = "manifest.json"
BUNDLE_CARDS = "cards.csv.gz"
BUNDLE_HASHES = "hashes.npy"
# Published alongside each release; override with MTGSCAN_BUNDLE_URL.
BUNDLE_URL = "https://github.com/crande25/mtg-screen-scanner/releases/latest/download/card-index.mtgidx"


def bundle_url() -> str:
    return os.environ.get("MTGSCAN_BUNDLE_URL") or BUNDLE_URL


def export_bundle(data_dir: Path, out_path: Path, progress: ProgressCb | None = None) -> dict:
    """Write a shareable index bundle holding every current hash and its card data."""
    import csv
    import gzip
    import io
    import zipfile

    data_dir, out_path = Path(data_dir), Path(out_path)
    con = _connect(data_dir / DB_NAME)
    _init_schema(con)
    rows = con.execute(
        f"SELECT {', '.join(COLUMNS)}, hash FROM cards "
        f"WHERE hash IS NOT NULL AND NOT ({STALE_HASH_SQL}) ORDER BY rowid").fetchall()
    if not rows:
        con.close()
        raise ValueError("this index has no current hashes to export")

    text = io.StringIO()
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(COLUMNS)
    blobs = []
    for row in rows:
        writer.writerow(row[:len(COLUMNS)])
        blobs.append(row[len(COLUMNS)])
    hashes = np.frombuffer(b"".join(blobs), dtype=np.uint8).reshape(len(blobs), HASH_BYTES)
    manifest = {
        "format": BUNDLE_FORMAT,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "rows": len(rows),
        "columns": COLUMNS,
        "hash_bytes": HASH_BYTES,
        "prices_at": _get_meta(con, "prices_at"),
        "source": "Scryfall bulk data; hashes and card data only, no card images",
    }
    con.close()

    npy = io.BytesIO()
    np.save(npy, hashes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(BUNDLE_MANIFEST, json.dumps(manifest, indent=1))
        z.writestr(BUNDLE_CARDS, gzip.compress(text.getvalue().encode("utf-8"), 6))
        z.writestr(BUNDLE_HASHES, npy.getvalue(), compress_type=zipfile.ZIP_STORED)
    tmp.replace(out_path)
    if progress:
        progress(Progress("done", 1, 1, f"exported {len(rows):,} printings"))
    manifest["bytes"] = out_path.stat().st_size
    return manifest


def import_bundle(source: str | Path, data_dir: Path, progress: ProgressCb | None = None,
                  stop: threading.Event | None = None) -> dict:
    """Install a prebuilt bundle, from a local file or an https URL."""
    import csv
    import gzip
    import io
    import zipfile

    def report(phase: str, done: int, total: int, msg: str) -> None:
        if progress:
            progress(Progress(phase, done, total, msg))

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    src = str(source)
    local: Path
    if src.startswith("http://") or src.startswith("https://"):
        local = data_dir / "card-index.mtgidx.part"
        report("download", 0, 0, "downloading the prebuilt index")
        with scryfall.session().get(src, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            with open(local, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if stop is not None and stop.is_set():
                        raise RuntimeError("cancelled")
                    f.write(chunk)
                    done += len(chunk)
                    report("download", done, total, "downloading the prebuilt index")
    else:
        local = Path(src)
        if not local.exists():
            raise FileNotFoundError(local)

    with zipfile.ZipFile(local) as z:
        manifest = json.loads(z.read(BUNDLE_MANIFEST))
        if manifest.get("format") != BUNDLE_FORMAT:
            raise ValueError(f"unsupported bundle format {manifest.get('format')}")
        if manifest.get("hash_bytes") != HASH_BYTES:
            raise ValueError("bundle was built with a different hash size")
        hashes = np.load(io.BytesIO(z.read(BUNDLE_HASHES)))
        cards_csv = gzip.decompress(z.read(BUNDLE_CARDS)).decode("utf-8")

    reader = csv.reader(io.StringIO(cards_csv))
    header = next(reader)
    if header != manifest["columns"]:
        raise ValueError("bundle card data does not match its manifest")

    con = _connect(data_dir / DB_NAME)
    _init_schema(con)
    total = manifest["rows"]
    report("import", 0, total, f"installing {total:,} printings")
    sql = _upsert_sql()
    idx_of = {c: i for i, c in enumerate(header)}
    batch, hash_batch, done = [], [], 0
    for i, row in enumerate(reader):
        if stop is not None and stop.is_set():
            con.close()
            raise RuntimeError("cancelled")
        typed = []
        for col, value in zip(header, row):
            if value == "":
                typed.append(None if col in _INT_COLS | _REAL_COLS else "")
            elif col in _INT_COLS:
                typed.append(int(value))
            elif col in _REAL_COLS:
                typed.append(float(value))
            else:
                typed.append(value)
        batch.append(tuple(typed))
        hash_batch.append((hashes[i].tobytes(), row[idx_of["image_key"]],
                           row[idx_of["id"]], int(row[idx_of["face"]])))
        if len(batch) >= 2000:
            con.executemany(sql, batch)
            con.executemany("UPDATE cards SET hash = ?, hash_key = ? WHERE id = ? AND face = ?", hash_batch)
            con.commit()
            done += len(batch)
            batch.clear(); hash_batch.clear()
            report("import", done, total, f"installing {done:,} of {total:,} printings")
    if batch:
        con.executemany(sql, batch)
        con.executemany("UPDATE cards SET hash = ?, hash_key = ? WHERE id = ? AND face = ?", hash_batch)
        con.commit()
        done += len(batch)

    _set_meta(con, "prices_at", manifest.get("prices_at", ""))
    _set_meta(con, "bundle_created", manifest.get("created", ""))
    report("export", 0, 0, "writing hash arrays")
    exported = export_arrays(con, data_dir)
    _set_meta(con, "built_at", dt.datetime.now().isoformat(timespec="seconds"))
    con.close()
    if local.name.endswith(".part"):
        local.unlink(missing_ok=True)
    report("done", 1, 1, f"index ready: {exported:,} printings")
    return {"rows": done, "hashed": exported, "created": manifest.get("created", "")}


def export_arrays(con: sqlite3.Connection, data_dir: Path) -> int:
    rows = con.execute(f"SELECT rowid, hash FROM cards WHERE hash IS NOT NULL "
                       f"AND NOT ({STALE_HASH_SQL}) ORDER BY rowid").fetchall()
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
        # An index built by an older version is missing columns added since. Migrating on
        # open keeps it usable instead of failing with a bare SQL error, or rebuilding.
        _migrate(self.con)
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
    ap.add_argument("--plan", action="store_true", help="report what a build would do, download nothing")
    ap.add_argument("--import-bundle", nargs="?", const="", metavar="PATH_OR_URL",
                    help="install a prebuilt index instead of hashing (no argument: the published one)")
    ap.add_argument("--export-bundle", metavar="PATH", help="write this index as a shareable bundle")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    target = args.data_dir or default_data_dir()

    if args.plan:
        p = plan_build(target)
        print(f"data folder : {p['data_dir']}")
        if not p["exists"]:
            print("no index here yet; a build would start from nothing")
            return 0
        print(f"printings   : {p['rows']:,}")
        print(f"current     : {p['current']:,}")
        print(f"to download : {p['to_hash']:,}" + ("  (nothing to do)" if not p["to_hash"] else ""))
        return 0

    if args.export_bundle:
        info = export_bundle(target, Path(args.export_bundle), progress=_console_progress)
        print(f"wrote {args.export_bundle}: {info['rows']:,} printings, {info['bytes'] / 1e6:.1f} MB")
        return 0

    if args.import_bundle is not None:
        source = args.import_bundle or bundle_url()
        info = import_bundle(source, target, progress=_console_progress)
        print(f"installed {info['hashed']:,} printings from {source}")
        return 0

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
