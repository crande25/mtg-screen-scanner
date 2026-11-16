"""Index upkeep: staleness, repair, migration and prebuilt bundles.

All of this came out of one incident. A rebuild re-downloaded every image because hashes
were keyed on an image URL carrying a cache-busting timestamp, and stopping that rebuild
would have published its half-finished arrays over a complete index. These tests build
small databases from fabricated records, so they need no network.
"""
import sqlite3

import numpy as np
import pytest

from mtgscan import index as idx_mod
from mtgscan.config import HASH_BYTES
from mtgscan.index import (DB_NAME, HASHES_NAME, HashIndex, ROWIDS_NAME, STALE_HASH_SQL,
                           _connect, _init_schema, _upsert, export_arrays, export_bundle,
                           import_bundle, plan_build, restore_hashes_from_arrays)
from mtgscan.scryfall import CardRecord, image_key


def make_record(card_id: str, stamp: str = "1111", name: str = "Test Card") -> CardRecord:
    url = f"https://cards.scryfall.io/small/front/a/b/{card_id}.jpg?{stamp}"
    values = {f: "" for f in CardRecord.columns()}
    values.update(id=card_id, face=0, name=name, face_name=name, set_code="tst",
                  set_name="Test", collector_number="1", illustration_id=f"art-{card_id}",
                  finishes="nonfoil", lang="en", promo=0, digital=0,
                  tcgplayer_id=None, tcgplayer_etched_id=None,
                  usd=None, usd_foil=None, usd_etched=None,
                  image_small=url, image_key=image_key(url), image_normal=url)
    return CardRecord(**values)


def seeded_db(path, records, hashed=True):
    con = _connect(path / DB_NAME)
    _init_schema(con)
    _upsert(con, records)
    if hashed:
        rng = np.random.default_rng(0)
        for rec in records:
            h = rng.integers(0, 255, HASH_BYTES, dtype=np.uint8).tobytes()
            con.execute("UPDATE cards SET hash = ?, hash_key = ? WHERE id = ? AND face = ?",
                        (h, rec.image_key, rec.id, rec.face))
        con.commit()
    return con


def stale_count(con) -> int:
    return con.execute(f"SELECT COUNT(*) FROM cards WHERE {STALE_HASH_SQL}").fetchone()[0]


class TestStaleness:
    def test_a_new_cache_busting_timestamp_does_not_invalidate_anything(self, tmp_path):
        """The actual bug: Scryfall changes ...jpg?<stamp> on every bulk export."""
        records = [make_record(f"c{i}") for i in range(5)]
        con = seeded_db(tmp_path, records)
        assert stale_count(con) == 0
        refreshed = [make_record(r.id, stamp="9999") for r in records]
        assert refreshed[0].image_small != records[0].image_small    # URL really did change
        assert refreshed[0].image_key == records[0].image_key        # the picture did not
        _upsert(con, refreshed)
        assert stale_count(con) == 0, "a changed timestamp must not force a re-download"

    def test_a_genuinely_different_picture_invalidates_just_that_row(self, tmp_path):
        records = [make_record(f"c{i}") for i in range(5)]
        con = seeded_db(tmp_path, records)
        rescanned = make_record("c2")
        new_url = "https://cards.scryfall.io/small/front/a/b/c2-v2.jpg?1"
        object.__setattr__(rescanned, "image_small", new_url)
        object.__setattr__(rescanned, "image_key", image_key(new_url))
        _upsert(con, [rescanned])
        assert stale_count(con) == 1

    def test_refreshing_card_data_updates_fields_without_touching_hashes(self, tmp_path):
        con = seeded_db(tmp_path, [make_record("c1", name="Old Name")])
        _upsert(con, [make_record("c1", stamp="2222", name="New Name")])
        name, = con.execute("SELECT name FROM cards WHERE id = 'c1'").fetchone()
        assert name == "New Name"
        assert stale_count(con) == 0


class TestRepair:
    def test_hashes_are_restored_from_the_exported_arrays(self, tmp_path):
        records = [make_record(f"c{i}") for i in range(6)]
        con = seeded_db(tmp_path, records)
        export_arrays(con, tmp_path)
        before = {r[0]: r[1] for r in con.execute("SELECT rowid, hash FROM cards")}

        con.execute("UPDATE cards SET hash = NULL, hash_key = NULL")   # the damaged state
        con.commit()
        assert stale_count(con) == 6

        assert restore_hashes_from_arrays(con, tmp_path) == 6
        assert stale_count(con) == 0
        after = {r[0]: r[1] for r in con.execute("SELECT rowid, hash FROM cards")}
        assert after == before, "restored hashes must be the ones that were exported"

    def test_restoring_is_a_no_op_when_nothing_is_missing(self, tmp_path):
        con = seeded_db(tmp_path, [make_record("c1")])
        export_arrays(con, tmp_path)
        assert restore_hashes_from_arrays(con, tmp_path) == 0

    def test_export_skips_stale_rows(self, tmp_path):
        records = [make_record(f"c{i}") for i in range(4)]
        con = seeded_db(tmp_path, records)
        con.execute("UPDATE cards SET hash_key = 'something-else' WHERE id = 'c1'")
        con.commit()
        assert export_arrays(con, tmp_path) == 3


class TestMigration:
    def test_an_index_from_an_older_version_still_opens(self, tmp_path):
        """Older databases have neither image_key nor hash_key; their hashes must survive."""
        cols = [c for c in CardRecord.columns() if c not in ("image_key",)]
        con = sqlite3.connect(str(tmp_path / DB_NAME))
        con.execute(f"CREATE TABLE cards (rowid INTEGER PRIMARY KEY, "
                    f"{', '.join(f'{c} TEXT' for c in cols)}, hash BLOB, UNIQUE(id, face))")
        url = "https://cards.scryfall.io/small/front/a/b/old.jpg?4242"
        con.execute(f"INSERT INTO cards ({', '.join(cols)}, hash) VALUES "
                    f"({', '.join('?' for _ in cols)}, ?)",
                    [("old" if c == "id" else "0" if c == "face" else "") for c in cols]
                    + [b"\x01" * HASH_BYTES])
        con.execute("UPDATE cards SET image_small = ?", (url,))
        con.commit()
        con.close()
        np.save(tmp_path / HASHES_NAME, np.zeros((1, HASH_BYTES), np.uint8))
        np.save(tmp_path / ROWIDS_NAME, np.array([1], np.int64))

        index = HashIndex(tmp_path)          # opening must migrate, not explode
        migrated = index.con.execute("SELECT image_key, hash_key FROM cards").fetchone()
        assert migrated[0] == image_key(url)
        assert migrated[1] == migrated[0], "an existing hash must be kept, not re-downloaded"
        assert stale_count(index.con) == 0


class TestPlan:
    def test_it_reports_nothing_to_do_for_a_complete_index(self, tmp_path):
        con = seeded_db(tmp_path, [make_record(f"c{i}") for i in range(3)])
        export_arrays(con, tmp_path)
        con.close()
        plan = plan_build(tmp_path)
        assert plan["rows"] == 3 and plan["current"] == 3 and plan["to_hash"] == 0

    def test_it_counts_what_needs_downloading(self, tmp_path):
        con = seeded_db(tmp_path, [make_record(f"c{i}") for i in range(3)], hashed=False)
        con.close()
        assert plan_build(tmp_path)["to_hash"] == 3

    def test_an_empty_folder_is_reported_as_having_no_index(self, tmp_path):
        assert plan_build(tmp_path)["exists"] is False


class TestBundle:
    def test_round_trip_reproduces_the_index(self, tmp_path):
        source, dest = tmp_path / "src", tmp_path / "dst"
        source.mkdir(); dest.mkdir()
        records = [make_record(f"c{i}", name=f"Card {i}") for i in range(7)]
        con = seeded_db(source, records)
        export_arrays(con, source)
        original = np.load(source / HASHES_NAME)
        con.close()

        bundle = tmp_path / "index.mtgidx"
        info = export_bundle(source, bundle)
        assert info["rows"] == 7 and bundle.exists()

        result = import_bundle(bundle, dest)
        assert result["hashed"] == 7
        assert np.array_equal(np.load(dest / HASHES_NAME), original)

        installed = HashIndex(dest)
        assert len(installed) == 7
        assert {c.name for c in installed.cards_by_name("Card 3")} == {"Card 3"}
        assert plan_build(dest)["to_hash"] == 0, "an imported index must need no downloads"

    def test_an_index_with_no_hashes_cannot_be_exported(self, tmp_path):
        con = seeded_db(tmp_path, [make_record("c1")], hashed=False)
        con.close()
        with pytest.raises(ValueError, match="no current hashes"):
            export_bundle(tmp_path, tmp_path / "x.mtgidx")

    def test_the_url_can_be_overridden_for_testing(self, monkeypatch):
        monkeypatch.setenv("MTGSCAN_BUNDLE_URL", "https://example.invalid/x.mtgidx")
        assert idx_mod.bundle_url() == "https://example.invalid/x.mtgidx"
