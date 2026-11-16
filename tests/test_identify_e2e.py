"""End to end: build a one-set index from Scryfall, paste real card images on a frame, identify them.

Needs network on the first run (about 250 small images); the index is cached in tests/.cache.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mtgscan import scryfall
from mtgscan.identify import Identifier
from mtgscan.index import BuildOptions, HashIndex, build_index
from mtgscan.pricing import ScryfallMarketProvider
from tests.test_detect import paste, paste_perspective

CACHE = Path(__file__).parent / ".cache"
INDEX_DIR = CACHE / "index-m10"


@pytest.fixture(scope="session")
def small_index() -> HashIndex:
    try:
        if not HashIndex.exists(INDEX_DIR):
            build_index(INDEX_DIR, BuildOptions(query="set:m10", workers=4))
        idx = HashIndex(INDEX_DIR)
    except Exception as e:  # pragma: no cover - offline
        pytest.skip(f"could not build test index: {e}")
    if len(idx) < 100:
        pytest.skip("test index too small")
    return idx


def _normal_image(rec) -> np.ndarray:
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"{rec.id}_{rec.face}_normal.jpg"
    if not p.exists():
        p.write_bytes(scryfall.download_image(rec.image_normal))
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    assert img is not None
    return img


def _pick(idx: HashIndex, n: int):
    rows = [int(r) for r in idx.rowids]
    step = max(1, len(rows) // n)
    return [idx.card(rows[i * step]) for i in range(n)]


def test_index_stats(small_index):
    st = small_index.stats()
    assert st["hashed"] >= 200
    assert small_index.card(int(small_index.rowids[0])).set_code == "m10"


def test_identify_cards_on_a_frame(small_index):
    recs = _pick(small_index, 4)
    imgs = [_normal_image(r) for r in recs]
    frame = np.full((900, 1400, 3), (235, 235, 235), np.uint8)
    paste(frame, imgs[0], 40, 40, 0.5)
    paste(frame, imgs[1], 340, 60, 0.35)
    paste_perspective(frame, imgs[2], np.array([[620, 80], [900, 110], [880, 520], [600, 480]], np.float32))
    side = cv2.rotate(cv2.resize(imgs[3], None, fx=0.45, fy=0.45), cv2.ROTATE_90_CLOCKWISE)
    paste(frame, side, 960, 300)

    dets = Identifier(small_index, ocr=None).identify_frame(frame)
    found = {d.card.id: d for d in dets if d.card}
    missing = [r.name for r in recs if r.id not in found]
    assert not missing, (missing, [(d.name, d.distance) for d in dets])
    for r in recs:
        assert found[r.id].confidence > 0.7, (r.name, found[r.id].distance)


def test_single_card_mode(small_index):
    rec = _pick(small_index, 3)[1]
    img = _normal_image(rec)
    dets = Identifier(small_index, ocr=None).identify_frame(img, single_card=True)
    assert len(dets) == 1 and dets[0].card is not None
    assert dets[0].card.id == rec.id, (dets[0].name, rec.name)


def test_upside_down_card_is_still_found(small_index):
    rec = _pick(small_index, 5)[4]
    img = cv2.rotate(_normal_image(rec), cv2.ROTATE_180)
    dets = Identifier(small_index, ocr=None).identify_frame(img, single_card=True)
    assert dets and dets[0].card is not None and dets[0].card.id == rec.id


def test_unrelated_image_is_not_identified(small_index):
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 255, (408, 292, 3), dtype=np.uint8)
    dets = Identifier(small_index, ocr=None).identify_frame(noise, single_card=True)
    assert dets and dets[0].card is None, (dets[0].name, dets[0].distance)


def test_scryfall_price_provider(small_index):
    rec = _pick(small_index, 2)[0]
    q = ScryfallMarketProvider().get(rec)
    assert q.source
    assert q.market is not None or q.market_foil is not None
    assert q.low is None and q.low_kind == ""
    assert q.url.startswith("http")


def test_tcgcsv_price_provider(small_index, tmp_path):
    from mtgscan.pricing import TcgcsvProvider

    bolt = [c for c in small_index.cards_by_name("Lightning Bolt") if c.set_code == "m10"]
    assert bolt, "Lightning Bolt (M10) missing from the test index"
    q = TcgcsvProvider(cache_dir=tmp_path).get(bolt[0])
    assert q.low_kind == "any condition"
    assert q.low is not None and q.market is not None and q.low <= q.market
    assert q.low_foil is not None and q.market_foil is not None
    assert "tcgcsv" in q.source


def test_price_service_falls_back_per_card(small_index, tmp_path):
    """A card tcgcsv cannot map falls back to Scryfall market without disabling the provider."""
    from dataclasses import replace

    from mtgscan.pricing import MODE_TCGCSV, PriceService

    rec = _pick(small_index, 2)[1]
    svc = PriceService(mode=MODE_TCGCSV, cache_dir=tmp_path)
    orphan = replace(rec, tcgplayer_id=None)   # same Scryfall id, no TCGplayer product
    q = svc.get(orphan)
    assert q.note.startswith("fallback"), q
    assert q.low is None and (q.market is not None or q.market_foil is not None)
    assert svc._disabled_until == {}
    # the provider is still live for the next card
    svc2 = PriceService(mode=MODE_TCGCSV, cache_dir=tmp_path)
    good = svc2.get(rec)
    assert good.low is not None and good.note == "", good
