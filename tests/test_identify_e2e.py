"""End to end: build a one-set index from Scryfall, paste real card images on a frame, identify them.

Needs network on the first run (about 250 small images); the index is cached in tests/.cache.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mtgscan import scryfall
from mtgscan.detect import whole_frame_quad
from mtgscan.identify import Detection, Identifier
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


class _StubOcr:
    """Stands in for RapidOCR so these tests run without the engine installed."""

    available = True

    def __init__(self, lines=()):
        self._lines = list(lines)

    def lines(self, img):
        assert img is not None and img.size > 0
        return self._lines


def test_ocr_fallback_on_an_unmatched_card_does_not_raise(small_index):
    """Regression: the fallback passed a numpy array through `or`, which raises.

    It needs all three conditions at once -- OCR enabled, no hash match, and a card big
    enough to be re-warped for OCR -- and the error was swallowed per card, so the only
    symptom was an empty table with no message.
    """
    rng = np.random.default_rng(3)
    noise = rng.integers(0, 255, (816, 584, 3), dtype=np.uint8)   # unmatchable, and large
    dets = Identifier(small_index, _StubOcr(), use_ocr=True).identify_frame(noise, single_card=True)
    assert len(dets) == 1
    assert dets[0].card is None
    assert not dets[0].note.startswith("error:"), dets[0].note


def test_a_failing_card_is_reported_not_dropped(small_index, monkeypatch):
    """A card that blows up must still appear, so the failure is visible to the user."""
    ident = Identifier(small_index, ocr=None)
    monkeypatch.setattr(ident, "identify_card",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    dets = ident.identify_frame(np.zeros((408, 292, 3), np.uint8), single_card=True)
    assert len(dets) == 1 and dets[0].card is None
    assert dets[0].note == "error: RuntimeError: boom"


def test_card_on_a_dark_background_with_margin_is_found(small_index):
    """Regression: the case that failed on a real screen.

    A card's black border against a dark background is a very weak edge, so the contour
    step returns a skewed quad spanning the frame and the hash, which is intolerant of a
    few percent of misalignment, finds nothing. The edge-projection fallback recovers it.
    """
    rec = _pick(small_index, 3)[1]
    card = _normal_image(rec)
    margin = 60
    h, w = card.shape[:2]
    frame = np.full((h + 2 * margin, w + 2 * margin, 3), (34, 36, 42), np.uint8)  # dark, as on screen
    frame[margin:margin + h, margin:margin + w] = card

    dets = Identifier(small_index, ocr=None).identify_frame(frame)
    assert any(d.card for d in dets), [d.note for d in dets]
    assert any(d.card.id == rec.id for d in dets if d.card), \
        [(d.card.name, d.distance) for d in dets if d.card]


def _film(card: np.ndarray, tilt: float = 0.018, glare: float = 0.45, exposure: float = 0.8,
          height: int = 430, quality: int = 45) -> np.ndarray:
    """Approximate a card seen through a webcam: tilt, glare, exposure, blur, compression."""
    h, w = card.shape[:2]
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    dx, dy = w * tilt, h * tilt
    dst = np.array([[dx, 0], [w - 1, dy], [w - 1 - dx, h - 1], [0, h - 1 - dy]], np.float32)
    out = cv2.warpPerspective(card, cv2.getPerspectiveTransform(src, dst), (w, h))
    gx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    gy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    hot = np.exp(-((gx - 0.3) ** 2 + (gy + 0.4) ** 2) * 3.0) * glare
    out = np.clip(out.astype(np.float32) * exposure * (1.0 + hot[..., None]), 0, 255).astype(np.uint8)
    out = cv2.GaussianBlur(out, (3, 3), 0)
    out = cv2.resize(out, (int(height * w / h), height), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def test_a_filmed_card_is_identified(small_index):
    """A card on a live stream lands far past the plain cutoff but stands clear of the field.

    Modelled on a real auction stream, where the correct card scored 106 against a cutoff
    of 112 while the nearest different artwork was 58 bits behind it.
    """
    rec = _pick(small_index, 3)[1]
    filmed = _film(_normal_image(rec))
    dets = Identifier(small_index, ocr=None).identify_frame(filmed, single_card=True)
    assert dets and dets[0].card is not None, dets[0].note
    assert dets[0].card.id == rec.id, (dets[0].card.name, dets[0].distance, dets[0].margin)


def test_single_card_mode_still_finds_a_card_inside_the_region(small_index):
    """"Region is one card" must be a hint, not a promise.

    A hand-held card on a stream drifts inside a fixed region, so the region often holds a
    card plus a good deal of other content. Treating the whole region as the card then fails
    every time, which is exactly what was reported from a live auction stream.
    """
    rec = _pick(small_index, 4)[2]
    card = _normal_image(rec)
    h, w = card.shape[:2]
    frame = np.full((int(h * 1.7), int(w * 1.6), 3), (30, 30, 30), np.uint8)
    frame[40:40 + h, 70:70 + w] = card          # card is only part of the region

    dets = Identifier(small_index, ocr=None).identify_frame(frame, single_card=True)
    assert any(d.card for d in dets), [d.note for d in dets]
    hit = next(d for d in dets if d.card)
    assert hit.card.id == rec.id, (hit.card.name, hit.distance)
    assert "found inside the region" in hit.note


def test_noise_is_still_rejected_under_the_margin_rule(small_index):
    """The relaxed ceiling must not let content that is not a card through."""
    rng = np.random.default_rng(11)
    for seed_shift in range(3):
        noise = rng.integers(0, 255, (430 + seed_shift, 308, 3), dtype=np.uint8)
        dets = Identifier(small_index, ocr=None).identify_frame(noise, single_card=True)
        assert dets[0].card is None, (dets[0].card.name, dets[0].distance, dets[0].margin)


def test_skewed_quads_are_rejected():
    from mtgscan.detect import quad_is_sane

    upright = np.array([[60, 60], [547, 60], [547, 739], [60, 739]], np.float32)
    assert quad_is_sane(upright)
    # a mild perspective view of a card is still acceptable
    tilted = np.array([[64, 58], [549, 66], [543, 737], [58, 731]], np.float32)
    assert quad_is_sane(tilted)
    # the real failure: a lopsided quad whose averaged sides looked card-shaped
    skewed = np.array([[36, 86], [607, 39], [518, 718], [89, 724]], np.float32)
    assert not quad_is_sane(skewed)


def test_scan_summary_explains_an_empty_result():
    from mtgscan.ui import ScannerApp

    assert "no card shapes found" in ScannerApp._scan_summary([], 0.01).lower()
    err = Detection(quad=whole_frame_quad(np.zeros((10, 10, 3), np.uint8)),
                    image=np.zeros((1, 1, 3), np.uint8), note="error: RuntimeError: boom")
    assert "boom" in ScannerApp._scan_summary([err], 0.01)


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
