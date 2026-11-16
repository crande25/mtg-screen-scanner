import itertools

import cv2
import numpy as np

from mtgscan.config import HASH_BITS, MATCH_ACCEPT, MATCH_STRONG
from mtgscan.hashing import card_hash, hamming, hamming_many, hash_from_hex, hash_to_hex, frame_signature, frames_differ
from tests.conftest import load


def test_hash_roundtrip_hex():
    img = np.random.default_rng(0).integers(0, 255, (204, 146, 3), dtype=np.uint8)
    h = card_hash(img)
    assert h.shape == (HASH_BITS // 8,)
    assert np.array_equal(hash_from_hex(hash_to_hex(h)), h)


def test_same_card_small_vs_normal_is_close(samples):
    """The index hashes the 146x204 image; the screen shows some other scale of the same art."""
    worst = 0
    for s in samples:
        a = card_hash(load(s["small"]))
        b = card_hash(load(s["normal"]))
        d = hamming(a, b)
        worst = max(worst, d)
    assert worst <= MATCH_STRONG * HASH_BITS, worst


def test_same_card_survives_jpeg_and_downscale(samples):
    worst = 0
    for s in samples:
        ref = card_hash(load(s["small"]))
        big = load(s["normal"])
        tiny = cv2.resize(big, (100, 140), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", tiny, [cv2.IMWRITE_JPEG_QUALITY, 60])
        tiny = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        worst = max(worst, hamming(ref, card_hash(tiny)))
    assert worst <= MATCH_ACCEPT * HASH_BITS, worst


def test_different_cards_are_far(samples):
    hashes = [card_hash(load(s["small"])) for s in samples]
    best, pair = HASH_BITS, None
    for (i, a), (j, b) in itertools.combinations(enumerate(hashes), 2):
        d = hamming(a, b)
        if d < best:
            best, pair = d, (samples[i]["name"], samples[j]["name"])
    # unrelated cards should sit clearly above the acceptance threshold
    assert best > MATCH_ACCEPT * HASH_BITS, (best, pair)


def test_hamming_many_matches_scalar(samples):
    hashes = np.stack([card_hash(load(s["small"])) for s in samples])
    q = hashes[0]
    d = hamming_many(q, hashes)
    assert d[0] == 0
    assert all(int(d[i]) == hamming(q, hashes[i]) for i in range(len(hashes)))


def test_frame_signature_detects_change():
    a = np.zeros((200, 300, 3), np.uint8)
    b = a.copy()
    b[50:150, 50:150] = 255
    assert not frames_differ(frame_signature(a), frame_signature(a))
    assert frames_differ(frame_signature(a), frame_signature(b))
