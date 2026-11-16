"""Synthetic screen frames: cards pasted on backgrounds, one with perspective, one sideways."""
import cv2
import numpy as np

from mtgscan.config import HASH_BITS, MATCH_ACCEPT
from mtgscan.detect import find_cards, warp_card
from mtgscan.hashing import card_hash, hamming
from tests.conftest import load


def paste(frame: np.ndarray, card: np.ndarray, x: int, y: int, scale: float = 1.0) -> tuple[int, int, int, int]:
    if scale != 1.0:
        card = cv2.resize(card, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = card.shape[:2]
    frame[y:y + h, x:x + w] = card
    return x, y, x + w, y + h


def paste_perspective(frame: np.ndarray, card: np.ndarray, quad: np.ndarray) -> None:
    h, w = card.shape[:2]
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    m = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    warped = cv2.warpPerspective(card, m, (frame.shape[1], frame.shape[0]))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), m, (frame.shape[1], frame.shape[0]))
    frame[mask > 0] = warped[mask > 0]


def make_frame(samples, bg=(240, 240, 240)):
    frame = np.full((900, 1400, 3), bg, np.uint8)
    cards = [load(s["normal"]) for s in samples[:4]]
    boxes = []
    boxes.append(paste(frame, cards[0], 40, 40, 0.5))                       # 244x340
    boxes.append(paste(frame, cards[1], 340, 60, 0.35))                      # small
    quad = np.array([[620, 80], [900, 110], [880, 520], [600, 480]], np.float32)
    paste_perspective(frame, cards[2], quad)
    boxes.append((600, 80, 900, 520))
    side = cv2.rotate(cv2.resize(cards[3], None, fx=0.45, fy=0.45), cv2.ROTATE_90_CLOCKWISE)
    boxes.append(paste(frame, side, 960, 300))
    return frame, cards, boxes


def _found(boxes, quads):
    hits = []
    for (x0, y0, x1, y1) in boxes:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hit = None
        for q in quads:
            bx0, by0, bx1, by1 = q.bbox
            if bx0 <= cx <= bx1 and by0 <= cy <= by1:
                hit = q
                break
        hits.append(hit)
    return hits


def test_find_cards_on_light_and_dark_backgrounds(samples):
    for bg in [(240, 240, 240), (20, 20, 20)]:
        frame, cards, boxes = make_frame(samples, bg)
        quads = find_cards(frame)
        hits = _found(boxes, quads)
        assert all(h is not None for h in hits), f"bg={bg} found {len(quads)} quads, hits={hits}"
        # no spurious extra cards (inner frames must have been suppressed)
        assert len(quads) <= len(boxes) + 1, len(quads)


def test_warped_cards_hash_to_their_source(samples):
    frame, cards, boxes = make_frame(samples)
    quads = find_cards(frame)
    hits = _found(boxes, quads)
    for card, q in zip(cards, hits):
        assert q is not None
        warped = warp_card(frame, q)
        ref = card_hash(cv2.resize(card, (146, 204), interpolation=cv2.INTER_AREA))
        d = hamming(ref, card_hash(warped))
        d_rot = hamming(ref, card_hash(cv2.rotate(warped, cv2.ROTATE_180)))
        assert min(d, d_rot) <= MATCH_ACCEPT * HASH_BITS, (d, d_rot)
