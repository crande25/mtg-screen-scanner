"""Perceptual hashing of card images.

Approach (the same family used by phone scanners and most open-source MTG
recognizers): shrink the card to a tiny grayscale image and derive a bit string
from its low-frequency DCT (pHash) and from neighbouring-pixel gradients (dHash).
Two hashes of the same artwork differ in only a few bits even after rescaling,
JPEG compression or mild colour shifts, while different cards differ in roughly
half of the bits. Matching is then a Hamming distance against the whole index,
which numpy does for 100k cards in a few milliseconds.
"""
from __future__ import annotations

import numpy as np
import cv2

from .config import HASH_SIZE, HASH_BYTES, CARD_INSET

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def inset(img: np.ndarray, frac: float = CARD_INSET) -> np.ndarray:
    """Trim a border fraction from every edge (hides rounded corners and border width)."""
    h, w = img.shape[:2]
    dx, dy = int(round(w * frac)), int(round(h * frac))
    if dx <= 0 or dy <= 0 or w - 2 * dx < 8 or h - 2 * dy < 8:
        return img
    return img[dy:h - dy, dx:w - dx]


def phash_bits(gray: np.ndarray, hash_size: int = HASH_SIZE) -> np.ndarray:
    n = hash_size * 4
    small = cv2.resize(gray, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    low = dct[:hash_size, :hash_size]
    med = np.median(low)
    return (low > med).flatten()


def dhash_bits(gray: np.ndarray, hash_size: int = HASH_SIZE) -> np.ndarray:
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA).astype(np.int16)
    return (small[:, 1:] > small[:, :-1]).flatten()


def card_hash(img: np.ndarray) -> np.ndarray:
    """Hash a portrait card image (any size, BGR or gray). Returns uint8[HASH_BYTES]."""
    g = to_gray(inset(img))
    g = cv2.GaussianBlur(g, (3, 3), 0)  # soften screen-scaling aliasing before downsampling
    bits = np.concatenate([phash_bits(g), dhash_bits(g)]).astype(np.uint8)
    packed = np.packbits(bits)
    assert packed.shape[0] == HASH_BYTES
    return packed


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(_POPCOUNT[np.bitwise_xor(a, b)].sum())


def hamming_many(query: np.ndarray, db: np.ndarray) -> np.ndarray:
    """Hamming distance from `query` (uint8[B]) to every row of `db` (uint8[N, B])."""
    if db.shape[0] == 0:
        return np.zeros(0, dtype=np.int32)
    return _POPCOUNT[np.bitwise_xor(db, query[None, :])].sum(axis=1, dtype=np.int32)


def hash_to_hex(h: np.ndarray) -> str:
    return h.tobytes().hex()


def hash_from_hex(s: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(s), dtype=np.uint8).copy()


# --- Cheap frame-change detection (so an unchanged screen is not re-scanned) ---

def frame_signature(frame: np.ndarray) -> np.ndarray:
    g = to_gray(frame)
    return cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.int16)


def frames_differ(sig_a: np.ndarray | None, sig_b: np.ndarray, threshold: float = 4.0) -> bool:
    if sig_a is None or sig_a.shape != sig_b.shape:
        return True
    return float(np.abs(sig_a - sig_b).mean()) > threshold
