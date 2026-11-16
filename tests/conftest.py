"""Shared fixtures: a handful of real Scryfall card images, cached on disk.

The tests need network access the first time; afterwards they run from tests/.cache.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

CACHE = Path(__file__).parent / ".cache"

# A mixed bag: old frame, modern frame, a planeswalker, a saga, a double-faced card.
SAMPLE_QUERY = "set:m10 (name:\"Lightning Bolt\" or name:\"Baneslayer Angel\" or name:\"Garruk Wildspeaker\") " \
               "or (set:dom name:\"History of Benalia\") or (set:ltr name:\"Delighted Halfling\") " \
               "or (set:mid name:\"Brutal Cathar\")"


def _download_samples() -> list[dict]:
    from mtgscan import scryfall
    from mtgscan.scryfall import records_from_card

    CACHE.mkdir(exist_ok=True)
    meta_path = CACHE / "samples.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    samples = []
    for card in scryfall.search_cards(SAMPLE_QUERY):
        for rec in records_from_card(card):
            small = CACHE / f"{rec.id}_{rec.face}_small.jpg"
            normal = CACHE / f"{rec.id}_{rec.face}_normal.jpg"
            small.write_bytes(scryfall.download_image(rec.image_small))
            normal.write_bytes(scryfall.download_image(rec.image_normal))
            samples.append({"id": rec.id, "face": rec.face, "name": rec.face_name, "set": rec.set_code,
                            "small": str(small), "normal": str(normal)})
    meta_path.write_text(json.dumps(samples, indent=1))
    return samples


@pytest.fixture(scope="session")
def samples() -> list[dict]:
    try:
        s = _download_samples()
    except Exception as e:  # pragma: no cover - offline
        pytest.skip(f"sample download failed: {e}")
    # Searches return every printing; keep one image per face name so the "different
    # cards" tests really compare different cards (same-art reprints hash alike).
    seen: set[str] = set()
    uniq = []
    for item in s:
        if item["name"] in seen:
            continue
        seen.add(item["name"])
        uniq.append(item)
    if len(uniq) < 4:
        pytest.skip("not enough samples")
    return uniq


def load(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    assert img is not None, path
    return img
