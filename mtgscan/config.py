"""Paths and tunable constants."""
from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "mtg-screen-scanner"
USER_AGENT = "mtg-screen-scanner/0.1 (desktop card scanner)"


def data_dir() -> Path:
    """Durable per-user folder for the card index and settings.

    Defaults to %LOCALAPPDATA%/mtg-screen-scanner; override with MTGSCAN_DATA_DIR.
    """
    env = os.environ.get("MTGSCAN_DATA_DIR")
    if env:
        p = Path(env)
    else:
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
        p = Path(base) / APP_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


# --- Hashing -----------------------------------------------------------------
HASH_SIZE = 16          # pHash and dHash grids: 16x16 = 256 bits each, 512 bits total
HASH_BITS = 2 * HASH_SIZE * HASH_SIZE
HASH_BYTES = HASH_BITS // 8
CARD_INSET = 0.05       # fraction trimmed from every edge before hashing (border/corner tolerance)

# --- Card geometry (63 x 88 mm) ----------------------------------------------
CARD_ASPECT = 63 / 88   # width / height = 0.716
WARP_W, WARP_H = 292, 408   # canonical size of a detected card (2x Scryfall "small")

# --- Matching thresholds (fractions of HASH_BITS) ------------------------------
# Calibrated on 60 Magic 2010 printings against a 249-card index (2026-11): the same
# card re-rendered at 200/120/80 px tall with JPEG q70/50/40 sits 6-24 / 16-36 / 27-58
# bits from its index hash, while the nearest *wrong* card is never closer than 128
# bits (median 170). The nearest wrong card creeps closer as the index grows, hence
# the margin below.
MATCH_ACCEPT = 0.22     # best distance must be at most this (112 bits) to count as a match
MATCH_STRONG = 0.12     # at or below this (61 bits) the match is reported as confident
SAME_ART_DELTA = 12     # printings closer than this (in bits) to the best are "ambiguous"

# A card filmed by a webcam (an auction stream, say) carries tilt, glare and video
# compression, so it lands near the rejection boundary even when correct: a real stream
# frame scored 106 against the 112 cutoff. What still separates it cleanly is the margin
# over the nearest *different* artwork. Measured on that frame and on 36 frames of a
# window with no card in them:
#     real filmed card   best 106   margin 58
#     no card present    best 189+  margin 0-6
# So a larger absolute distance is accepted when the winner stands well clear of the field.
MATCH_RELAXED = 0.33    # 169 bits: the furthest a match may sit when its margin is wide
MATCH_MIN_MARGIN = 28   # bits the runner-up artwork must trail by for that to apply
