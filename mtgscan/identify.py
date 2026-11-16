"""Identification pipeline: frame -> card quads -> hash match -> printing choice.

Mirrors what the phone scanners do (artwork match first, then printed collector
information to pick the exact printing, then manual choice): see ManaBox and
TCG Machines write-ups referenced in the README.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .config import HASH_BITS, MATCH_ACCEPT, MATCH_STRONG, SAME_ART_DELTA, WARP_H, WARP_W
from .detect import Quad, find_cards, warp_card, whole_frame_quad
from .hashing import card_hash
from .index import HashIndex
from .ocr import Ocr, printing_similarity, read_collector_line, read_title
from .scryfall import CardRecord

log = logging.getLogger(__name__)

MIN_QUAD_WIDTH = 36          # below this a hash is noise
OCR_MIN_HEIGHT = 300         # on-screen card height needed before the collector line is readable
OCR_MAX_HEIGHT = 2 * WARP_H  # reading the line is much more reliable above the hashing size
# Weights for choosing a printing from the collector line. An exact collector number is the
# strongest single signal; a set code alone is weaker because promos and token sheets print
# their base set's code. The floor keeps a single fuzzy glyph match from deciding anything.
OCR_NUMBER_WEIGHT = 2.0
OCR_SET_WEIGHT = 1.5
OCR_MIN_SCORE = 1.0
# A promo or The List sibling reprints the base set's scan and its collector line, so it
# scores exactly (1.0-0.9)*SET + (1.0-0.85)*NUMBER = 0.45 below the base printing. Nothing in
# the image separates them, so the margin sits just above that and the hash keeps the call.
OCR_TIE_MARGIN = 0.5


@dataclass
class Candidate:
    card: CardRecord
    distance: Optional[int]   # None when the candidate came from OCR rather than hashing

    @property
    def label(self) -> str:
        d = "" if self.distance is None else f" (d={self.distance})"
        return f"{self.card.set_code.upper()} #{self.card.collector_number}{d}"


@dataclass
class Detection:
    quad: Quad
    image: np.ndarray
    card: Optional[CardRecord] = None
    candidates: list[Candidate] = field(default_factory=list)
    distance: Optional[int] = None
    confidence: float = 0.0
    method: str = "none"          # hash | hash+ocr | ocr-name | none
    ocr_set: Optional[str] = None
    ocr_number: Optional[str] = None
    rotated: bool = False         # the card was upside down in the frame
    note: str = ""

    @property
    def name(self) -> str:
        return self.card.name if self.card else "(unknown)"

    @property
    def ambiguous(self) -> bool:
        return len(self.candidates) > 1


class Identifier:
    def __init__(self, index: HashIndex, ocr: Optional[Ocr] = None, use_ocr: bool = True):
        self.index = index
        self.ocr = ocr if (ocr is not None and ocr.available and use_ocr) else None

    # --- frame level --------------------------------------------------------------

    def identify_frame(self, frame: np.ndarray, single_card: bool = False) -> list[Detection]:
        if frame is None or frame.size == 0:
            return []
        if single_card:
            quads = [whole_frame_quad(frame)]
        else:
            quads = [q for q in find_cards(frame) if q.size[0] >= MIN_QUAD_WIDTH]
        out = []
        for q in quads:
            try:
                out.append(self.identify_card(frame, q))
            except Exception as e:  # keep scanning the other cards
                log.exception("identify failed: %s", e)
        out.sort(key=lambda d: (d.quad.bbox[1] // 40, d.quad.bbox[0]))
        return out

    # --- card level ---------------------------------------------------------------

    def identify_card(self, frame: np.ndarray, quad: Quad) -> Detection:
        img = warp_card(frame, quad)
        det = Detection(quad=quad, image=img)

        best = self._best_orientation(img)
        if best is None:
            det.note = "empty index"
            return det
        img, matches, rotated = best
        det.image = img
        det.rotated = rotated

        accept = MATCH_ACCEPT * HASH_BITS
        if matches and matches[0][1] <= accept:
            self._resolve_hash_match(det, matches, frame)
        else:
            det.note = f"no hash match (best d={matches[0][1] if matches else 'n/a'})"
            self._fallback_ocr_name(det, frame)
        return det

    def _best_orientation(self, img: np.ndarray):
        """Hash the card upright and rotated 180 degrees; keep the better orientation."""
        if len(self.index) == 0:
            return None
        h = card_hash(img)
        m = self.index.query(h, k=15)
        if m and m[0][1] <= MATCH_STRONG * HASH_BITS:
            return img, m, False
        rot = cv2.rotate(img, cv2.ROTATE_180)
        m2 = self.index.query(card_hash(rot), k=15)
        if m2 and (not m or m2[0][1] < m[0][1]):
            return rot, m2, True
        return img, m, False

    def _ocr_image(self, frame: np.ndarray, det: Detection) -> Optional[np.ndarray]:
        """The card re-warped large enough for the printed collector line to be legible.

        The hash runs at the canonical 292x408 it was calibrated for; OCR gets its own,
        bigger warp straight from the frame, because that line is only a few pixels tall
        at hashing size. Cards too small on screen are skipped rather than guessed at.
        """
        w, h = det.quad.size
        long_side = max(w, h)
        if long_side < OCR_MIN_HEIGHT:
            return None
        out_h = int(min(max(long_side, WARP_H), OCR_MAX_HEIGHT))
        out_w = int(round(out_h * WARP_W / WARP_H))
        img = warp_card(frame, det.quad, out_w=out_w, out_h=out_h)
        return cv2.rotate(img, cv2.ROTATE_180) if det.rotated else img

    def _resolve_hash_match(self, det: Detection, matches: list[tuple[int, int]], frame: np.ndarray) -> None:
        best_d = matches[0][1]
        close = [(rid, d) for rid, d in matches if d - best_d <= SAME_ART_DELTA]
        cands = [Candidate(self.index.card(rid), d) for rid, d in close]
        # Different faces of the same printing or the same card twice: dedupe by (id, face).
        seen = set()
        uniq: list[Candidate] = []
        for c in cands:
            key = (c.card.id, c.card.face)
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        cands = uniq
        det.candidates = cands
        det.distance = best_d
        det.confidence = max(0.0, 1.0 - best_d / HASH_BITS)
        det.method = "hash"
        det.card = cands[0].card

        if len(cands) > 1 and self.ocr is not None:
            chosen = self._pick_by_collector_line(det, cands, frame)
            if chosen is not None:
                det.card = chosen.card
                det.method = "hash+ocr"
                # put the chosen printing first
                det.candidates = [chosen] + [c for c in cands if c is not chosen]
        if len(det.candidates) > 1 and det.method == "hash":
            det.note = f"{len(det.candidates)} printings look identical; showing the closest"

    def _pick_by_collector_line(self, det: Detection, cands: list[Candidate],
                                frame: np.ndarray) -> Optional[Candidate]:
        """Score each candidate against the printed line and take the best, if any is good.

        Scoring rather than exact matching, because the printed code legitimately differs
        from the Scryfall set code for promos and tokens, and a line this small is read
        with the odd wrong glyph.
        """
        if self.ocr is None:
            return None
        img = self._ocr_image(frame, det)
        if img is None:
            det.note = "card too small on screen to read its collector line"
            return None
        info = read_collector_line(self.ocr, img)
        det.ocr_set, det.ocr_number = info.set_code, info.number
        if not info.set_code and not info.number:
            return None
        scored = []
        for i, c in enumerate(cands):
            set_score, num_score = printing_similarity(info.set_code, info.number,
                                                       c.card.set_code, c.card.collector_number)
            scored.append((OCR_SET_WEIGHT * set_score + OCR_NUMBER_WEIGHT * num_score, i, c))
        top = max(scored, key=lambda s: (s[0], -s[1]))
        if top[0] < OCR_MIN_SCORE:
            return None
        # Several printings can fit the same line (The List and promos reprint a set's own
        # collector line). When the best scores are that close the line cannot separate them,
        # so defer to the hash rather than flip to an equally plausible printing.
        near = [s for s in scored if top[0] - s[0] <= OCR_TIE_MARGIN]
        if len(near) > 1:
            return min(near, key=lambda s: s[1])[2]
        return top[2]

    def _fallback_ocr_name(self, det: Detection, frame: np.ndarray) -> None:
        if self.ocr is None:
            return
        title = read_title(self.ocr, self._ocr_image(frame, det) or det.image)
        if len(title) < 3:
            return
        try:
            from rapidfuzz import fuzz, process
        except ImportError:  # pragma: no cover
            return
        names = self.index.unique_names()
        if not names:
            return
        hit = process.extractOne(title, names, scorer=fuzz.WRatio, score_cutoff=82)
        if not hit:
            det.note += f"; title OCR '{title}' matched nothing"
            return
        name, score, _ = hit
        printings = self.index.cards_by_name(name)
        if not printings:
            return
        printings.sort(key=lambda c: c.released_at, reverse=True)
        cands = [Candidate(c, None) for c in printings]
        det.candidates = cands
        det.card = cands[0].card
        det.method = "ocr-name"
        det.confidence = float(score) / 100.0 * 0.9
        det.note = f"matched by title OCR ({title!r})"
        chosen = self._pick_by_collector_line(det, cands, frame)
        if chosen is not None:
            det.card = chosen.card
            det.candidates = [chosen] + [c for c in cands if c is not chosen]


def draw_detections(frame: np.ndarray, dets: list[Detection]) -> np.ndarray:
    """Annotated copy of the frame for the preview panel."""
    out = frame.copy()
    for d in dets:
        color = (0, 200, 0) if d.card and d.confidence >= 1 - MATCH_STRONG else ((0, 165, 255) if d.card else (0, 0, 255))
        pts = d.quad.pts.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, color, 2)
        x, y = d.quad.bbox[0], max(14, d.quad.bbox[1] - 4)
        cv2.putText(out, d.name[:28], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out
