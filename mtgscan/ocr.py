"""Optional OCR, used the way phone scanners use it: as a second signal.

* The collector line printed at the bottom-left of every card since Magic 2015
  ("0146/0264 R" and "M15 - EN") tells reprints with identical artwork apart.
* The title bar gives a name when the hash match is weak (tiny or odd images).

Backend: rapidocr_onnxruntime (``pip install rapidocr-onnxruntime``), a pure
pip install with bundled models. When it is missing OCR is silently disabled
and identification relies on hashes alone.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class OcrLine:
    text: str
    conf: float


class Ocr:
    def __init__(self) -> None:
        self._engine = None
        self._lock = threading.Lock()
        self.available = self._load()

    def _load(self) -> bool:
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
        except Exception as e:  # ImportError or a broken onnxruntime install
            log.info("OCR disabled (rapidocr_onnxruntime not available): %s", e)
            return False
        try:
            self._engine = RapidOCR()
        except Exception as e:
            log.warning("OCR disabled (engine failed to start): %s", e)
            return False
        return True

    def lines(self, img: np.ndarray) -> list[OcrLine]:
        if not self.available or img is None or img.size == 0:
            return []
        with self._lock:
            try:
                result, _ = self._engine(img)
            except Exception as e:
                log.warning("OCR failed: %s", e)
                return []
        out: list[OcrLine] = []
        for item in result or []:
            try:
                _box, text, conf = item
            except (TypeError, ValueError):
                continue
            out.append(OcrLine(str(text), float(conf)))
        return out


# --- Region helpers (fractions of a portrait card) --------------------------------

def _crop(img: np.ndarray, x0: float, y0: float, x1: float, y1: float, scale: float = 1.0) -> np.ndarray:
    h, w = img.shape[:2]
    c = img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    if scale != 1.0 and c.size:
        c = cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return c


def title_crop(card: np.ndarray) -> np.ndarray:
    return _crop(card, 0.04, 0.03, 0.82, 0.115, scale=2.0)


def collector_crop(card: np.ndarray) -> np.ndarray:
    return _crop(card, 0.02, 0.90, 0.65, 0.99, scale=3.0)


# The bottom-left of a modern card reads, over two lines:
#     0087/0281 R        (older sets)   or   R 0087   /   0087 R   (newer sets)
#     WOE · EN · DONATO GIANCOLA
# OCR merges the lines and often loses the separator, so the patterns below are
# deliberately tolerant: "R0087 WOE?ENDONATO GIANCOLA" must still parse.
LANGS = "EN|DE|FR|IT|ES|PT|JA|JP|KO|RU|ZH|CS|CT|PH"
RARITY = "CURMLSTPB"

# Set code sits immediately before the language code; the separator may be missing or
# garbled and the language may be glued to the artist name, so no trailing boundary.
_SET_LANG_RE = re.compile(rf"\b([A-Z0-9]{{2,5}})\b[^A-Z0-9]{{0,3}}(?:{LANGS})")
# "123/264", optionally with a rarity letter on either side.
_NUM_SLASH_RE = re.compile(r"(?<![\d/])(\d{1,4})\s*/\s*\d{1,4}")
# A standalone collector number: zero padded, or carrying a rarity letter.
_NUM_TOKEN_RE = re.compile(rf"(?<![A-Z0-9])(?:[{RARITY}]\s*)?(\d{{2,4}})(?:\s*[{RARITY}])?(?![\d])")
_SET_LOOSE_RE = re.compile(r"\b([A-Z][A-Z0-9]{2,4})\b")
_NOT_A_SET = {"THE", "AND", "ART", "ILLUS", "WOTC", "INC", "LLC", "TM", "ALL", "RIGHTS"}


@dataclass
class CollectorInfo:
    set_code: Optional[str]
    number: Optional[str]
    raw: str


def _clean_number(digits: str) -> Optional[str]:
    """Strip the zero padding Magic prints ('0087' -> '87'); keep '0' itself out."""
    n = digits.lstrip("0")
    return n if n else None


def parse_collector_text(texts: list[str]) -> CollectorInfo:
    joined = " ".join(t.strip() for t in texts if t.strip())
    # OCR confuses O/0 inside what should be digits; normalise only around digits.
    up = joined.upper()
    up = re.sub(r"(?<=\d)O", "0", up)
    up = re.sub(r"O(?=\d)", "0", up)

    number = None
    m = _NUM_SLASH_RE.search(up)
    if m:
        number = _clean_number(m.group(1))
    if number is None:
        # Rank the standalone candidates: Magic zero-pads the collector number to four
        # digits, so a padded token beats a bare one ("30A ... C0195" must give 195, not 30).
        best = None
        for m in _NUM_TOKEN_RE.finditer(up):
            digits = m.group(1)
            cleaned = _clean_number(digits)
            if cleaned is None:
                continue
            score = (digits.startswith("0"), len(digits), -m.start())
            if best is None or score > best[0]:
                best = (score, cleaned)
        if best is not None:
            number = best[1]

    set_code = None
    matches = _SET_LANG_RE.findall(up)
    if matches:
        # The language code appears once, right after the set; take the last match in case
        # the artist name happens to contain a language-like fragment earlier.
        set_code = matches[-1].lower()
    else:
        for tok in _SET_LOOSE_RE.findall(up):
            if tok not in _NOT_A_SET and not tok.isdigit():
                set_code = tok.lower()
                break
    return CollectorInfo(set_code, number, joined)


def read_collector_line(ocr: Ocr, card: np.ndarray) -> CollectorInfo:
    lines = ocr.lines(collector_crop(card))
    return parse_collector_text([ln.text for ln in lines if ln.conf >= 0.3])


# --- Fuzzy comparison against a known printing -------------------------------------
# OCR of a ~30 px tall line confuses similar glyphs, and the code printed on the card is
# not always the Scryfall set code: promos print the base set ("pafr" prints AFR), token
# sheets likewise ("ttla" prints TLA). Both are handled by comparison, not by the parser.

_CONFUSIONS = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "D": "O", "Q": "O"})


def _canon(s: str) -> str:
    return s.upper().translate(_CONFUSIONS)


def set_similarity(ocr_code: Optional[str], card_set: str) -> float:
    """1.0 exact, 0.8 same after glyph confusions or a promo/token prefix, 0.0 unrelated."""
    if not ocr_code or not card_set:
        return 0.0
    a, b = ocr_code.lower(), card_set.lower()
    if a == b:
        return 1.0
    # Promo and token sets print their base set's code.
    if (b.startswith(("p", "t")) and b[1:] == a) or (a.startswith(("p", "t")) and a[1:] == b):
        return 0.9
    if _canon(a) == _canon(b):
        return 0.8
    if len(a) == len(b) and sum(x != y for x, y in zip(_canon(a), _canon(b))) == 1:
        return 0.6
    return 0.0


_EMBEDDED_SET_RE = re.compile(r"^([A-Za-z0-9]{2,5})-(\d+)$")


def printing_similarity(ocr_set: Optional[str], ocr_number: Optional[str],
                        card_set: str, collector_number: str) -> tuple[float, float]:
    """(set score, number score) for one printing, as a pair so callers can weight them.

    Handles collector numbers that embed their origin set, which is how The List is
    numbered ("EMA-198"): such a card prints the original set's line, so it must be
    compared against that as well as against its own set code.
    """
    s = set_similarity(ocr_set, card_set)
    n = number_similarity(ocr_number, collector_number)
    m = _EMBEDDED_SET_RE.match(collector_number or "")
    if m:
        s = max(s, set_similarity(ocr_set, m.group(1)))
        n = max(n, number_similarity(ocr_number, m.group(2)))
    return s, n


def number_similarity(ocr_number: Optional[str], collector_number: str) -> float:
    """Collector numbers carry suffixes the printed line may omit ('381z' vs '381')."""
    if not ocr_number or not collector_number:
        return 0.0
    a = ocr_number.lower().lstrip("0")
    b = collector_number.lower().lstrip("0")
    if a == b:
        return 1.0
    digits_b = re.sub(r"\D", "", b).lstrip("0")
    if a == digits_b and digits_b:
        return 0.85
    return 0.0


def read_title(ocr: Ocr, card: np.ndarray) -> str:
    lines = ocr.lines(title_crop(card))
    if not lines:
        return ""
    # The title is the longest confident line in the bar (mana cost may also be read).
    lines.sort(key=lambda ln: (ln.conf >= 0.4, len(ln.text)), reverse=True)
    text = lines[0].text.strip()
    return re.sub(r"[^A-Za-z0-9 ,'\-]", "", text).strip()
