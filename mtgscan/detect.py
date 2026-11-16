"""Find card-shaped quadrilaterals in a frame and warp them to a canonical portrait image.

Classic OpenCV pipeline (the same one phone scanners and magic_card_detector use):
edges/threshold -> contours -> 4-point polygon approximation -> aspect-ratio check
-> perspective transform. Several binarisations are tried because cards appear on
light web pages, dark stream overlays and photographed tables alike.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import WARP_W, WARP_H


@dataclass
class Quad:
    pts: np.ndarray  # float32 (4, 2): tl, tr, br, bl in frame coordinates

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        x0, y0 = self.pts.min(axis=0)
        x1, y1 = self.pts.max(axis=0)
        return int(x0), int(y0), int(x1), int(y1)

    @property
    def area(self) -> float:
        return float(abs(cv2.contourArea(self.pts.astype(np.float32))))

    @property
    def center(self) -> tuple[float, float]:
        c = self.pts.mean(axis=0)
        return float(c[0]), float(c[1])

    @property
    def size(self) -> tuple[float, float]:
        """Approximate (width, height) in frame pixels."""
        return _side_lengths(self.pts)


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _side_lengths(pts: np.ndarray) -> tuple[float, float]:
    tl, tr, br, bl = pts
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    return float(w), float(h)


def _aspect_ok(pts: np.ndarray, lo: float = 0.60, hi: float = 0.84) -> bool:
    w, h = _side_lengths(pts)
    if w < 8 or h < 8:
        return False
    r = min(w, h) / max(w, h)
    return lo <= r <= hi


def quad_is_sane(pts: np.ndarray, side_ratio: float = 0.80,
                 min_angle: float = 55.0, max_angle: float = 125.0) -> bool:
    """Reject quads that are not a plausible view of a rectangle.

    The aspect test alone averages opposite sides, so a badly skewed shape whose mean
    dimensions happen to look card-like slips through. On a dark background a card's own
    black border gives a weak outer edge, and the contour step then produces exactly that:
    a lopsided quad spanning most of the frame. Here opposite sides must be of similar
    length and every corner must be somewhere near a right angle, which still allows the
    perspective of a photographed card while discarding noise.
    """
    tl, tr, br, bl = pts
    top, bottom = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    left, right = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
    if min(top, bottom) < 1 or min(left, right) < 1:
        return False
    if min(top, bottom) / max(top, bottom) < side_ratio:
        return False
    if min(left, right) / max(left, right) < side_ratio:
        return False
    for a, vertex, c in ((bl, tl, tr), (tl, tr, br), (tr, br, bl), (br, bl, tl)):
        v1, v2 = a - vertex, c - vertex
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom < 1e-6:
            return False
        angle = np.degrees(np.arccos(np.clip(float(np.dot(v1, v2)) / denom, -1.0, 1.0)))
        if not (min_angle <= angle <= max_angle):
            return False
    return True


def _quads_from_binary(binary: np.ndarray, min_area: float, max_area: float) -> list[np.ndarray]:
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out: list[np.ndarray] = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < min_area or a > max_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = order_points(approx)
        else:
            # Rounded corners or a slightly broken outline: accept the min-area
            # rectangle when the contour fills it well.
            rect = cv2.minAreaRect(c)
            (rw, rh) = rect[1]
            if rw * rh <= 0 or a / (rw * rh) < 0.88:
                continue
            pts = order_points(cv2.boxPoints(rect))
        if _aspect_ok(pts) and quad_is_sane(pts):
            out.append(pts)
    return out


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _contains(outer: tuple[int, int, int, int], pt: tuple[float, float]) -> bool:
    return outer[0] <= pt[0] <= outer[2] and outer[1] <= pt[1] <= outer[3]


def nms(quads: list[Quad], iou_thresh: float = 0.3) -> list[Quad]:
    """Keep the largest of overlapping candidates (outer card edge beats inner frame)."""
    quads = sorted(quads, key=lambda q: q.area, reverse=True)
    kept: list[Quad] = []
    for q in quads:
        bb = q.bbox
        dup = False
        for k in kept:
            kb = k.bbox
            if _iou(bb, kb) > iou_thresh or _contains(kb, q.center):
                dup = True
                break
        if not dup:
            kept.append(q)
    return kept


def find_cards(frame: np.ndarray, min_area_frac: float = 0.003, max_area_frac: float = 0.985) -> list[Quad]:
    h, w = frame.shape[:2]
    area = float(h * w)
    min_area, max_area = area * min_area_frac, area * max_area_frac
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    cands: list[np.ndarray] = []

    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    cands += _quads_from_binary(edges, min_area, max_area)

    # A card's black border against a dark background is a very weak edge (seen on screen:
    # background 37, border 18), which the pass above misses. Local contrast equalisation
    # plus lower thresholds recovers it. CLAHE is what magic_card_detector uses for the
    # same reason.
    equalised = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blur)
    soft = cv2.Canny(equalised, 15, 60)
    soft = cv2.dilate(soft, np.ones((3, 3), np.uint8), iterations=1)
    cands += _quads_from_binary(soft, min_area, max_area)

    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7)
    cands += _quads_from_binary(thr, min_area, max_area)

    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cands += _quads_from_binary(otsu, min_area, max_area)
    cands += _quads_from_binary(cv2.bitwise_not(otsu), min_area, max_area)

    return nms([Quad(p) for p in cands])


def warp_card(frame: np.ndarray, quad: Quad, out_w: int = WARP_W, out_h: int = WARP_H) -> np.ndarray:
    """Perspective-correct a quad into a portrait card image."""
    pts = quad.pts.copy()
    w, h = _side_lengths(pts)
    if w > h:  # card lying sideways: rotate the point order so the output is portrait
        pts = np.roll(pts, -1, axis=0)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(frame, m, (out_w, out_h), flags=cv2.INTER_AREA)


def _peaks(energy: np.ndarray, limit: int, min_gap: int) -> list[int]:
    """Strongest positions in a 1-D edge-energy profile, kept apart by min_gap."""
    order = np.argsort(energy)[::-1]
    chosen: list[int] = []
    floor = float(energy.mean()) + 0.5 * float(energy.std())
    for i in order:
        if energy[i] < floor:
            break
        if all(abs(int(i) - c) >= min_gap for c in chosen):
            chosen.append(int(i))
        if len(chosen) >= limit:
            break
    return sorted(chosen)


def projection_quads(frame: np.ndarray, min_area: float, max_area: float,
                     max_quads: int = 4, peaks_per_axis: int = 7) -> list[Quad]:
    """Upright card rectangles found by projecting edge energy onto each axis.

    Per-pixel thresholding needs local contrast, which a card's black border on a dark
    background does not have. Summing the gradient down whole columns and across whole
    rows accumulates that weak edge over hundreds of pixels until it stands out, which
    suits screen captures because cards on a screen are almost always upright. Pairs of
    peaks are kept only where the spacing matches a card's 63:88 proportions.
    """
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    h, w = gray.shape[:2]
    col = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)).sum(axis=0)
    row = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)).sum(axis=1)
    xs = _peaks(col, peaks_per_axis, max(8, w // 40))
    ys = _peaks(row, peaks_per_axis, max(8, h // 40))

    scored: list[tuple[float, Quad]] = []
    for i, x0 in enumerate(xs):
        for x1 in xs[i + 1:]:
            width = x1 - x0
            if width < 8:
                continue
            for j, y0 in enumerate(ys):
                for y1 in ys[j + 1:]:
                    height = y1 - y0
                    if height < 8 or not (0.60 <= width / height <= 0.84):
                        continue
                    area = float(width * height)
                    if area < min_area or area > max_area:
                        continue
                    strength = float(col[x0] + col[x1] + row[y0] + row[y1])
                    scored.append((strength, Quad(np.array(
                        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32))))
    scored.sort(key=lambda s: s[0], reverse=True)
    return [q for _, q in scored[:max_quads]]


def axis_aligned(quad: Quad) -> Quad:
    """The quad's bounding box as a quad.

    Cards on a screen are nearly always upright rectangles, so when the contour step
    returns a slightly tilted or ragged outline its bounding box is often the better
    reading. Offered as an extra candidate rather than a replacement, since a photographed
    card genuinely does need the perspective transform.
    """
    x0, y0, x1, y1 = quad.bbox
    return Quad(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32))


def whole_frame_quad(frame: np.ndarray) -> Quad:
    h, w = frame.shape[:2]
    return Quad(np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32))


def looks_like_card(frame: np.ndarray) -> bool:
    h, w = frame.shape[:2]
    r = min(w, h) / max(w, h)
    return 0.55 <= r <= 0.90
