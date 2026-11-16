"""Screen capture of a rectangular region (physical pixels) using mss."""
from __future__ import annotations

import threading

import numpy as np
import mss

Region = tuple[int, int, int, int]  # left, top, width, height


class ScreenCapture:
    """Thread-safe wrapper: mss handles are per-thread on Windows (GDI)."""

    def __init__(self) -> None:
        self._local = threading.local()

    def _mss(self):
        m = getattr(self._local, "m", None)
        if m is None:
            # mss.mss() is deprecated in favour of mss.MSS(); keep working on both.
            factory = getattr(mss, "MSS", None) or mss.mss
            m = factory()
            self._local.m = m
        return m

    def grab(self, region: Region) -> np.ndarray:
        """Return a BGR uint8 array of the region."""
        left, top, width, height = region
        shot = self._mss().grab({"left": int(left), "top": int(top), "width": int(width), "height": int(height)})
        arr = np.frombuffer(shot.bgra, dtype=np.uint8).reshape(shot.height, shot.width, 4)
        return np.ascontiguousarray(arr[:, :, :3])

    def virtual_screen(self) -> dict:
        """Bounding box of all monitors: keys left, top, width, height."""
        return dict(self._mss().monitors[0])

    def monitors(self) -> list[dict]:
        return [dict(m) for m in self._mss().monitors[1:]]
