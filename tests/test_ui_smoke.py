"""Construct the Tk app hidden, feed it a frame, check the tables fill. Skips without a display."""
from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from tests.test_identify_e2e import _normal_image, _pick, small_index  # noqa: F401  (fixture)
from tests.test_detect import paste


def test_app_handles_a_frame(small_index, tmp_path, monkeypatch):
    tk = pytest.importorskip("tkinter")
    monkeypatch.setenv("MTGSCAN_DATA_DIR", str(tmp_path))  # settings + (missing) index live here
    from mtgscan import ui

    try:
        root = tk.Tk()
    except tk.TclError as e:  # pragma: no cover
        pytest.skip(f"no display: {e}")
    root.withdraw()
    app = ui.ScannerApp(root)
    # let the background index probe report "no index" before we inject our own
    for _ in range(20):
        root.update()
        time.sleep(0.05)
    app._on_index(small_index, None, "")
    assert app.identifier is not None

    recs = _pick(small_index, 2)
    frame = np.full((600, 900, 3), (240, 240, 240), np.uint8)
    paste(frame, _normal_image(recs[0]), 30, 30, 0.6)
    paste(frame, _normal_image(recs[1]), 420, 60, 0.6)
    dets = app.identifier.identify_frame(frame)
    app._on_frame(frame, dets, 0.05)
    root.update()

    assert len(app.tree.get_children()) == len(dets) >= 2
    names = {app.tree.item(i)["values"][0] for i in app.tree.get_children()}
    assert {r.name for r in recs} <= names
    assert len(app.hist.get_children()) >= 2

    # before the live quote arrives the market columns show the daily index price (~)
    vals = [app.tree.item(i)["values"] for i in app.tree.get_children()]
    assert all(str(v[5]).startswith("~") or v[5] in ("...", "n/a") for v in vals), vals

    # prices arrive asynchronously (tcgcsv / Scryfall); wait and check the low column fills
    for _ in range(150):
        root.update()
        time.sleep(0.1)
        vals = [app.tree.item(i)["values"] for i in app.tree.get_children()]
        if all(v[3] != "..." for v in vals):
            break
    vals = [app.tree.item(i)["values"] for i in app.tree.get_children()]
    assert all(v[3] != "..." for v in vals), vals
    assert all(not str(v[5]).startswith("~") for v in vals), vals

    # selecting a row fills the detail panel; alternative list exists
    app.tree.selection_set(app.tree.get_children()[0])
    app._on_select()
    assert app.detail.cget("text")

    app.on_close()
