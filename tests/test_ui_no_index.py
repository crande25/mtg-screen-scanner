"""How the window behaves with no index loaded.

This state was indistinguishable from a dead app: the preview stayed black, the OCR
indicator read "off" whatever the checkbox did, and nothing said which folder had been
searched. All three are checked here because all three misled the diagnosis.
"""
import time

import pytest


@pytest.fixture()
def app_without_index(tmp_path, monkeypatch):
    tk = pytest.importorskip("tkinter")
    monkeypatch.setenv("MTGSCAN_DATA_DIR", str(tmp_path))     # an empty data folder
    from mtgscan import ui

    try:
        root = tk.Tk()
    except tk.TclError as e:                                   # pragma: no cover
        pytest.skip(f"no display: {e}")
    root.withdraw()
    app = ui.ScannerApp(root)
    for _ in range(60):
        root.update()
        time.sleep(0.05)
        if app.index is None and app.ocr_ready:
            break
    yield app, root
    app.on_close()


def test_it_says_there_is_no_index_and_where_it_looked(app_without_index, tmp_path):
    app, _ = app_without_index
    assert app.index is None and app.identifier is None
    status = app.status.cget("text")
    assert "Index: none" in status
    assert str(tmp_path) in status, "the folder searched must be on screen, not guessed at"


def test_the_preview_still_shows_what_is_being_captured(app_without_index):
    """A black preview used to mean 'no index', 'not scanning' or 'broken capture'."""
    app, root = app_without_index
    app.settings["region"] = [0, 0, 240, 320]
    app.toggle_scan()
    for _ in range(80):
        root.update()
        time.sleep(0.05)
        if app._preview_photo is not None:
            break
    assert app._preview_photo is not None, "the capture preview must paint without an index"
    assert app.current == []


def test_the_ocr_indicator_distinguishes_its_states(app_without_index):
    app, _ = app_without_index
    assert app.ocr_ready
    # "idle" rather than "on": OCR is ready, but with no index nothing is using it. The old
    # label said "off" for this, for a missing package and for a user's own choice alike.
    expected = "not installed" if (app.ocr is None or not app.ocr.available) else "idle"
    assert app._ocr_state() == expected
    assert f"OCR: {expected}" in app.status.cget("text")


def test_a_missing_ocr_package_never_overwrites_the_preference(app_without_index):
    """It once did, which left OCR off for good after the package was installed."""
    from mtgscan.ocr import Ocr

    app, _ = app_without_index
    app.settings["use_ocr"] = True
    unavailable = Ocr.__new__(Ocr)
    unavailable.available = False
    app._on_ocr(unavailable)
    assert app._ocr_state() == "not installed"
    app._settings_changed()
    assert app.settings["use_ocr"] is True, "the saved choice must survive a missing package"


def test_the_startup_log_records_the_folder_and_index_state(tmp_path, monkeypatch):
    monkeypatch.setenv("MTGSCAN_DATA_DIR", str(tmp_path))
    from mtgscan import ui

    path = ui.setup_logging()
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "data folder" in text and "index present" in text
