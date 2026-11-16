"""Tkinter desktop UI: pick a screen region, scan it on a timer, list cards and prices."""
from __future__ import annotations

import csv
import ctypes
import datetime as dt
import json
import logging
import queue
import sys
import threading
import time
import tkinter as tk
import webbrowser
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
from PIL import Image, ImageTk

from . import __version__
from .capture import ScreenCapture
from .config import MATCH_STRONG, data_dir
from .hashing import frame_signature, frames_differ
from .identify import Detection, Identifier, draw_detections
from .index import BuildOptions, HashIndex, Progress, build_index
from .ocr import Ocr
from .pricing import MODES, MODE_SHORT, MODE_TCGCSV, PriceQuote, PriceService, keys_from_env
from .scryfall import CardRecord

log = logging.getLogger(__name__)

DEFAULT_SETTINGS = {
    "region": None,          # [left, top, width, height] in physical pixels
    "interval": 2.0,
    "single_card": False,
    "pricing": MODE_TCGCSV,
    "use_ocr": True,
    "tcgplayer_public_key": "",
    "tcgplayer_private_key": "",
}


def enable_dpi_awareness() -> None:
    """Make Tk coordinates match physical pixels so the selected region lines up with mss."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def settings_path():
    return data_dir() / "settings.json"


def load_settings() -> dict:
    s = dict(DEFAULT_SETTINGS)
    try:
        s.update(json.loads(settings_path().read_text()))
    except (OSError, ValueError):
        pass
    return s


def save_settings(s: dict) -> None:
    try:
        settings_path().write_text(json.dumps(s, indent=1))
    except OSError as e:
        log.warning("could not save settings: %s", e)


# --- Region picker ----------------------------------------------------------------

class RegionSelector:
    """Full-screen translucent overlay; drag a rectangle, Esc cancels."""

    def __init__(self, root: tk.Tk, virtual: dict, on_done):
        self.on_done = on_done
        self.vx, self.vy = int(virtual["left"]), int(virtual["top"])
        self.top = tk.Toplevel(root)
        self.top.overrideredirect(True)
        self.top.attributes("-topmost", True)
        self.top.attributes("-alpha", 0.30)
        self.top.configure(bg="black")
        self.top.geometry(f"{int(virtual['width'])}x{int(virtual['height'])}+{self.vx}+{self.vy}")
        self.canvas = tk.Canvas(self.top, bg="black", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_text(int(virtual["width"]) // 2, 60, fill="white", font=("Segoe UI", 18),
                                text="Drag to select the area to scan  (Esc cancels)")
        self.start = None
        self.rect = None
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.top.bind("<Escape>", self._cancel)
        self.top.focus_force()
        self.canvas.focus_set()

    def _press(self, e):
        self.start = (e.x, e.y)
        self.rect = self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="#ff3b30", width=3)

    def _drag(self, e):
        if self.start and self.rect:
            self.canvas.coords(self.rect, self.start[0], self.start[1], e.x, e.y)

    def _release(self, e):
        if not self.start:
            return
        x0, y0 = self.start
        x1, y1 = e.x, e.y
        left, top = min(x0, x1), min(y0, y1)
        w, h = abs(x1 - x0), abs(y1 - y0)
        self.top.destroy()
        if w >= 20 and h >= 20:
            self.on_done((self.vx + left, self.vy + top, w, h))
        else:
            self.on_done(None)

    def _cancel(self, _e=None):
        self.top.destroy()
        self.on_done(None)


# --- Index build dialog ---------------------------------------------------------------

class BuildDialog(tk.Toplevel):
    def __init__(self, app: "ScannerApp"):
        super().__init__(app.root)
        self.app = app
        self.title("Build / update card index")
        self.resizable(False, False)
        self.stop_evt = threading.Event()
        self.thread: Optional[threading.Thread] = None

        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="The index hashes one Scryfall image per printing. A full build downloads\n"
                            "~100k small images at Scryfall's rate limit and takes a few hours; it is\n"
                            "resumable and later runs only add new printings.", justify="left").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        self.scope = tk.StringVar(value="since")
        ttk.Radiobutton(frm, text="Everything (all paper printings)", variable=self.scope, value="all").grid(row=1, column=0, columnspan=3, sticky="w")
        ttk.Radiobutton(frm, text="Printings released since year", variable=self.scope, value="since").grid(row=2, column=0, sticky="w")
        self.year = tk.StringVar(value=str(dt.date.today().year - 3))
        ttk.Spinbox(frm, from_=1993, to=dt.date.today().year, textvariable=self.year, width=6).grid(row=2, column=1, sticky="w")
        ttk.Radiobutton(frm, text="Only these set codes (comma separated)", variable=self.scope, value="sets").grid(row=3, column=0, sticky="w")
        self.sets = tk.StringVar(value="")
        ttk.Entry(frm, textvariable=self.sets, width=28).grid(row=3, column=1, columnspan=2, sticky="w")
        self.skip_tokens = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="Skip tokens and emblems", variable=self.skip_tokens).grid(row=4, column=0, columnspan=3, sticky="w", pady=(4, 8))

        self.progress = ttk.Progressbar(frm, length=420, mode="determinate")
        self.progress.grid(row=5, column=0, columnspan=3, sticky="we", pady=(4, 2))
        self.status = ttk.Label(frm, text="")
        self.status.grid(row=6, column=0, columnspan=3, sticky="w")

        btns = ttk.Frame(frm)
        btns.grid(row=7, column=0, columnspan=3, sticky="e", pady=(10, 0))
        self.start_btn = ttk.Button(btns, text="Start", command=self.start)
        self.start_btn.pack(side="left", padx=4)
        ttk.Button(btns, text="Close", command=self.close).pack(side="left", padx=4)
        self.protocol("WM_DELETE_WINDOW", self.close)

    def start(self):
        scope = self.scope.get()
        opts = BuildOptions(include_tokens=not self.skip_tokens.get())
        if scope == "since":
            opts.since = f"{int(self.year.get()):04d}-01-01"
        elif scope == "sets":
            codes = {s.strip().lower() for s in self.sets.get().split(",") if s.strip()}
            if not codes:
                messagebox.showwarning("Sets", "Enter at least one set code, e.g. mh3, blb", parent=self)
                return
            opts.sets = codes
        self.start_btn.configure(state="disabled")
        self.stop_evt.clear()

        def run():
            try:
                build_index(data_dir(), opts, progress=lambda p: self.app.events.put(("build", p)), stop=self.stop_evt)
            except Exception as e:
                log.exception("index build failed")
                self.app.events.put(("build", Progress("error", 0, 0, f"Build failed: {e}")))

        self.thread = threading.Thread(target=run, name="index-build", daemon=True)
        self.thread.start()

    def on_progress(self, p: Progress):
        if not self.winfo_exists():
            return
        if p.total:
            self.progress.configure(maximum=p.total, value=p.done)
        self.status.configure(text=p.message)
        if p.phase in ("done", "error"):
            self.start_btn.configure(state="normal")

    def close(self):
        if self.thread and self.thread.is_alive():
            if not messagebox.askyesno("Stop build?", "Stop the index build? Progress so far is kept and the next run resumes.", parent=self):
                return
            self.stop_evt.set()
        self.destroy()


# --- Main application ----------------------------------------------------------------

@dataclass
class HistoryEntry:
    card: CardRecord
    first_seen: float
    last_seen: float
    count: int = 1
    quote: Optional[PriceQuote] = None


PRICE_COLS = ("low", "low_foil", "market", "market_foil")
PRICE_HEADS = {"low": "Low", "low_foil": "Low foil", "market": "Market", "market_foil": "Market foil"}


def _money(q: Optional[PriceQuote], which: str, card: CardRecord) -> str:
    """Cell text for one of PRICE_COLS. '~' marks the daily price stored in the index."""
    finishes = card.finishes.split(",")
    is_foil = which.endswith("foil")
    if is_foil and "foil" not in finishes and "etched" not in finishes:
        return "n/a"
    if not is_foil and "nonfoil" not in finishes:
        return "n/a"
    index_market = (card.usd_foil if card.usd_foil is not None else card.usd_etched) if is_foil else card.usd
    if q is None:
        if which.startswith("market") and index_market is not None:
            return f"~{PriceQuote.fmt(index_market)}"
        return "..."
    if q.source == "unavailable":
        return "?"
    v = getattr(q, which)
    if v is None and is_foil:
        v = getattr(q, which.replace("foil", "etched"))
    if v is None and which.startswith("market") and index_market is not None:
        return f"~{PriceQuote.fmt(index_market)}"
    return PriceQuote.fmt(v)


class ScannerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title(f"MTG Screen Scanner {__version__}")
        root.geometry("1240x760")
        root.minsize(980, 600)

        self.settings = load_settings()
        self.capture = ScreenCapture()
        self.index: Optional[HashIndex] = None
        self.identifier: Optional[Identifier] = None
        self.ocr: Optional[Ocr] = None
        self.prices = PriceService(mode=self.settings.get("pricing", MODE_TCGCSV), keys=self._tcg_keys())
        self.events: "queue.Queue[tuple]" = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.stop_evt = threading.Event()
        self.force_scan = threading.Event()
        self.current: list[Detection] = []
        self.history: dict[str, HistoryEntry] = {}
        self._last_ids: set[str] = set()
        self._preview_photo = None
        self._last_frame = None
        self.build_dialog: Optional[BuildDialog] = None

        self._build_ui()
        self._apply_settings_to_widgets()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(100, self._poll_events)
        threading.Thread(target=self._load_index, name="index-load", daemon=True).start()

    # -- UI construction ---------------------------------------------------------------
    def _build_ui(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass

        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(side="top", fill="x")
        ttk.Button(bar, text="Select region...", command=self.select_region).pack(side="left")
        self.region_lbl = ttk.Label(bar, text="no region", width=26)
        self.region_lbl.pack(side="left", padx=(6, 12))
        self.start_btn = ttk.Button(bar, text="Start scanning", command=self.toggle_scan, state="disabled")
        self.start_btn.pack(side="left")
        ttk.Button(bar, text="Scan now", command=self.scan_now).pack(side="left", padx=(6, 12))
        ttk.Label(bar, text="Every").pack(side="left")
        self.interval_var = tk.StringVar(value="2.0")
        sp = ttk.Spinbox(bar, from_=0.5, to=60, increment=0.5, width=5, textvariable=self.interval_var, command=self._settings_changed)
        sp.pack(side="left", padx=(4, 2))
        sp.bind("<FocusOut>", lambda e: self._settings_changed())
        ttk.Label(bar, text="s").pack(side="left", padx=(0, 12))
        self.single_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Region is one card", variable=self.single_var, command=self._settings_changed).pack(side="left", padx=(0, 12))
        self.ocr_var = tk.BooleanVar(value=True)
        self.ocr_chk = ttk.Checkbutton(bar, text="Use OCR", variable=self.ocr_var, command=self._settings_changed)
        self.ocr_chk.pack(side="left", padx=(0, 12))
        ttk.Label(bar, text="Prices:").pack(side="left")
        self.pricing_var = tk.StringVar()
        self.pricing_cb = ttk.Combobox(bar, state="readonly", width=38, values=list(MODES.values()), textvariable=self.pricing_var)
        self.pricing_cb.pack(side="left", padx=(4, 12))
        self.pricing_cb.bind("<<ComboboxSelected>>", lambda e: self._settings_changed())
        ttk.Button(bar, text="API keys...", command=self.edit_keys).pack(side="left", padx=(0, 12))
        ttk.Button(bar, text="Build / update index...", command=self.open_build_dialog).pack(side="right")

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8, pady=(0, 4))

        left = ttk.Frame(body)
        body.add(left, weight=3)
        self.nb = ttk.Notebook(left)
        self.nb.pack(fill="both", expand=True)

        cols = ("name", "set", "number") + PRICE_COLS + ("conf", "how")
        heads = {"name": "Card", "set": "Set", "number": "#", "conf": "Match", "how": "Method", **PRICE_HEADS}
        widths = {"name": 210, "set": 210, "number": 55, "low": 80, "low_foil": 85, "market": 80, "market_foil": 95,
                  "conf": 60, "how": 85}
        right_cols = set(PRICE_COLS) | {"conf", "number", "count"}

        cur_frame = ttk.Frame(self.nb)
        self.nb.add(cur_frame, text="On screen now")
        self.tree = ttk.Treeview(cur_frame, columns=cols, show="headings", selectmode="browse")
        for c in cols:
            self.tree.heading(c, text=heads[c])
            self.tree.column(c, width=widths[c], anchor="e" if c in right_cols else "w")
        vs = ttk.Scrollbar(cur_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vs.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vs.pack(side="right", fill="y")
        self.tree.tag_configure("strong", foreground="#0a6b0a")
        self.tree.tag_configure("weak", foreground="#9a5b00")
        self.tree.tag_configure("none", foreground="#a00000")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-1>", lambda e: self.open_selected())

        hist_frame = ttk.Frame(self.nb)
        self.nb.add(hist_frame, text="Seen this session")
        hcols = ("name", "set", "number") + PRICE_COLS + ("count", "first")
        hheads = {"name": "Card", "set": "Set", "number": "#", "count": "Times", "first": "First seen", **PRICE_HEADS}
        self.hist = ttk.Treeview(hist_frame, columns=hcols, show="headings", selectmode="browse")
        for c in hcols:
            self.hist.heading(c, text=hheads[c])
            self.hist.column(c, width=widths.get(c, 80), anchor="e" if c in right_cols else "w")
        hvs = ttk.Scrollbar(hist_frame, orient="vertical", command=self.hist.yview)
        self.hist.configure(yscrollcommand=hvs.set)
        self.hist.pack(side="left", fill="both", expand=True)
        hvs.pack(side="right", fill="y")
        self.hist.bind("<Double-1>", lambda e: self.open_selected(history=True))

        actions = ttk.Frame(left)
        actions.pack(fill="x", pady=(4, 0))
        ttk.Button(actions, text="Open on TCGplayer", command=self.open_selected).pack(side="left")
        ttk.Button(actions, text="Export history CSV...", command=self.export_csv).pack(side="left", padx=6)
        ttk.Button(actions, text="Clear history", command=self.clear_history).pack(side="left")

        right = ttk.Frame(body)
        body.add(right, weight=2)
        ttk.Label(right, text="Capture preview").pack(anchor="w")
        self.preview = tk.Label(right, bg="#202020", width=56, height=18)
        self.preview.pack(fill="both", expand=True)
        ttk.Label(right, text="Other printings with the same look (double-click to use)").pack(anchor="w", pady=(6, 0))
        self.alts = tk.Listbox(right, height=6, activestyle="dotbox")
        self.alts.pack(fill="x")
        self.alts.bind("<Double-1>", self._use_alternative)
        self.detail = ttk.Label(right, text="", wraplength=420, justify="left")
        self.detail.pack(anchor="w", pady=(4, 0))

        self.status = ttk.Label(self.root, text="Loading index...", anchor="w", relief="sunken", padding=(6, 2))
        self.status.pack(side="bottom", fill="x")

    def _apply_settings_to_widgets(self):
        self.interval_var.set(str(self.settings.get("interval", 2.0)))
        self.single_var.set(bool(self.settings.get("single_card", False)))
        self.ocr_var.set(bool(self.settings.get("use_ocr", True)))
        self.pricing_var.set(MODES.get(self.settings.get("pricing", MODE_TCGCSV), MODES[MODE_TCGCSV]))
        self._update_region_label()

    def _settings_changed(self):
        try:
            self.settings["interval"] = max(0.3, float(self.interval_var.get()))
        except ValueError:
            pass
        self.settings["single_card"] = bool(self.single_var.get())
        self.settings["use_ocr"] = bool(self.ocr_var.get())
        for key, label in MODES.items():
            if label == self.pricing_var.get():
                self.settings["pricing"] = key
        if self.prices.set_mode(self.settings["pricing"]):
            self._rerequest_prices()
        if self.index is not None:
            self.identifier = Identifier(self.index, self.ocr, use_ocr=self.settings["use_ocr"])
        save_settings(self.settings)
        self._update_status()

    def _update_region_label(self):
        r = self.settings.get("region")
        self.region_lbl.configure(text=f"{r[2]}x{r[3]} at ({r[0]},{r[1]})" if r else "no region selected")

    def _tcg_keys(self) -> tuple[str, str]:
        pub = self.settings.get("tcgplayer_public_key") or ""
        priv = self.settings.get("tcgplayer_private_key") or ""
        if pub and priv:
            return pub, priv
        return keys_from_env()

    def _rerequest_prices(self):
        """Price cache is per mode: ask again for everything on screen and in history."""
        cards = {d.card.id: d.card for d in self.current if d.card}
        cards.update({cid: e.card for cid, e in self.history.items()})
        for e in self.history.values():
            e.quote = self.prices.cached(e.card)
        for card in cards.values():
            self.prices.request(card, lambda c, q: self.events.put(("price", c, q)))
        self._refresh_current()
        self._refresh_history()

    def edit_keys(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("TCGplayer API keys")
        dlg.resizable(False, False)
        dlg.transient(self.root)
        frm = ttk.Frame(dlg, padding=12)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, justify="left", text=(
            "Only needed for the 'official API' price mode. TCGplayer issues keys to\n"
            "approved developers; if you have a pair, paste it here. Keys are stored in\n"
            "settings.json in the app data folder (or set TCGPLAYER_PUBLIC_KEY /\n"
            "TCGPLAYER_PRIVATE_KEY in the environment).")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        pub = tk.StringVar(value=self.settings.get("tcgplayer_public_key", ""))
        priv = tk.StringVar(value=self.settings.get("tcgplayer_private_key", ""))
        ttk.Label(frm, text="Public key").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm, textvariable=pub, width=48).grid(row=1, column=1, sticky="w", pady=2)
        ttk.Label(frm, text="Private key").grid(row=2, column=0, sticky="w")
        ttk.Entry(frm, textvariable=priv, width=48, show="*").grid(row=2, column=1, sticky="w", pady=2)

        def save():
            self.settings["tcgplayer_public_key"] = pub.get().strip()
            self.settings["tcgplayer_private_key"] = priv.get().strip()
            save_settings(self.settings)
            self.prices.set_keys(*self._tcg_keys())
            self._rerequest_prices()
            dlg.destroy()

        btns = ttk.Frame(frm)
        btns.grid(row=3, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(btns, text="Save", command=save).pack(side="left", padx=4)
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side="left", padx=4)

    # -- index -------------------------------------------------------------------------
    def _load_index(self):
        try:
            if not HashIndex.exists():
                self.events.put(("index", None, None, "No card index yet. Click 'Build / update index...' to create one."))
                return
            idx = HashIndex()
            ocr = Ocr()
            self.events.put(("index", idx, ocr, ""))
        except Exception as e:
            log.exception("index load failed")
            self.events.put(("index", None, None, f"Index failed to load: {e}"))

    def _on_index(self, idx: Optional[HashIndex], ocr: Optional[Ocr], msg: str):
        self.index, self.ocr = idx, ocr
        if idx is not None:
            self.identifier = Identifier(idx, ocr, use_ocr=self.settings.get("use_ocr", True))
            self.start_btn.configure(state="normal")
            if ocr is not None and not ocr.available:
                self.ocr_chk.configure(state="disabled")
                self.ocr_var.set(False)
        else:
            self.identifier = None
            self.start_btn.configure(state="disabled")
        self._update_status(msg)

    def open_build_dialog(self):
        if self.build_dialog is not None and self.build_dialog.winfo_exists():
            self.build_dialog.lift()
            return
        self.build_dialog = BuildDialog(self)

    # -- scanning ----------------------------------------------------------------------
    def select_region(self):
        was_scanning = self.worker is not None and self.worker.is_alive()
        self.root.withdraw()
        self.root.update_idletasks()
        time.sleep(0.15)

        def done(region):
            self.root.deiconify()
            if region:
                self.settings["region"] = list(region)
                save_settings(self.settings)
                self._update_region_label()
                self.force_scan.set()
                if not was_scanning and self.identifier is not None:
                    self.toggle_scan()
            self._update_status()

        RegionSelector(self.root, self.capture.virtual_screen(), done)

    def toggle_scan(self):
        if self.worker is not None and self.worker.is_alive():
            self.stop_evt.set()
            self.worker.join(timeout=3)
            self.worker = None
            self.start_btn.configure(text="Start scanning")
            self._update_status("Stopped.")
            return
        if not self.settings.get("region"):
            self.select_region()
            return
        self.stop_evt.clear()
        self.force_scan.set()
        self.worker = threading.Thread(target=self._scan_loop, name="scanner", daemon=True)
        self.worker.start()
        self.start_btn.configure(text="Stop scanning")
        self._update_status("Scanning...")

    def scan_now(self):
        if self.worker is None or not self.worker.is_alive():
            self.toggle_scan()
        self.force_scan.set()

    def _scan_loop(self):
        prev_sig = None
        while not self.stop_evt.is_set():
            region = self.settings.get("region")
            ident = self.identifier
            if region and ident is not None:
                try:
                    frame = self.capture.grab(tuple(region))
                    sig = frame_signature(frame)
                    if self.force_scan.is_set() or frames_differ(prev_sig, sig):
                        self.force_scan.clear()
                        prev_sig = sig
                        t0 = time.perf_counter()
                        dets = ident.identify_frame(frame, single_card=bool(self.settings.get("single_card")))
                        self.events.put(("frame", frame, dets, time.perf_counter() - t0))
                except Exception as e:
                    log.exception("scan failed")
                    self.events.put(("status", f"Scan error: {e}"))
            try:
                interval = float(self.settings.get("interval", 2.0))
            except (TypeError, ValueError):
                interval = 2.0
            end = time.monotonic() + interval
            while time.monotonic() < end and not self.stop_evt.is_set() and not self.force_scan.is_set():
                time.sleep(0.05)

    # -- event pump --------------------------------------------------------------------
    def _poll_events(self):
        try:
            while True:
                ev = self.events.get_nowait()
                kind = ev[0]
                if kind == "frame":
                    self._on_frame(ev[1], ev[2], ev[3])
                elif kind == "price":
                    self._on_price(ev[1], ev[2])
                elif kind == "status":
                    self._update_status(ev[1])
                elif kind == "index":
                    self._on_index(ev[1], ev[2], ev[3])
                elif kind == "build":
                    self._on_build(ev[1])
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _on_build(self, p: Progress):
        if self.build_dialog is not None and self.build_dialog.winfo_exists():
            self.build_dialog.on_progress(p)
        if p.phase == "done":
            self._update_status("Index built; loading...")
            threading.Thread(target=self._load_index, daemon=True).start()
        elif p.phase == "error":
            self._update_status(p.message)

    def _on_frame(self, frame, dets: list[Detection], secs: float):
        self.current = dets
        self._last_frame = frame
        self._render_preview(frame, dets)
        self._refresh_current()
        now = time.time()
        ids_now = {d.card.id for d in dets if d.card}
        new_ids = ids_now - self._last_ids
        for d in dets:
            if not d.card:
                continue
            e = self.history.get(d.card.id)
            if e is None:
                self.history[d.card.id] = HistoryEntry(d.card, now, now, 1, self.prices.cached(d.card))
            else:
                e.last_seen = now
                if d.card.id in new_ids:
                    e.count += 1
            self.prices.request(d.card, lambda c, q: self.events.put(("price", c, q)))
        self._last_ids = ids_now
        self._refresh_history()
        found = sum(1 for d in dets if d.card)
        self._update_status(f"Last scan: {len(dets)} card shape(s), {found} identified in {secs * 1000:.0f} ms")

    def _on_price(self, card: CardRecord, quote: PriceQuote):
        e = self.history.get(card.id)
        if e is not None:
            e.quote = quote
        self._refresh_current()
        self._refresh_history()

    # -- tables -------------------------------------------------------------------------
    def _row_for(self, d: Detection) -> tuple[tuple, str]:
        if not d.card:
            return (("(unrecognised)", "", "") + ("",) * len(PRICE_COLS) + ("", d.note[:40]), "none")
        q = self.prices.cached(d.card)
        conf = f"{d.confidence * 100:.0f}%"
        tag = "strong" if d.confidence >= 1 - MATCH_STRONG else "weak"
        how = d.method + (" *" if d.ambiguous else "")
        prices = tuple(_money(q, c, d.card) for c in PRICE_COLS)
        return ((d.card.name, d.card.set_label, d.card.collector_number) + prices + (conf, how), tag)

    def _refresh_current(self):
        sel = self.tree.selection()
        sel_idx = int(sel[0]) if sel and sel[0].isdigit() else None
        self.tree.delete(*self.tree.get_children())
        for i, d in enumerate(self.current):
            values, tag = self._row_for(d)
            self.tree.insert("", "end", iid=str(i), values=values, tags=(tag,))
        if sel_idx is not None and sel_idx < len(self.current):
            self.tree.selection_set(str(sel_idx))

    def _refresh_history(self):
        sel = self.hist.selection()
        self.hist.delete(*self.hist.get_children())
        entries = sorted(self.history.values(), key=lambda e: e.first_seen, reverse=True)
        for e in entries:
            first = dt.datetime.fromtimestamp(e.first_seen).strftime("%H:%M:%S")
            prices = tuple(_money(e.quote, c, e.card) for c in PRICE_COLS)
            self.hist.insert("", "end", iid=e.card.id, values=(
                (e.card.name, e.card.set_label, e.card.collector_number) + prices + (e.count, first)))
        if sel and self.hist.exists(sel[0]):
            self.hist.selection_set(sel[0])

    def _selected_detection(self) -> Optional[Detection]:
        sel = self.tree.selection()
        if not sel or not sel[0].isdigit():
            return None
        i = int(sel[0])
        return self.current[i] if i < len(self.current) else None

    def _on_select(self, _e=None):
        d = self._selected_detection()
        self.alts.delete(0, "end")
        if d is None:
            self.detail.configure(text="")
            return
        for c in d.candidates:
            mark = ">" if d.card is c.card else " "
            self.alts.insert("end", f"{mark} {c.card.name} - {c.card.set_label} #{c.card.collector_number} {c.label.split(' ', 1)[-1] if c.distance is not None else ''}")
        q = self.prices.cached(d.card) if d.card else None
        bits = []
        if d.card:
            bits.append(f"{d.card.name} - {d.card.set_label} #{d.card.collector_number} ({d.card.rarity}, {d.card.released_at})")
            bits.append(f"Finishes: {d.card.finishes or '-'}")
        if q is not None:
            bits.append(q.summary())
        elif d.card and (d.card.usd is not None or d.card.usd_foil is not None):
            bits.append(f"~ daily market price from the index (as of {d.card.prices_at[:10] or '?'}); live quote pending")
        if d.ocr_set or d.ocr_number:
            bits.append(f"Collector line read: set={d.ocr_set or '?'} number={d.ocr_number or '?'}")
        if d.note:
            bits.append(d.note)
        self.detail.configure(text="\n".join(bits))

    def _use_alternative(self, _e=None):
        d = self._selected_detection()
        sel = self.alts.curselection()
        if d is None or not sel or sel[0] >= len(d.candidates):
            return
        cand = d.candidates[sel[0]]
        d.card = cand.card
        d.method = d.method.split("+")[0] + "+manual"
        self.prices.request(d.card, lambda c, q: self.events.put(("price", c, q)))
        if d.card.id not in self.history:
            now = time.time()
            self.history[d.card.id] = HistoryEntry(d.card, now, now, 1, self.prices.cached(d.card))
        self._refresh_current()
        self._refresh_history()
        self._on_select()

    def open_selected(self, history: bool = False):
        url = ""
        if history:
            sel = self.hist.selection()
            if sel and sel[0] in self.history:
                url = self.history[sel[0]].card.tcgplayer_product_url
        else:
            d = self._selected_detection()
            if d and d.card:
                url = d.card.tcgplayer_product_url
        if url:
            webbrowser.open(url)

    def export_csv(self):
        if not self.history:
            messagebox.showinfo("Export", "Nothing seen yet.", parent=self.root)
            return
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                            initialfile=f"mtg-scan-{dt.date.today().isoformat()}.csv")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name", "set_code", "set_name", "collector_number", "low", "low_foil", "market", "market_foil",
                        "low_kind", "price_source", "times_seen", "first_seen", "scryfall_id", "tcgplayer_url"])

            def num(v):
                return "" if v is None else f"{v:.2f}"

            for e in sorted(self.history.values(), key=lambda e: e.first_seen):
                q = e.quote
                if q is None or q.source == "unavailable":
                    low, low_foil, market, market_foil, kind, src = "", "", num(e.card.usd), num(e.card.usd_foil), "", "index (daily)"
                else:
                    low, low_foil = num(q.low), num(q.low_foil if q.low_foil is not None else q.low_etched)
                    market = num(q.market if q.market is not None else e.card.usd)
                    market_foil = num(q.market_foil if q.market_foil is not None else (q.market_etched if q.market_etched is not None else e.card.usd_foil))
                    kind, src = q.low_kind, q.source
                w.writerow([e.card.name, e.card.set_code.upper(), e.card.set_name, e.card.collector_number,
                            low, low_foil, market, market_foil, kind, src, e.count,
                            dt.datetime.fromtimestamp(e.first_seen).isoformat(timespec="seconds"),
                            e.card.id, e.card.tcgplayer_product_url])
        self._update_status(f"Exported {len(self.history)} cards to {path}")

    def clear_history(self):
        self.history.clear()
        self._last_ids = set()
        self._refresh_history()

    # -- preview / status --------------------------------------------------------------------
    def _render_preview(self, frame, dets):
        try:
            annotated = draw_detections(frame, dets)
            h, w = annotated.shape[:2]
            pw = max(200, self.preview.winfo_width() - 4)
            ph = max(150, self.preview.winfo_height() - 4)
            scale = min(pw / w, ph / h, 1.0)
            if scale < 1.0:
                annotated = cv2.resize(annotated, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
            img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            self._preview_photo = ImageTk.PhotoImage(img)
            self.preview.configure(image=self._preview_photo)
        except Exception as e:
            log.warning("preview failed: %s", e)

    def _update_status(self, msg: str = ""):
        parts = []
        if self.index is not None:
            st = self.index.stats()
            parts.append(f"Index: {st['hashed']:,} printings (prices as of {st['prices_at'][:10] or '?'})")
        else:
            parts.append("Index: none")
        parts.append("OCR: " + ("on" if (self.identifier and self.identifier.ocr) else "off"))
        parts.append("Prices: " + MODE_SHORT.get(self.settings.get("pricing", ""), "?"))
        if msg:
            parts.append(msg)
        self.status.configure(text="  |  ".join(parts))

    def on_close(self):
        self.stop_evt.set()
        save_settings(self.settings)
        self.root.destroy()


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    enable_dpi_awareness()
    root = tk.Tk()
    if sys.platform == "win32":
        try:  # keep fonts readable on 125%/150% displays now that the process is DPI aware
            root.tk.call("tk", "scaling", ctypes.windll.user32.GetDpiForSystem() / 72.0)
        except Exception:
            pass
    ScannerApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
