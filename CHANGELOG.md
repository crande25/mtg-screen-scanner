# Changelog

All notable changes to mtg-screen-scanner are recorded here. Newest entries first.

## Unreleased

### Repository (2026-11-16)
- Published to GitHub as a public repository under the Apache License 2.0 (`LICENSE`).
- README gained License, and Attribution and data use, sections recording that card data and
  images come from Scryfall under its guidelines, that prices are third-party estimates from
  TCGplayer via tcgcsv.com and Scryfall, and that this is unofficial Fan Content under the
  Wizards of the Coast policy. No card data, card images or price data are redistributed; the
  index is built on the user's own machine and `tests/.cache` stays git-ignored.

### User-facing (2026-11-16, OCR pass)
- Collector-line OCR now works on real cards. It is read at up to twice the hashing
  resolution, so the printed set code and number are legible, and it is skipped entirely for
  cards under 300 px tall on screen rather than guessed at.
- Choosing between look-alike printings is now scored rather than matched exactly, so a
  misread glyph or a promo that prints its base set's code no longer sends it to the wrong
  printing. Measured on 30 reprinted-art printings at three sizes, OCR no longer changes any
  answer for the worse; previously it broke three.
- Promos, token sheets and The List reprint the base set's scan *and* its collector line.
  Those are now treated as ties and left to the hash, with the alternatives still listed.

### Internal (2026-11-16, OCR pass)
- `ocr.parse_collector_text` rewritten against real RapidOCR output: handles the merged
  two-line layout, a lost separator before the language code (`WOE?ENDONATO`), rarity letters
  glued to the number (`R0087`), and prefers zero-padded candidates so `30A ... C0195` gives
  195. `LANGS`/`RARITY` character classes replace the old slash-only patterns.
- New `set_similarity`, `number_similarity` and `printing_similarity` in `ocr`, covering glyph
  confusions (`1/I`, `5/S`, `D/O`), promo and token prefixes, collector-number suffixes
  (`381z`), and The List's embedded origin set (`EMA-198`).
- `identify`: `_ocr_image()` re-warps the quad from the frame at OCR resolution while hashing
  stays at the calibrated 292x408; `Detection.rotated` so the re-warp keeps the orientation the
  hash chose; `_pick_by_collector_line` scores all candidates with `OCR_SET_WEIGHT` /
  `OCR_NUMBER_WEIGHT`, requires `OCR_MIN_SCORE`, and defers to the hash inside
  `OCR_TIE_MARGIN` (0.5, just above the 0.45 a promo sibling scores below its base printing).
- `tests/test_ocr.py`: 20 tests over captured real OCR strings and the similarity rules; no
  OCR engine or network needed.

### User-facing (2026-11-16)
- Price table now has four columns, Low / Low foil / Market / Market foil, instead of two.
- New default price source: tcgcsv.com, a free daily mirror of TCGplayer's official price
  points, giving TCGplayer's lowest listing and market price per printing without keys.
- Market columns show the daily price stored in the index immediately (prefixed `~`) until the
  live quote arrives.
- `python -m mtgscan.pricing "Card" --set xxx` prints what every price source returns.
- Index build reads Scryfall's gzipped JSON-lines dump (81 MB instead of ~500 MB) and fetches
  images at 20/s by default (`--rate`), roughly a third of the previous build time.

### Internal (2026-11-16)
- `pricing.PriceQuote` fields renamed to `low`, `low_foil`, `low_etched`, `market`,
  `market_foil`, `market_etched`, plus `low_kind`; `PriceNotFound` (per-card fallback) split
  from `PriceUnavailable` (5-minute provider back-off); `TcgcsvProvider` with a
  product-to-group map persisted in `tcgcsv_products.json` and built in the background;
  `MODE_TCGCSV` default; `PriceService(cache_dir=...)`; `pricing.cli`.
- `scryfall.download_bulk(bulk_type, dest_dir)` now returns the file path and prefers
  `jsonl_download_uri`; `iter_bulk_cards` streams `.jsonl.gz`/`.jsonl`/`.json`;
  `set_image_rate()` and automatic halving of the image pace on HTTP 429.
- `index.BuildOptions.image_rate` (CLI `--rate`), default workers 8.
- UI: `PRICE_COLS`/`PRICE_HEADS`, `_money()` takes a column name, CSV export has the four price
  columns plus `low_kind`; status bar shows the price mode.
- Full-index reference build: 106,931 face images hashed in 95 minutes at 18.9 images/s
  (no 429s); re-checked thresholds on it with 40 random printings at 160 px and 90 px: correct
  card at most 57 bits away, nearest different artwork never under 121 bits (accept is 112),
  query 21 ms over the whole index.

### User-facing
- New Windows desktop app that scans a chosen screen rectangle on a timer, finds card-shaped
  regions, identifies Magic: The Gathering printings and lists name, set, collector number and
  Near Mint prices (normal and foil) in a table. (`python run.py`)
- Full-screen drag-to-select region picker (works across monitors, DPI aware).
- "Region is one card" mode for when the rectangle is drawn tightly around a single card.
- Per-session history tab with times seen, CSV export, double-click to open the TCGplayer page.
- Alternative printings panel: when several printings look identical, pick the right one by hand.
- Three price modes: TCGplayer market price via Scryfall (default), TCGplayer lowest Near Mint
  listing via the official API (needs keys), TCGplayer lowest Near Mint listing from website data
  (unofficial, opt-in). Falls back to Scryfall automatically with the source shown per row.
- Index builder dialog and CLI (`python build_index.py`) with scope options: everything, since a
  year, or specific sets. Resumable; later runs only hash new printings.
- Optional OCR (`pip install rapidocr-onnxruntime`) reads the bottom-left collector line to tell
  same-art reprints apart and reads the title bar as a fallback match.

### Internal
- `mtgscan.hashing`: 512-bit perceptual hash per card = 16x16 DCT pHash + 16x16 dHash on the inner
  95% of the card (own implementation on OpenCV, no imagehash/scipy dependency). Hamming search over
  the whole index with a popcount table in numpy.
- `mtgscan.detect`: contour/quad detection with Canny, adaptive and Otsu binarisations, min-area-rect
  fallback for rounded corners, IoU/containment NMS, perspective warp to 292x408 portrait.
- `mtgscan.index`: SQLite `cards` table (one row per face image) + `hashes.npy`/`rowids.npy`;
  upsert keyed on (scryfall id, face); a changed image URL clears the stored hash so rescans re-hash.
- `mtgscan.scryfall`: rate-limited session (8 API req/s, 10 image req/s), bulk download with
  `updated_at` check, streaming `ijson` parse, compact `CardRecord`.
- `mtgscan.identify`: hash match with 180-degree retry, candidate grouping within 12 bits, OCR
  disambiguation, title-OCR fallback through rapidfuzz.
- Match thresholds calibrated on 60 Magic 2010 printings at 200/120/80 px card heights:
  accept <= 0.22 of 512 bits, confident <= 0.12 (correct card 6-58 bits away, nearest wrong card
  >= 128 bits).
- `mtgscan.pricing`: `PriceService` with per-mode cache, background worker, 5-minute back-off on
  provider failure.
- `mtgscan.capture`: thread-local `mss` instances (GDI handles are per thread on Windows).
- Data lives in `%LOCALAPPDATA%/mtg-screen-scanner` (override with `MTGSCAN_DATA_DIR`).
- Tests: hashing calibration, synthetic-frame detection, end-to-end identification against a
  one-set index built from Scryfall, hidden-window UI smoke test.
