# Changelog

All notable changes to mtg-screen-scanner are recorded here. Newest entries first.

## Unreleased

### User-facing (2026-11-16, index and diagnosis)
- **A prebuilt index can be downloaded instead of built.** About 21 MB, installed in roughly
  ten seconds, against about ninety minutes of hashing ~107k thumbnails. Available as
  **Build / update index... > Download prebuilt index** or `build_index.py --import-bundle`.
  It carries hashes and card data only, never card images.
- **Refreshing the index no longer re-downloads every image.** Hashes were keyed on the image
  URL, which carries a cache-busting timestamp that changes on every Scryfall bulk export, so
  a refresh invalidated all ~107k hashes and started again from nothing. Only a card whose
  picture actually changed is re-hashed now.
- **Stopping a build keeps the index you already had**, instead of publishing its partial
  hashes over a complete one.
- `build_index.py --plan` reports the data folder, what is current and what needs downloading,
  without downloading anything. `--export-bundle` publishes an index for other machines.
- **The window now says where it is looking.** The data folder appears in the status bar and
  the build dialog, and a log is written to `mtgscan.log` in that folder recording the folder,
  whether an index was found, and whether OCR loaded.
- **The preview shows the captured region even with no index loaded**, so a black preview no
  longer means "no index", "not scanning" and "capture is broken" at once.
- **The OCR indicator distinguishes its states**: loading, not installed, off, idle, on. It
  previously read "off" for all of them, including when OCR was working.
- OCR loads independently of the index, so an OCR problem can no longer leave the app with no
  index and no explanation.

### Internal (2026-11-16, index and diagnosis)
- `scryfall.image_key()` strips the query from an image URL; `CardRecord.image_key` stores it.
  `cards.hash_key` records the picture each hash was computed from, and `index.STALE_HASH_SQL`
  (`hash IS NULL OR hash_key IS NOT image_key`) is the single definition of a stale hash, used
  by the build, the export and the planner. The upsert no longer touches hashes at all.
- `index._migrate()` adds columns to an older database and backfills `image_key`/`hash_key` so
  existing hashes survive an upgrade; `HashIndex` migrates on open.
- `index.restore_hashes_from_arrays()` repairs a database that lost its hashes from the
  exported arrays, and the build calls it before downloading anything.
- `index.export_bundle()` / `import_bundle()` / `bundle_url()` and the `.mtgidx` format
  (zip of `manifest.json`, gzipped card CSV, `hashes.npy`); `plan_build()` for the dry run.
- `ui.setup_logging()`; `_load_index` and `_load_ocr` are separate threads with their own
  events; `_rebuild_identifier()` replaces three copies of that logic; `_ocr_state()`;
  `_scan_loop` emits frames with an empty detection list when no identifier is loaded.
- `capture` uses `mss.MSS` where available, silencing a deprecation warning.
- Tests: `test_index_maintenance.py` (staleness, repair, migration, planning, bundle round
  trip) and `test_ui_no_index.py` (the no-index window). 54 tests to 72.

### Fixed (2026-11-16, region modes)
- **"Region is one card" could make identification impossible.** It treated the region as a
  whole card and tried nothing else, so a region holding part of a card, which is what a fixed
  region catches when a hand-held card drifts on a stream, never matched. Each mode now falls
  back to the other: single-card mode searches inside the region, and detection mode reads the
  region as one card. Confirmed on a live stream, where the same frame went from no match to
  identifying Sol Ring.
- The status bar now names this specific case and says what to change, instead of suggesting a
  larger region.

### User-facing (2026-11-16, live video)
- **Cards on a live video stream are now identified.** A filmed card matches far less exactly
  than a digital image, so it was being rejected by a fixed distance cutoff even when correct.
  A match is now accepted on how far it leads the nearest different artwork, which separates a
  real card from noise cleanly where absolute distance does not.
- **The reading is held across frames.** A stream gives many looks at one card and some are
  unreadable; the table no longer flickers between a card and nothing, and shows `held Ns`
  while relying on earlier frames. The card several frames agree on beats a one-frame fluke.
- "Use OCR" no longer stays off for good after being turned on. When the package was missing
  the app unticked it, and the next settings change saved that as a deliberate preference.
- README gained a "Scanning live video" section with the measured numbers and the limits.

### Internal (2026-11-16, live video)
- `config.MATCH_RELAXED` (0.33) and `MATCH_MIN_MARGIN` (28) alongside the existing
  `MATCH_ACCEPT`. `Identifier._accept()` takes a distance and a margin;
  `Identifier._margin()` returns how far the nearest different `illustration_id` trails the
  winner, skipping reprints of the same art, and reports a very large margin when no other
  artwork appears in the candidates. Candidate queries widened from k=15 to k=40 so a
  competing artwork is actually present. `Detection.margin` added, and a match accepted this
  way reports its method as `hash+margin`.
- `identify.TemporalVoter` collects votes over a 12 second window, needs 2 agreeing frames,
  keeps the clearest look at each card and expires silently. The UI holds the stable card when
  a frame yields nothing, while the preview still draws the current frame's own geometry.
- Measured, and the basis for those constants: a live auction stream gave a correct filmed
  card at distance 106 with a margin of 58; 36 frames with no card in them gave 189 or more
  with margins of 0 to 6; after the change the real frame identifies correctly at 121 with a
  margin of 43, with no false positives across those 36 frames.
- Art-crop matching was measured and rejected. On the same real frame the true card scored 194
  by artwork with a margin of 7, against 106 and 58 for the whole card, so the second index it
  would have required was not built.
- `tests/test_matching.py` covers the acceptance rule and the voter; `tests/test_identify_e2e.py`
  gained a webcam simulation (tilt, glare, exposure, blur, compression) and a noise-rejection
  test. 41 tests to 53.

### Fixed (2026-11-16, first real-screen testing)
- **Selecting a region produced nothing at all, with no error.** The OCR fallback passed a
  numpy array through `or`, which raises, and the per-card handler swallowed it, so every
  card vanished and the table stayed empty. Fixed, and a card that fails now appears as a row
  carrying the error instead of disappearing.
- The status bar now explains an empty result (no card shapes, nothing matched, or the error),
  rather than only counting shapes.
- **A wrong card could be reported with confidence.** A partially read title matched a token
  card of the same short name ("Ragavan" for "Ragavan, Nimble Pilferer"). Tokens and emblems
  are now excluded from title matching, whole-string scoring replaced substring scoring, and
  very short reads are ignored. For a tool that quotes prices, no answer beats a wrong one.
- **Cards on a dark background were not found.** A card's black border against a dark
  backdrop is a very weak edge, so the contour step returned a lopsided quad spanning the
  frame, and the hash is intolerant of a few percent of misalignment. On a real screen this
  meant nothing was identified unless the region was drawn exactly on the card.

### Internal (2026-11-16, detection)
- `detect.quad_is_sane()` rejects quads whose opposite sides differ by more than 20% or whose
  corners stray outside 55-125 degrees; the aspect test alone averaged opposite sides and let
  skewed noise through. `find_cards` gained a CLAHE-equalised, low-threshold Canny pass for
  weak edges.
- `detect.projection_quads()` finds upright cards by projecting Sobel energy onto each axis
  and pairing peaks whose spacing matches 63:88. Summing along whole rows and columns
  accumulates an edge too weak for per-pixel thresholding. `detect.axis_aligned()` offers a
  quad's bounding box as an alternative reading.
- `identify.identify_card()` now tries several geometries and lets the hash choose. Fallback
  order in `identify_frame`: detected quads, then the whole frame when card-shaped, then
  `_projection_fallback()` (projection peaks land on the card's inner frame, so a short
  outward expansion is searched, 4 steps x 3 candidates, one index search each), then the
  title. Measured on a real screen: distance 169 before the expansion, 77 after, against an
  accept threshold of 112.
- Title matching moved out of `identify_card` into a once-per-frame step that runs only when
  no geometry matched; previously it ran per card and its wrong answer masked the correct one
  the later fallbacks would have found. `HashIndex.unique_names()` excludes tokens.

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
