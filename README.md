# MTG Screen Scanner

A Windows desktop app that watches a rectangle of your screen, finds Magic: The Gathering
cards in it, identifies the exact printing and shows the card name, set and Near Mint price
(normal and foil) in a table. Point it at a stream, a video of a pack opening, a marketplace
page or a photo gallery and it keeps the list up to date on a timer.

## How it works (and why)

The approach follows what phone scanners and the open-source recognizers do:

1. **Find the cards.** OpenCV edge/threshold detection, 4-point contours with a 63:88 aspect
   ratio, perspective-warped to an upright 292x408 card. Three binarisations are tried so cards
   on white pages, dark overlays and photographed tables all work. If no card shape is found and
   the region itself is card shaped, the whole region is treated as one card (or tick
   "Region is one card").
2. **Match the artwork with perceptual hashes.** Every printing's Scryfall image is reduced to a
   512-bit hash (DCT pHash + gradient dHash). A screen capture of the same card differs in only a
   few bits, a different card in roughly half. Searching 100k hashes is a single numpy Hamming
   distance pass (milliseconds). This is the technique used by Timo Ikonen's
   [magic_card_detector](https://tmikonen.github.io/quantitatively/2020-01-01-magic-card-detector),
   [YamCR](https://github.com/ForOhForError/Yet-Another-Magic-Card-Recognizer) (as its first pass),
   [mtg-detector](https://gh.r-mj.com/jew256/mtg-detector) and
   [GathererImageGatherer](https://github.com/Machine-Learning-Labs/GathererImageGatherer).
   ManaBox's own docs say its scanner also
   [identifies cards by artwork](https://manabox.app/guides/scanner/faq), which is why it (and
   this app) can confuse reprints that share art.
3. **Pick the printing.** Reprints with the same art in the same frame hash alike. Like
   [TCG Machines' sorting robots](https://tcgmachines.com/blog/our-robots-can-read), the app
   optionally OCRs the collector line printed at the bottom-left of cards since Magic 2015
   ("0146/0264" and "M15 - EN") to choose the exact set. When that is not readable the closest
   hash wins and the other candidates are listed so you can pick by hand, which is also what
   ManaBox does.
4. **Price it, at scan time.** Prices are never baked into the index. Each identified card is
   looked up when it appears (cached 15 minutes). Scryfall supplies the TCGplayer product id;
   [tcgcsv.com](https://tcgcsv.com) republishes TCGplayer's official daily price points for
   that product, so the table shows TCGplayer's **Low** (lowest listing) and **Market** price
   for the normal and foil printings. See Prices below for the other sources.

Scryfall is the card database: its bulk dump covers every printing with images,
`illustration_id`, `tcgplayer_id`, finishes and daily prices.

## Requirements

- Windows 10/11, Python 3.11+ ([uv](https://docs.astral.sh/uv/) or python.org).
- Internet for building the index and for prices. Scanning itself is offline.

## Setup

```bash
uv venv --python 3.12 .venv
```

```bash
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

Optional OCR (set disambiguation and title fallback), a plain pip package with bundled models:

```bash
uv pip install --python .venv/Scripts/python.exe rapidocr-onnxruntime
```

## Get the card index (once)

The quickest route is the prebuilt index, about 21 MB, which installs in well under a minute:

```bash
.venv/Scripts/python.exe build_index.py --import-bundle
```

The same thing is in the app under **Build / update index... > Download prebuilt index**. It
carries 512-bit image hashes and Scryfall card data, never card images. Afterwards an ordinary
build only fetches printings released since the bundle was made.

To see what a build would do before it downloads anything:

```bash
.venv/Scripts/python.exe build_index.py --plan
```

That prints the data folder, how many printings are current, and how many need downloading.

## Build the card index yourself

Card data comes from Scryfall's daily bulk dump (`default_cards`, an 81 MB gzipped JSON-lines
file, no API calls). What takes time is the images: the index hashes one 146x204 thumbnail
per printing (about 100k paper printings), fetched from Scryfall's image CDN at 20 per
second (`--rate`, backs off automatically if the CDN pushes back). A full build takes about
an hour and a half; it is resumable and later runs only add new printings. Start smaller if
you only care about recent sets:

```bash
.venv/Scripts/python.exe build_index.py --since 2022
```

```bash
.venv/Scripts/python.exe build_index.py --sets mh3,blb,dsk,fdn
```

```bash
.venv/Scripts/python.exe build_index.py
```

The same options are in the app under **Build / update index...**. Re-run any time to pick up
new sets and refresh the bundled daily prices; only printings whose picture is new are
downloaded again, so a refresh is quick. Stopping a build keeps the index you already had.

Data lives in `%LOCALAPPDATA%\mtg-screen-scanner` (`cards.sqlite`, `hashes.npy`, and
`mtgscan.log`); set `MTGSCAN_DATA_DIR` to move it. The app shows the folder it is using in the
status bar and in the build dialog, and writes the same to the log at startup.

To publish your own index for others, or to move one between machines:

```bash
.venv/Scripts/python.exe build_index.py --export-bundle card-index.mtgidx
```

## Run

```bash
.venv/Scripts/python.exe run.py
```

1. **Select region...** dims the screen; drag a rectangle over the area to watch. Esc cancels.
2. **Start scanning** captures the region every N seconds (default 2). Unchanged frames are
   skipped, so idle scanning is cheap. **Scan now** forces a pass.
3. The **On screen now** tab lists each detected card with set, collector number, Low and
   Market price for the normal and foil printing, match confidence and method. A `~` price is
   the daily figure stored in the index, shown instantly until the live quote replaces it.
   Green rows are confident hash matches; orange rows are weaker or were resolved by OCR; a `*`
   in Method means several printings look identical and the right-hand panel lists them,
   double-click one to use it instead.
4. **Seen this session** accumulates every card with how often it appeared; export it as CSV.
   Double-click a row to open the TCGplayer product page.

## Prices

All sources are queried at scan time. The **Prices** dropdown selects the source; the detail
panel shows what the selected row's quote came from, and the app falls back to Scryfall's
market price automatically when a source has nothing for a card or is down.

| Mode | Low columns | Market columns | Notes |
| --- | --- | --- | --- |
| TCGplayer low + market (tcgcsv.com, daily) | TCGplayer's lowest listing, any condition | TCGplayer market | **Default.** [tcgcsv.com](https://tcgcsv.com) mirrors the official TCGplayer price points daily and is free to use. Needs no keys. One request per set, cached an hour. |
| JustTCG Near Mint (6-hourly, free key, 100/day) | Near Mint price per finish as [JustTCG](https://justtcg.com) tracks it from TCGplayer | TCGplayer market (Scryfall's daily figure) | **The freshest condition-level figure available without a TCGplayer key:** JustTCG updates about every six hours. Needs a free JustTCG key (**API keys...**): 1,000 requests a month, 100 a day, 10 a minute. One request per card, kept for six hours; the price window shows how many requests are left today. When the day's quota is spent the app falls back to tcgcsv until midnight UTC and says so. |
| TCGplayer market only (Scryfall, daily) | none | TCGplayer market | What Scryfall's `usd`/`usd_foil` are: the market price, synced every 24 hours ([Scryfall FAQ](https://scryfall.com/docs/faqs/where-do-scryfall-prices-come-from-7)). |
| TCGplayer lowest Near Mint (official API) | lowest Near Mint listing per finish | NM market | Needs a TCGplayer API key pair (**API keys...**). TCGplayer rarely issues keys any more; written from the public docs, untested without keys. |
| TCGplayer lowest Near Mint (website data) | lowest Near Mint listing per finish | none | **Does not work: TCGplayer answers these requests with HTTP 403.** Kept in case that changes; the app falls back to the market price and says so in the status bar. |

Condition note: a Near Mint-only lowest price exists only at TCGplayer's SKU level. TCGplayer's
best grade is Near Mint (there is no "Mint"). Cards without a foil printing show `n/a` in the
foil columns.

### Getting a Near Mint price

TCGplayer refuses automated requests for its listings data, and getting past that would mean
impersonating a browser, which this project will not do. Two honest routes remain:

- **In the app:** the JustTCG mode above, with a free key. **Price details** (the button, or
  Enter on a row) shows the Near Mint figure for each finish, how old it is and the quota left;
  **Refresh now** fetches it again on the spot.
- **On the page:** double-clicking a row opens the product page already filtered to English
  Near Mint in the normal printing, and the price window links there too. The lowest Near Mint
  listing is then the first figure on the page.

No free source offers true intraday TCGplayer prices; six-hourly is the ceiling this project
found after reading each candidate's own documentation (JustTCG, TCG Price Lookup, MTGJSON).

The gap is real but modest. For Lightning Bolt (M10) the any-condition low is $0.74 while the
Near Mint low is $0.89.

Check any card from the command line (add `--modes` to pick sources):

```bash
.venv/Scripts/python.exe -m mtgscan.pricing "Lightning Bolt" --set m10
```

## Scanning live video

Pointing the region at a live stream of physical cards, such as an auction broadcast, works
and is handled deliberately, but it is a harder problem than reading card images on a web
page. A filmed card carries tilt, glare, webcam exposure and video compression, so it matches
much less exactly: on a real stream a correct card scored 106 where a flat digital scan of the
same card scores about 18, against a plain cutoff of 112.

Two things make it work:

- **A match is accepted on its margin, not only its distance.** What separates a real card
  from noise is how far it stands clear of the nearest *different* artwork. Measured on a live
  stream, a correct filmed card led by 58 bits, while 36 frames containing no card at all led
  by 6 or less. So a more distant match is accepted when it leads the field convincingly, and
  a close-looking one is refused when it does not.
- **Votes are collected across frames.** A stream offers the same card dozens of times and
  some looks are ruined by a passing hand or a glare. The reading is held across those frames
  rather than flickering between a card and nothing, and the table shows `held Ns` while it
  is relying on earlier frames. The card several frames agree on wins over a one-frame fluke.

**Draw the region around the video, not tightly around the card.** On a stream the card is
usually hand-held and drifts, so a tight region ends up holding a corner of it, which can
never be matched. Give the region the area the card moves within and let the app find the card
inside it. "Region is one card" is only a hint about where to look first; if that reading
fails the app searches inside the region anyway, and the other way round.

Note that the app reads the screen, so any window placed over the region is what gets scanned.
A card is also identified only while it is fully visible: fingers across the face, a sharp
tilt or a glare covering the art will drop frames, which is what the held reading is for.

## Accuracy notes and limits

- **Draw the region close around the cards.** Measured against a card displayed on a real
  desktop, a region drawn tightly on the card, or with a modest margin, is identified; a
  region with a lot of cluttered desktop around a single card is not. Finding a card's outer
  edge is hard when its black border sits on a dark background, and the match is intolerant of
  a few percent of misalignment. Three things mitigate it: the region's bounding box is tried
  as well as the detected outline, a card-shaped region is read as a single card if detection
  finds nothing, and an edge-projection pass recovers upright cards the outline step misses.
  A clean background, such as a web page of card images, works much better than a busy desktop.
- Cards must be at least ~40 px wide on screen. The collector-line OCR needs ~300 px tall and
  is skipped below that rather than guessed at.
- Heavy glare, fingers over the art or extreme angles can push a card past the match threshold;
  it is then listed as unrecognised rather than guessed.
- **What OCR does and does not buy you.** Measured against the full index on 30 reprinted-art
  printings at 680, 500 and 380 px tall, hashing alone already picked the exact printing 28 to
  30 times out of 30, and adding OCR changed no answer for the better or worse. Its value is in
  the harder cases it is designed for, and in showing you the set code and number it read. It
  costs roughly 0.4 s per ambiguous card and only runs when several printings are in contention.
- **Some printings are genuinely indistinguishable.** Promos, token sheets and The List reuse
  the base set's scan *and* print the base set's collector line, so no amount of image
  processing separates them. The app treats those as ties, keeps the hash's pick and lists the
  alternatives for you to choose from.
- Only English, paper printings are indexed by default (`--include-digital` for Arena/MTGO
  objects, which have no TCGplayer prices anyway).

## Tests

```bash
.venv/Scripts/python.exe -m pytest -q
```

The suite downloads a handful of card images and builds a one-set index on first run
(cached in `tests/.cache`).

## Packaging as an .exe

```bash
uv pip install --python .venv/Scripts/python.exe pyinstaller
```

```bash
.venv/Scripts/pyinstaller --noconsole --name mtg-screen-scanner --collect-all rapidocr_onnxruntime run.py
```

Drop `--collect-all rapidocr_onnxruntime` if OCR is not installed.

## Project layout

```
mtgscan/
  capture.py   screen capture (mss)
  detect.py    card quad detection + perspective warp
  hashing.py   pHash/dHash, Hamming search, frame-change check
  scryfall.py  API/bulk access, CardRecord
  index.py     index build (CLI) and HashIndex queries
  ocr.py       optional RapidOCR wrapper, collector line / title parsing
  identify.py  frame -> detections pipeline
  pricing.py   price providers and PriceService
  ui.py        Tkinter application
run.py, build_index.py, tests/
```

## License

Apache License 2.0. See [LICENSE](LICENSE).

This covers the code in this repository only. It does not cover Magic: The Gathering card
data, card images or price data, none of which are redistributed here. The index is built on
your own machine from Scryfall, and card images are never committed to this repository.

## Attribution and data use

- **Card data and images: [Scryfall](https://scryfall.com).** Used under Scryfall's
  [data and image guidelines](https://scryfall.com/docs/api). This tool does not paywall or
  resell Scryfall data, does not alter card images, and identifies cards only as
  Magic: The Gathering cards. Requests carry an honest `User-Agent` and stay inside the
  documented rate limits.
- **Prices: [TCGplayer](https://www.tcgplayer.com), via [tcgcsv.com](https://tcgcsv.com) and
  Scryfall.** Prices are estimates supplied by third parties and carry no guarantee. Check the
  store before buying or selling. The optional unofficial price mode is off by default.
- Magic: The Gathering is © Wizards of the Coast LLC. This is unofficial Fan Content permitted
  under the [Wizards of the Coast Fan Content Policy](https://company.wizards.com/en/legal/fancontentpolicy),
  and is neither produced nor endorsed by Wizards of the Coast or by Scryfall or TCGplayer.
