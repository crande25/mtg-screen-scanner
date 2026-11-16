"""Collector-line parsing and printing comparison.

The raw strings below are real RapidOCR output captured from Scryfall card images at the
resolution the scanner uses, including its mistakes: merged lines, a lost separator before
the language code, a rarity letter glued to the number, and O/0 confusion. They are
fixtures so the parser can be changed without re-running OCR.
"""
import pytest

from mtgscan.ocr import (
    CollectorInfo,
    number_similarity,
    parse_collector_text,
    printing_similarity,
    set_similarity,
)


# (raw OCR text, expected set, expected number)
REAL_READS = [
    ("121/281 R AFR�EN IASON FEUX", "afr", "121"),
    ("137/279R XLN�EN WINONANELSON", "xln", "137"),
    ("251/342U CIS�EN MARTINA PILCTROYA", "cis", "251"),
    ("C0440 J2S�EN KARI CHRISTENSIN", "j2s", "440"),
    ("C0134 0202 TLA�EN RAIATER", "tla", "134"),
    ("CC 0057 MB2�EN KARLKOPINSKI", "mb2", "57"),
    ("159 V& NCC�ENALEXANDERMOKHOY", "ncc", "159"),
    ("way R0087 WOE�ENDONATO GNCOLA", "woe", "87"),
    ("R1942 SLO�EN JONYERMIIYEA", "slo", "1942"),
    ("T0002 TLA�EN LEMUONG", "tla", "2"),
]


@pytest.mark.parametrize("raw,set_code,number", REAL_READS)
def test_parses_real_ocr_output(raw, set_code, number):
    info = parse_collector_text([raw])
    assert info.set_code == set_code
    assert info.number == number


def test_zero_padded_number_beats_a_bare_one():
    """'30A*EN JAMES C0195' must yield 195, not the 30 inside the set code."""
    assert parse_collector_text(["30A�ENIA.MES C0195"]).number == "195"


def test_empty_read_is_not_a_guess():
    info = parse_collector_text([])
    assert info == CollectorInfo(None, None, "")


def test_slash_form_is_preferred():
    assert parse_collector_text(["0146/0264 R", "M15 - EN - CHRISTOPHER MOELLER"]).number == "146"


class TestSimilarity:
    def test_exact_and_unrelated(self):
        assert set_similarity("woe", "woe") == 1.0
        assert set_similarity("woe", "mh3") == 0.0
        assert number_similarity("87", "87") == 1.0
        assert number_similarity("87", "88") == 0.0

    def test_promo_and_token_sets_print_their_base_code(self):
        assert set_similarity("afr", "pafr") == pytest.approx(0.9)
        assert set_similarity("tla", "ttla") == pytest.approx(0.9)

    def test_glyph_confusions(self):
        assert set_similarity("cis", "c15") == pytest.approx(0.8)   # 1 read as I, 5 as S
        assert set_similarity("slo", "sld") == pytest.approx(0.8)   # D read as O

    def test_collector_number_suffixes(self):
        assert number_similarity("381", "381z") == pytest.approx(0.85)
        assert number_similarity("121", "121p") == pytest.approx(0.85)

    def test_the_list_numbering_matches_its_origin_set(self):
        """The List prints the original set's line and numbers itself 'EMA-198'."""
        s, n = printing_similarity("ema", "198", "plst", "EMA-198")
        assert s == 1.0 and n == 1.0

    def test_missing_read_scores_zero(self):
        assert printing_similarity(None, None, "woe", "87") == (0.0, 0.0)
