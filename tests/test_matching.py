"""Acceptance rule and temporal voting.

Numbers here come from measurements on a live auction stream: a correct match on a
filmed card sat at distance 106 with a margin of 58 over the nearest different artwork,
while 36 frames containing no card at all sat at 189 or more with a margin of 6 or less.
"""
import time

import numpy as np
import pytest

from mtgscan.config import HASH_BITS, MATCH_ACCEPT, MATCH_MIN_MARGIN, MATCH_RELAXED
from mtgscan.identify import Identifier, TemporalVoter
from mtgscan.scryfall import CardRecord


def _card(card_id="a", name="Test Card", illustration="art-1", tcgplayer_id=None,
          finishes="") -> CardRecord:
    values = {f: "" for f in CardRecord.columns()}
    values.update(id=card_id, face=0, name=name, face_name=name, illustration_id=illustration,
                  tcgplayer_id=tcgplayer_id, tcgplayer_etched_id=None, promo=0, digital=0,
                  finishes=finishes, usd=None, usd_foil=None, usd_etched=None)
    return CardRecord(**values)


class TestPriceDescription:
    """Wording for the in-app price popup, kept testable away from the window code."""

    def test_it_lists_only_the_finishes_a_card_was_printed_in(self):
        from mtgscan.pricing import PriceQuote, describe_quote

        card = _card(tcgplayer_id=1, finishes="nonfoil")
        quote = PriceQuote(low=1.5, market=2.0, low_foil=99.0, source="test", low_kind="Near Mint")
        info = describe_quote(card, quote, age_s=10)
        assert [r["finish"] for r in info["rows"]] == ["Normal"]
        assert info["rows"][0]["low"] == "$1.50" and info["rows"][0]["market"] == "$2.00"

    def test_it_shows_foil_and_etched_when_they_exist(self):
        from mtgscan.pricing import PriceQuote, describe_quote

        card = _card(tcgplayer_id=1, finishes="nonfoil,foil,etched")
        info = describe_quote(card, PriceQuote(low=1.0, low_foil=5.0, low_etched=9.0, source="t"), 5)
        assert [r["finish"] for r in info["rows"]] == ["Normal", "Foil", "Etched"]
        assert info["rows"][2]["low"] == "$9.00"

    def test_it_reports_how_old_the_number_is(self):
        from mtgscan.pricing import PriceQuote, describe_quote

        card = _card(tcgplayer_id=1, finishes="nonfoil")
        assert describe_quote(card, PriceQuote(source="t"), 20)["freshness"] == "fetched 20s ago"
        assert describe_quote(card, PriceQuote(source="t"), 600)["freshness"] == "fetched 10 min ago"

    def test_it_names_the_source_and_what_its_low_covers(self):
        from mtgscan.pricing import PriceQuote, describe_quote

        info = describe_quote(_card(tcgplayer_id=1, finishes="nonfoil"),
                              PriceQuote(source="tcgcsv", low_kind="any condition"), 1)
        assert "tcgcsv" in info["source"] and "any condition" in info["source"]

    def test_it_says_when_no_price_has_arrived_yet(self):
        from mtgscan.pricing import describe_quote

        info = describe_quote(_card(tcgplayer_id=1, finishes="nonfoil"), None, None)
        assert "waiting" in info["source"]

    def test_it_carries_the_reason_a_source_was_skipped(self):
        from mtgscan.pricing import PriceQuote, describe_quote

        info = describe_quote(_card(tcgplayer_id=1, finishes="nonfoil"),
                              PriceQuote(source="s", note="fallback: HTTP 403"), 1)
        assert info["note"] == "fallback: HTTP 403"


class TestListingLinks:
    """The Near Mint price is reached by opening the page, since TCGplayer refuses
    automated requests for its listings data (it answers them with HTTP 403)."""

    def test_the_link_is_filtered_to_near_mint(self):
        card = _card(tcgplayer_id=32656, finishes="nonfoil,foil")
        url = card.tcgplayer_listing_url()
        assert url.startswith("https://www.tcgplayer.com/product/32656?")
        assert "Condition=Near+Mint" in url and "Printing=Normal" in url
        assert "Language=English" in url

    def test_the_foil_link_selects_the_foil_printing(self):
        url = _card(tcgplayer_id=32656, finishes="nonfoil,foil").tcgplayer_listing_url(foil=True)
        assert "Printing=Foil" in url and "Condition=Near+Mint" in url

    def test_a_card_without_a_product_id_has_no_link(self):
        assert _card().tcgplayer_listing_url() == ""


class TestAcceptance:
    def test_a_close_match_is_accepted_regardless_of_margin(self):
        assert Identifier._accept(int(0.1 * HASH_BITS), margin=0)

    def test_a_distant_match_with_a_wide_margin_is_accepted(self):
        """The filmed-card case: 106 is past the 112 cutoff once geometry shifts, but it
        stands 58 clear of every other artwork."""
        assert Identifier._accept(121, margin=43)

    def test_a_distant_match_without_a_margin_is_rejected(self):
        """A frame with no card in it: everything is roughly equidistant."""
        assert not Identifier._accept(189, margin=4)
        assert not Identifier._accept(198, margin=0)

    def test_the_relaxed_ceiling_still_applies(self):
        beyond = int(MATCH_RELAXED * HASH_BITS) + 10
        assert not Identifier._accept(beyond, margin=200)

    def test_thresholds_leave_the_measured_cases_separated(self):
        accept, relaxed = MATCH_ACCEPT * HASH_BITS, MATCH_RELAXED * HASH_BITS
        assert accept < 121 <= relaxed, "the measured filmed card must need the margin rule"
        assert MATCH_MIN_MARGIN > 6, "frames with no card showed margins up to 6"
        assert MATCH_MIN_MARGIN < 43, "the measured filmed card had a margin of 43"


class _Det:
    """Minimal stand-in for a Detection: the voter only reads card and distance."""

    def __init__(self, card, distance):
        self.card, self.distance = card, distance


class TestTemporalVoter:
    def test_one_sighting_is_not_enough(self):
        v = TemporalVoter(min_votes=2)
        v.observe([_Det(_card(), 50)], now=100.0)
        assert v.stable(now=101.0) is None

    def test_agreement_across_frames_is_held_through_bad_ones(self):
        v = TemporalVoter(window_s=12.0, min_votes=2)
        card = _card()
        v.observe([_Det(card, 120)], now=100.0)
        v.observe([_Det(card, 108)], now=102.0)
        for t in (104.0, 106.0, 108.0):
            v.observe([], now=t)          # frames the card cannot be read from
        held = v.stable(now=109.0)
        assert held is not None
        det, age = held
        assert det.card.id == card.id
        assert age == pytest.approx(7.0)
        assert det.distance == 108, "the clearest look at the card should be the one kept"

    def test_it_expires_after_the_window(self):
        v = TemporalVoter(window_s=12.0, min_votes=2)
        card = _card()
        v.observe([_Det(card, 50)], now=100.0)
        v.observe([_Det(card, 50)], now=101.0)
        assert v.stable(now=105.0) is not None
        assert v.stable(now=130.0) is None

    def test_the_card_most_frames_agree_on_wins(self):
        v = TemporalVoter(min_votes=2)
        often, once = _card("often", "Often", "art-a"), _card("once", "Once", "art-b")
        now = time.time()
        for i in range(4):
            v.observe([_Det(often, 130)], now=now + i)
        v.observe([_Det(once, 60)], now=now + 4)
        v.observe([_Det(once, 60)], now=now + 5)
        held = v.stable(now=now + 6)
        assert held is not None and held[0].card.id == "often"

    def test_clear_forgets_everything(self):
        v = TemporalVoter(min_votes=1)
        v.observe([_Det(_card(), 50)], now=100.0)
        v.clear()
        assert v.stable(now=100.5) is None
