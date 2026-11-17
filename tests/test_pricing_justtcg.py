"""JustTCG provider and the service's fallback chain.

The sample answer below follows the shape documented at justtcg.com/docs/api/cards:
cards under "data", each with "variants" carrying condition, printing, price and
lastKnownPrice, and the quota left under "_metadata".
"""
import time

import pytest

from mtgscan import pricing
from mtgscan.pricing import (MODE_JUSTTCG, MODE_TCGCSV, JustTcgProvider, PriceNotFound, PriceQuote,
                             PriceService, PriceUnavailable, seconds_until_utc_midnight)
from mtgscan.scryfall import CardRecord


def _card(tcgplayer_id=32656, finishes="nonfoil,foil", etched_id=None) -> CardRecord:
    values = {f: "" for f in CardRecord.columns()}
    values.update(id="abc", face=0, name="Test Card", face_name="Test Card", illustration_id="art",
                  tcgplayer_id=tcgplayer_id, tcgplayer_etched_id=etched_id, promo=0, digital=0,
                  finishes=finishes, usd=2.5, usd_foil=7.0, usd_etched=None)
    return CardRecord(**values)


def _body(product_id=32656, variants=None, daily_left=87):
    if variants is None:
        variants = [
            {"condition": "Near Mint", "printing": "Normal", "price": 1.25, "lastKnownPrice": None},
            {"condition": "Lightly Played", "printing": "Normal", "price": 0.9, "lastKnownPrice": None},
            {"condition": "Near Mint", "printing": "Foil", "price": None, "lastKnownPrice": 4.5},
        ]
    return {"data": [{"tcgplayerId": str(product_id), "name": "Test Card", "variants": variants}],
            "meta": {"total": 1}, "_metadata": {"apiPlan": "Free", "apiRequestsRemaining": 900,
                                                 "apiDailyRequestsRemaining": daily_left}}


class TestParsing:
    def test_near_mint_price_per_printing(self):
        prices = JustTcgProvider.nm_prices(_body(), 32656)
        assert prices == {"Normal": 1.25, "Foil": 4.5}, "LP is skipped, a null price falls back to the last known one"

    def test_the_matching_product_is_picked_out_of_several(self):
        body = _body()
        body["data"].insert(0, {"tcgplayerId": "1", "variants": [
            {"condition": "Near Mint", "printing": "Normal", "price": 99.0}]})
        assert JustTcgProvider.nm_prices(body, 32656) == {"Normal": 1.25, "Foil": 4.5}

    def test_nothing_when_the_answer_is_empty(self):
        assert JustTcgProvider.nm_prices({"data": []}, 32656) == {}


class _Stub(JustTcgProvider):
    """Answers from canned bodies instead of the network, recording what was asked."""

    def __init__(self, bodies, key="tcg_test"):
        super().__init__(key)
        self.bodies, self.calls = list(bodies), []

    def _get(self, **params):
        self.calls.append(params)
        body = self.bodies.pop(0)
        if isinstance(body, Exception):
            raise body
        meta = body.get("_metadata") or {}
        self.requests_left_today = meta.get("apiDailyRequestsRemaining")
        self.requests_left_month = meta.get("apiRequestsRemaining")
        return body


class TestQuote:
    def test_low_is_near_mint_and_market_comes_from_the_index(self):
        p = _Stub([_body()])
        q = p.get(_card())
        assert (q.low, q.low_foil, q.low_kind) == (1.25, 4.5, "Near Mint")
        assert (q.market, q.market_foil) == (2.5, 7.0)
        assert "Scryfall" in q.note and "87 JustTCG requests left today" in q.note
        assert q.url.startswith("https://www.tcgplayer.com/product/32656?")

    def test_it_asks_once_per_product_filtered_to_english_near_mint(self):
        p = _Stub([_body()])
        p.get(_card())
        assert len(p.calls) == 1
        call = p.calls[0]
        assert call["tcgplayerId"] == "32656" and call["condition"] == "NM" and call["language"] == "English"
        assert call["include_price_history"] == "false", "history would only make the answer bigger"

    def test_etched_comes_from_its_own_product(self):
        etched = _body(product_id=77, variants=[{"condition": "Near Mint", "printing": "Foil", "price": 12.0}])
        p = _Stub([_body(), etched])
        q = p.get(_card(finishes="nonfoil,foil,etched", etched_id=77))
        assert q.low_etched == 12.0
        assert [c["tcgplayerId"] for c in p.calls] == ["32656", "77"]

    def test_a_card_without_a_product_id_is_not_found(self):
        with pytest.raises(PriceNotFound):
            _Stub([]).get(_card(tcgplayer_id=None))

    def test_a_product_without_near_mint_prices_is_not_found(self):
        with pytest.raises(PriceNotFound):
            _Stub([_body(variants=[])]).get(_card())

    def test_no_key_means_unavailable_not_a_crash(self):
        with pytest.raises(PriceUnavailable) as e:
            JustTcgProvider("").get(_card())
        assert "no JustTCG key" in str(e.value)


class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body

    def raise_for_status(self):
        pass


class TestErrors:
    def _provider(self, status, body):
        p = JustTcgProvider("tcg_test", per_minute=100000)
        p.s.get = lambda *a, **k: _Resp(status, body)
        return p

    def test_a_spent_daily_quota_pauses_until_midnight_utc(self):
        p = self._provider(429, {"error": "Daily limit exceeded", "code": "DAILY_LIMIT_EXCEEDED"})
        with pytest.raises(PriceUnavailable) as e:
            p.get(_card())
        assert e.value.retry_s == pytest.approx(seconds_until_utc_midnight(), abs=5)
        assert p.requests_left_today == 0

    def test_a_per_minute_limit_pauses_about_a_minute(self):
        p = self._provider(429, {"error": "Rate limit exceeded.", "code": "RATE_LIMIT_EXCEEDED"})
        with pytest.raises(PriceUnavailable) as e:
            p.get(_card())
        assert 60 <= e.value.retry_s <= 120

    def test_a_bad_key_is_said_plainly(self):
        p = self._provider(401, {"error": "Missing or invalid API key.", "code": "INVALID_API_KEY"})
        with pytest.raises(PriceUnavailable) as e:
            p.get(_card())
        assert "rejected the key" in str(e.value)

    def test_the_quota_is_read_from_the_answer(self):
        p = self._provider(200, _body(daily_left=42))
        p.get(_card())
        assert p.requests_left_today == 42 and p.requests_left_month == 900 and p.plan == "Free"
        assert p.quota_note() == "42 JustTCG requests left today, 900 this month"


class _Svc(PriceService):
    """A service whose network providers are replaced after construction."""

    def __init__(self, mode):
        super().__init__(mode=mode, cache_dir=None, justtcg_key="tcg_test")


class TestServiceFallback:
    def test_justtcg_quotes_keep_for_six_hours(self):
        svc = _Svc(MODE_JUSTTCG)
        assert svc.ttl_for(MODE_JUSTTCG) == 6 * 3600
        assert svc.ttl_for(MODE_TCGCSV) == svc.ttl_s

    def test_when_justtcg_is_paused_tcgcsv_answers_with_the_reason(self, monkeypatch):
        svc = _Svc(MODE_JUSTTCG)
        asked = []

        def just_get(card):
            asked.append(card.id)
            raise PriceUnavailable("JustTCG daily quota used up; resets at midnight UTC", retry_s=1000)

        monkeypatch.setattr(svc._just, "get", just_get)
        monkeypatch.setattr(svc._csv, "get", lambda card: PriceQuote(low=0.74, source="tcgcsv", low_kind="any condition"))
        monkeypatch.setattr(svc._scry, "get", lambda card: pytest.fail("Scryfall should not be needed"))
        q = svc.get(_card())
        assert q.source == "tcgcsv" and q.low == 0.74
        assert q.note.startswith("fallback: JustTCG daily quota used up")
        assert "quota used up" in svc.paused_reason()
        svc._cache.clear()
        svc.get(_card())
        assert asked == ["abc"], "a paused provider is not asked again before its time"

    def test_refresh_gives_a_paused_provider_another_go(self, monkeypatch):
        svc = _Svc(MODE_JUSTTCG)
        answers = [PriceUnavailable("rate limit", retry_s=65), PriceQuote(low=3.0, source="JustTCG Near Mint")]

        def just_get(card):
            a = answers.pop(0)
            if isinstance(a, Exception):
                raise a
            return a

        monkeypatch.setattr(svc._just, "get", just_get)
        monkeypatch.setattr(svc._csv, "get", lambda card: PriceQuote(low=0.74, source="tcgcsv"))
        assert svc.get(_card()).source == "tcgcsv"
        assert svc.refresh(_card()).source == "JustTCG Near Mint"

    def test_the_tcgcsv_mode_never_falls_back_to_itself(self, monkeypatch):
        svc = _Svc(MODE_TCGCSV)
        monkeypatch.setattr(svc._csv, "get", lambda card: (_ for _ in ()).throw(PriceNotFound("missing")))
        monkeypatch.setattr(svc._scry, "get", lambda card: PriceQuote(market=2.0, source="scry"))
        q = svc.get(_card())
        assert q.source == "scry" and q.note == "fallback: missing"

    def test_setting_a_key_replaces_the_provider_and_drops_its_cache(self):
        svc = _Svc(MODE_JUSTTCG)
        svc._cache[(MODE_JUSTTCG, "abc")] = PriceQuote(low=1.0, source="old")
        svc._cache[(MODE_TCGCSV, "abc")] = PriceQuote(low=1.0, source="keep")
        svc.set_justtcg_key("tcg_new")
        assert svc.justtcg.api_key == "tcg_new"
        assert list(svc._cache) == [(MODE_TCGCSV, "abc")]


def test_midnight_utc_is_at_most_a_day_away():
    s = seconds_until_utc_midnight(time.time())
    assert 0 < s <= 86400
    assert seconds_until_utc_midnight(86400 * 10 + 100) == pytest.approx(86300)
