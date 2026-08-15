"""Claim resolution tests.

The scoreboard is only worth anything if settling a claim is deterministic, so
these pin the arithmetic. No model is involved in resolution and none is
involved here either - synthetic XBRL in, hit/miss out.
"""

from __future__ import annotations

from datetime import date

import pytest

from earnings_agent.scoring import Scorecard, measure, resolve
from tests.test_xbrl import facts, q

THESIS_DAY = date(2026, 5, 1)


def nand_like() -> object:
    """Two reported quarters plus their prior-year comparables."""
    from earnings_agent.xbrl import Financials

    return Financials(facts(
        Revenues=[
            q("2025-01-01", "2025-03-31", 1_000),   # prior year, Q1
            q("2025-04-01", "2025-06-30", 1_100),   # prior year, Q2
            q("2026-01-01", "2026-03-31", 2_000),   # base quarter
            q("2026-04-01", "2026-06-30", 2_200),   # base + 1
        ],
        GrossProfit=[q("2026-04-01", "2026-06-30", 1_320)],       # 60% of 2200
        OperatingIncomeLoss=[q("2026-04-01", "2026-06-30", 880)],  # 40%
        EarningsPerShareDiluted=[q("2026-04-01", "2026-06-30", 4.25)],
    ))


BASE = date(2026, 3, 31)


def res(check, comparator, threshold, horizon=1, fin=None):
    return resolve(check, comparator, threshold, fin or nand_like(),
                   BASE, horizon, "TEST", THESIS_DAY)


class TestNextQuarterSelection:
    def test_horizon_one_picks_the_following_quarter(self):
        val, note = measure("revenue_usd", nand_like(), BASE, 1, "TEST", THESIS_DAY)
        assert val == 2_200
        assert "2026-06-30" in note

    def test_horizon_beyond_reported_data_is_pending(self):
        r = res("revenue_usd", "above", 0, horizon=2)
        assert r.status == "pending"
        assert "not reported yet" in r.note

    def test_pending_claims_do_not_count_as_wrong(self):
        assert res("revenue_usd", "above", 0, horizon=2).status != "miss"


class TestArithmetic:
    def test_yoy_uses_the_prior_year_quarter(self):
        # 2200 vs 1100 == +100%
        val, _ = measure("revenue_yoy_pct", nand_like(), BASE, 1, "TEST", THESIS_DAY)
        assert val == pytest.approx(100.0)

    @pytest.mark.parametrize("check,expected", [
        ("gross_margin_pct", 60.0),
        ("operating_margin_pct", 40.0),
        ("eps_diluted_usd", 4.25),
    ])
    def test_margins_and_eps(self, check, expected):
        val, _ = measure(check, nand_like(), BASE, 1, "TEST", THESIS_DAY)
        assert val == pytest.approx(expected)

    def test_missing_numerator_is_unresolvable_not_zero(self):
        r = res("net_margin_pct", "above", 10)
        assert r.status == "unresolvable"
        assert r.actual is None


class TestComparators:
    @pytest.mark.parametrize("comparator,threshold,expected", [
        ("above", 80.0, "hit"),    # actual +100% YoY
        ("above", 120.0, "miss"),
        ("below", 120.0, "hit"),
        ("below", 80.0, "miss"),
    ])
    def test_both_directions(self, comparator, threshold, expected):
        r = res("revenue_yoy_pct", comparator, threshold)
        assert r.status == expected
        assert r.actual == pytest.approx(100.0)


class TestQualitative:
    def test_qualitative_never_auto_settles(self):
        r = res("qualitative", "above", 0)
        assert r.status == "unresolvable"
        assert "by hand" in r.note

    def test_unresolvable_is_not_retried_forever(self):
        # unresolvable != pending, so it leaves the queue and asks for a human
        assert res("qualitative", "above", 0).status != "pending"


class TestScorecard:
    def test_hit_rate_is_none_before_anything_settles(self):
        c = Scorecard(pending=4)
        assert c.hit_rate is None, "0% would read as 'wrong', not 'unknown'"
        assert "nothing settled" in c.summary()

    def test_hit_rate_ignores_pending_and_unresolvable(self):
        c = Scorecard(hit=3, miss=1, pending=9, unresolvable=2)
        assert c.settled == 4
        assert c.hit_rate == pytest.approx(75.0)

    def test_summary_reports_the_unsettled_too(self):
        s = Scorecard(hit=3, miss=1, pending=9, unresolvable=2).summary()
        assert "3/4" in s and "9 pending" in s and "2 unresolvable" in s


class TestPriceCheckIsOptional:
    def test_price_failure_degrades_to_pending_not_crash(self, monkeypatch):
        import earnings_agent.prices as p

        monkeypatch.setattr(p, "return_since", lambda *a, **k: None)
        r = res("price_return_pct", "above", 10)
        assert r.status == "pending"
        assert "unavailable" in r.note
