"""XBRL extraction tests.

These run offline against synthetic companyfacts blobs shaped like the real
API response. Both cases are regressions from bugs found against live SEC data.
"""

from __future__ import annotations

from datetime import date

import pytest

from earnings_agent.xbrl import INCOME_CONCEPTS, Financials


def facts(**tags: list[dict]) -> dict:
    return {
        "entityName": "Test Co",
        "cik": 1,
        "facts": {"us-gaap": {t: {"units": {"USD": e}} for t, e in tags.items()}},
    }


def q(start: str, end: str, val: float, accn: str = "a-1", filed: str = "2026-01-01") -> dict:
    return {"start": start, "end": end, "val": val, "accn": accn,
            "form": "10-Q", "filed": filed, "fy": 2026, "fp": "Q1"}


def instant(end: str, val: float, accn: str = "a-1", filed: str = "2026-01-01") -> dict:
    return {"end": end, "val": val, "accn": accn,
            "form": "10-Q", "filed": filed, "fy": 2026, "fp": "Q1"}


class TestTagChainMerge:
    """A filer that switches XBRL tags mid-life must not strand the new data.

    NVIDIA reported revenue as RevenueFromContractWithCustomerExcludingAssessed-
    Tax until 2020 and as Revenues afterward. Resolving the fallback chain by
    "first tag with any data" returned the 2020 figure and reported the current
    quarter as missing.
    """

    def test_picks_current_tag_when_earlier_chain_entry_is_stale(self):
        fin = Financials(facts(
            RevenueFromContractWithCustomerExcludingAssessedTax=[
                q("2019-10-28", "2020-01-26", 3_105_000_000),
            ],
            Revenues=[
                q("2019-10-28", "2020-01-26", 3_105_000_000),
                q("2026-01-26", "2026-04-26", 81_615_000_000),
            ],
        ))
        rows, missing = fin.income_statement("quarter")
        rev = next(f for f in rows if f.concept == "Revenue")
        assert rev.value == 81_615_000_000
        assert rev.tag == "Revenues"
        assert "Revenue" not in missing

    def test_earlier_chain_entry_wins_when_both_cover_the_period(self):
        fin = Financials(facts(
            RevenueFromContractWithCustomerExcludingAssessedTax=[
                q("2026-01-26", "2026-04-26", 100),
            ],
            Revenues=[q("2026-01-26", "2026-04-26", 999)],
        ))
        rev = fin.series("Revenue", INCOME_CONCEPTS["Revenue"])[-1]
        assert rev.value == 100, "the more specific ASC 606 tag should win a tie"


class TestPeriodCoherence:
    """Every line in one statement must cover the same period.

    Apple reports cash flow year-to-date in 10-Qs, so selecting each line
    independently mixed a Q1 cash-flow figure into a Q3 income statement - a
    table that looks right and is not.
    """

    def test_lines_share_one_period_end(self):
        fin = Financials(facts(
            Revenues=[q("2026-03-29", "2026-06-27", 109_417)],
            OperatingIncomeLoss=[q("2026-03-29", "2026-06-27", 35_695)],
            # only ever reported quarterly in Q1; YTD thereafter
            NetCashProvidedByUsedInOperatingActivities=[
                q("2025-09-28", "2025-12-27", 53_925),
                q("2025-09-28", "2026-06-27", 116_996),
            ],
        ))
        rows, missing = fin.income_statement("quarter")
        assert {f.end for f in rows} == {date(2026, 6, 27)}
        assert "Operating cash flow" in missing, "stale quarter must be omitted, not back-filled"

    def test_ytd_surfaces_the_cumulative_figure(self):
        fin = Financials(facts(
            Revenues=[q("2026-03-29", "2026-06-27", 109_417),
                      q("2025-09-28", "2026-06-27", 364_357)],
            NetCashProvidedByUsedInOperatingActivities=[
                q("2025-09-28", "2026-06-27", 116_996),
            ],
        ))
        ocf = next(f for f in fin.ytd() if f.concept == "Operating cash flow")
        assert ocf.value == 116_996
        assert ocf.period_kind == "ytd-9mo"


class TestRestatements:
    def test_latest_filed_wins_for_a_repeated_period(self):
        fin = Financials(facts(Revenues=[
            q("2026-01-01", "2026-03-31", 100, accn="orig", filed="2026-04-01"),
            q("2026-01-01", "2026-03-31", 110, accn="restated", filed="2026-08-01"),
        ]))
        s = fin.series("Revenue", INCOME_CONCEPTS["Revenue"])
        assert len(s) == 1, "one fact per period"
        assert s[0].value == 110 and s[0].accession == "restated"


class TestYoY:
    def test_matches_the_prior_year_period(self):
        fin = Financials(facts(Revenues=[
            q("2025-03-30", "2025-06-28", 94_036),
            q("2026-03-29", "2026-06-27", 109_417),
        ]))
        pair = fin.yoy("Revenue", INCOME_CONCEPTS["Revenue"])
        assert pair is not None
        prior, latest = pair
        assert (prior.value, latest.value) == (94_036, 109_417)

    def test_returns_none_without_a_comparable_period(self):
        fin = Financials(facts(Revenues=[q("2026-03-29", "2026-06-27", 109_417)]))
        assert fin.yoy("Revenue", INCOME_CONCEPTS["Revenue"]) is None


class TestBalanceSheet:
    def test_uses_instant_facts_only(self):
        fin = Financials(facts(
            Assets=[instant("2026-06-27", 383_266)],
            Revenues=[q("2026-03-29", "2026-06-27", 109_417)],
        ))
        bs = fin.balance_sheet()
        assert len(bs) == 1
        assert bs[0].concept == "Total assets"
        assert bs[0].start is None and bs[0].period_kind == "instant"


def test_empty_facts_do_not_raise():
    fin = Financials(facts())
    assert fin.income_statement("quarter") == ([], [])
    assert fin.balance_sheet() == []
    assert fin.period_end("quarter") is None
