"""XBRL financial statement extraction from SEC companyfacts.

Deliberately no LLM anywhere in this module. Every number here is lifted
verbatim from a filed XBRL fact and carries the accession number it came from,
so any figure in a brief can be traced back to the filing that reported it.
Language models are for reading prose, not for reading balance sheets.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable

# Filers do not all use the same tag for the same concept, so each concept maps
# to a priority-ordered fallback chain. First tag that yields data wins.
INCOME_CONCEPTS: dict[str, list[str]] = {
    "Revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "Cost of revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfServices"],
    "Gross profit": ["GrossProfit"],
    "R&D expense": ["ResearchAndDevelopmentExpense"],
    "Operating income": ["OperatingIncomeLoss"],
    "Net income": ["NetIncomeLoss", "ProfitLoss"],
}

BALANCE_CONCEPTS: dict[str, list[str]] = {
    "Total assets": ["Assets"],
    "Current assets": ["AssetsCurrent"],
    "Cash & equivalents": ["CashAndCashEquivalentsAtCarryingValue"],
    "Short-term investments": ["ShortTermInvestments", "MarketableSecuritiesCurrent"],
    "Inventory": ["InventoryNet"],
    "Total liabilities": ["Liabilities"],
    "Current liabilities": ["LiabilitiesCurrent"],
    "Long-term debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "Shareholders' equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
}

CASHFLOW_CONCEPTS: dict[str, list[str]] = {
    "Operating cash flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "Capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
}

PER_SHARE_CONCEPTS: dict[str, list[str]] = {
    "Diluted EPS": ["EarningsPerShareDiluted"],
    "Basic EPS": ["EarningsPerShareBasic"],
}


@dataclass(frozen=True)
class Fact:
    """A single reported number, traceable to the filing that reported it."""

    concept: str
    tag: str
    value: float
    unit: str
    end: date
    start: date | None
    form: str
    accession: str
    filed: date
    fy: int | None
    fp: str | None

    @property
    def days(self) -> int | None:
        return (self.end - self.start).days if self.start else None

    @property
    def period_kind(self) -> str:
        d = self.days
        if d is None:
            return "instant"
        if 80 <= d <= 100:
            return "quarter"
        if 170 <= d <= 195:
            return "half"
        if 260 <= d <= 285:
            return "ytd-9mo"
        if 350 <= d <= 380:
            return "annual"
        return f"{d}d"

    @property
    def period_label(self) -> str:
        if self.start is None:
            return f"as of {self.end.isoformat()}"
        return f"{self.start.isoformat()} to {self.end.isoformat()} ({self.period_kind})"

    def format(self) -> str:
        if self.unit == "USD":
            return f"${self.value:,.0f}"
        if self.unit in ("USD/shares", "USD-per-shares"):
            return f"${self.value:,.2f}"
        return f"{self.value:,.0f} {self.unit}"


def _parse(d: str | None) -> date | None:
    return date.fromisoformat(d) if d else None


class Financials:
    """Queryable view over one company's XBRL facts."""

    def __init__(self, raw: dict[str, Any]):
        self.raw = raw
        self.entity = raw.get("entityName", "")
        self.cik = str(raw.get("cik", "")).zfill(10)
        self._us_gaap: dict[str, Any] = raw.get("facts", {}).get("us-gaap", {})

    # -- core lookup ------------------------------------------------------

    def _facts_for_tag(self, concept: str, tag: str) -> list[Fact]:
        node = self._us_gaap.get(tag)
        if not node:
            return []
        out: list[Fact] = []
        for unit, entries in node.get("units", {}).items():
            for e in entries:
                end = _parse(e.get("end"))
                filed = _parse(e.get("filed"))
                if end is None or filed is None:
                    continue
                out.append(Fact(
                    concept=concept, tag=tag, value=e["val"], unit=unit,
                    end=end, start=_parse(e.get("start")),
                    form=e.get("form", ""), accession=e.get("accn", ""),
                    filed=filed, fy=e.get("fy"), fp=e.get("fp"),
                ))
        return out

    @staticmethod
    def _dedupe(facts: Iterable[Fact]) -> list[Fact]:
        """Collapse restatements: one fact per period, keeping the latest filed.

        The same period gets re-reported across subsequent filings and
        amendments. Without this you get duplicate - and sometimes conflicting
        - rows for a single quarter.
        """
        best: dict[tuple[date | None, date, str], Fact] = {}
        for f in facts:
            key = (f.start, f.end, f.unit)
            cur = best.get(key)
            if cur is None or f.filed > cur.filed:
                best[key] = f
        return sorted(best.values(), key=lambda f: (f.end, f.start or date.min))

    def series(self, concept: str, chain: list[str]) -> list[Fact]:
        """All deduped facts for a concept, oldest first, merged across the tag chain.

        The merge is per *period*, not per tag. Filers change tags over time -
        NVIDIA reported revenue as RevenueFromContractWithCustomerExcluding-
        AssessedTax until 2020 and as Revenues after - so resolving the chain
        by "first tag with any data" silently returns a figure years out of
        date. Walking every tag and keying by period avoids that; where two
        tags both cover a period, the earlier (more specific) chain entry wins.
        """
        best: dict[tuple[date | None, date, str], tuple[int, Fact]] = {}
        for rank, tag in enumerate(chain):
            for f in self._dedupe(self._facts_for_tag(concept, tag)):
                key = (f.start, f.end, f.unit)
                cur = best.get(key)
                if cur is None or rank < cur[0]:
                    best[key] = (rank, f)
        return sorted(
            (f for _, f in best.values()),
            key=lambda f: (f.end, f.start or date.min),
        )

    # -- statement views --------------------------------------------------

    def balance_sheet(self, as_of: date | None = None) -> list[Fact]:
        """Latest balance sheet (instant facts) at or before `as_of`."""
        rows: list[Fact] = []
        for concept, chain in BALANCE_CONCEPTS.items():
            facts = [f for f in self.series(concept, chain) if f.start is None]
            if as_of:
                facts = [f for f in facts if f.end <= as_of]
            if facts:
                rows.append(facts[-1])
        return rows

    def period_end(self, kind: str = "quarter", as_of: date | None = None) -> date | None:
        """The reporting period end that the latest income statement covers."""
        ends: list[date] = []
        for concept, chain in INCOME_CONCEPTS.items():
            facts = [f for f in self.series(concept, chain) if f.period_kind == kind]
            if as_of:
                facts = [f for f in facts if f.end <= as_of]
            if facts:
                ends.append(facts[-1].end)
        return max(ends) if ends else None

    def income_statement(
        self, kind: str = "quarter", as_of: date | None = None
    ) -> tuple[list[Fact], list[str]]:
        """Income statement for one coherent period.

        Every returned fact shares the same period end. This matters: filers
        report some lines quarterly and others year-to-date within the same
        10-Q - Apple's cash flow statement is YTD-only - so selecting each line
        independently silently mixes periods and produces a table that looks
        right and is not. Concepts with no fact for the target period are
        reported as omissions rather than back-filled from another quarter.
        """
        target = self.period_end(kind, as_of)
        if target is None:
            return [], []

        rows: list[Fact] = []
        missing: list[str] = []
        groups = {**INCOME_CONCEPTS, **CASHFLOW_CONCEPTS, **PER_SHARE_CONCEPTS}
        for concept, chain in groups.items():
            match = [
                f for f in self.series(concept, chain)
                if f.end == target and f.period_kind == kind
            ]
            if match:
                rows.append(match[-1])
            else:
                missing.append(concept)
        return rows, missing

    def ytd(self, as_of: date | None = None) -> list[Fact]:
        """Year-to-date facts, for concepts filers only report cumulatively."""
        target = self.period_end("quarter", as_of)
        if target is None:
            return []
        rows: list[Fact] = []
        for concept, chain in {**INCOME_CONCEPTS, **CASHFLOW_CONCEPTS}.items():
            match = [
                f for f in self.series(concept, chain)
                if f.end == target and f.period_kind in ("half", "ytd-9mo", "annual")
            ]
            if match:
                rows.append(match[-1])
        return rows

    def yoy(self, concept: str, chain: list[str], kind: str = "quarter") -> tuple[Fact, Fact] | None:
        """Latest period paired with the same period one year earlier."""
        facts = [f for f in self.series(concept, chain) if f.period_kind == kind]
        if len(facts) < 2:
            return None
        latest = facts[-1]
        target = latest.end.replace(year=latest.end.year - 1)
        prior = min(
            (f for f in facts if abs((f.end - target).days) <= 20),
            key=lambda f: abs((f.end - target).days),
            default=None,
        )
        return (prior, latest) if prior else None

    def growth_table(self, kind: str = "quarter") -> list[tuple[str, Fact, Fact, float | None]]:
        """Year-over-year comparison for the headline income statement lines."""
        out = []
        for concept, chain in INCOME_CONCEPTS.items():
            pair = self.yoy(concept, chain, kind)
            if not pair:
                continue
            prior, latest = pair
            pct = ((latest.value - prior.value) / abs(prior.value) * 100) if prior.value else None
            out.append((concept, prior, latest, pct))
        return out
