"""Claim resolution.

Deliberately dumb arithmetic. A claim was recorded as (quantity, comparator,
threshold, horizon); resolving it means measuring that quantity against XBRL
actuals and comparing. No model is involved in deciding whether a claim was
right — the same reason no model touches the reported figures. A scoreboard
graded by the thing being graded is worthless.

Claims settle against reported fundamentals rather than share price. A price
move over one quarter is mostly noise; "did revenue growth actually decelerate
below 40%" is a real question with an exact answer, published quarterly, free.
Price checks exist but lean on an unofficial feed, so they are best-effort.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from earnings_agent import prices
from earnings_agent.xbrl import INCOME_CONCEPTS, Financials

Status = Literal["hit", "miss", "pending", "unresolvable"]


@dataclass
class Resolution:
    status: Status
    actual: float | None
    note: str


def _quarter_after(fin: Financials, base_end: date, n: int) -> date | None:
    """The period end `n` reported quarters after `base_end`."""
    ends = sorted({
        f.end
        for f in fin.series("Revenue", INCOME_CONCEPTS["Revenue"])
        if f.period_kind == "quarter" and f.end > base_end
    })
    return ends[n - 1] if len(ends) >= n else None


def _at(fin: Financials, concept: str, end: date) -> float | None:
    for f in fin.series(concept, INCOME_CONCEPTS[concept]):
        if f.end == end and f.period_kind == "quarter":
            return f.value
    return None


def _margin(fin: Financials, numerator: str, end: date) -> float | None:
    num, rev = _at(fin, numerator, end), _at(fin, "Revenue", end)
    if num is None or not rev:
        return None
    return num / rev * 100.0


def measure(
    check: str, fin: Financials, base_end: date, horizon_quarters: int,
    ticker: str, thesis_date: date,
) -> tuple[float | None, str]:
    """Measure the quantity a claim is about. Returns (value, note)."""
    if check == "qualitative":
        return None, "qualitative claim - no numeric check, resolve by hand"

    if check == "price_return_pct":
        r = prices.return_since(ticker, thesis_date)
        if r is None:
            return None, "price feed unavailable (unofficial source)"
        return r, f"return since {thesis_date.isoformat()}"

    target = _quarter_after(fin, base_end, horizon_quarters)
    if target is None:
        return None, (
            f"quarter {horizon_quarters} after {base_end.isoformat()} not reported yet"
        )

    label = f"quarter ending {target.isoformat()}"

    if check == "revenue_usd":
        return _at(fin, "Revenue", target), label
    if check == "eps_diluted_usd":
        for f in fin.series("Diluted EPS", ["EarningsPerShareDiluted"]):
            if f.end == target and f.period_kind == "quarter":
                return f.value, label
        return None, f"diluted EPS not reported for {label}"
    if check == "gross_margin_pct":
        return _margin(fin, "Gross profit", target), label
    if check == "operating_margin_pct":
        return _margin(fin, "Operating income", target), label
    if check == "net_margin_pct":
        return _margin(fin, "Net income", target), label
    if check == "revenue_yoy_pct":
        now = _at(fin, "Revenue", target)
        prior_end = target.replace(year=target.year - 1)
        candidates = [
            f for f in fin.series("Revenue", INCOME_CONCEPTS["Revenue"])
            if f.period_kind == "quarter" and abs((f.end - prior_end).days) <= 20
        ]
        if now is None or not candidates:
            return None, f"no comparable prior-year quarter for {label}"
        prior = candidates[0].value
        if not prior:
            return None, "prior-year revenue is zero"
        return (now - prior) / abs(prior) * 100.0, label

    return None, f"unknown check {check!r}"


def resolve(
    check: str, comparator: str, threshold: float,
    fin: Financials, base_end: date, horizon_quarters: int,
    ticker: str, thesis_date: date,
) -> Resolution:
    actual, note = measure(check, fin, base_end, horizon_quarters, ticker, thesis_date)

    if actual is None:
        # "not reported yet" is pending and will resolve on its own; anything
        # else needs a human and should not sit in the queue forever.
        pending = "not reported yet" in note or "unavailable" in note
        return Resolution("pending" if pending else "unresolvable", None, note)

    hit = actual > threshold if comparator == "above" else actual < threshold
    return Resolution("hit" if hit else "miss", actual, note)


@dataclass
class Scorecard:
    hit: int = 0
    miss: int = 0
    pending: int = 0
    unresolvable: int = 0

    @property
    def settled(self) -> int:
        return self.hit + self.miss

    @property
    def hit_rate(self) -> float | None:
        """None until anything has actually settled - not 0%, which reads as 'bad'."""
        return self.hit / self.settled * 100.0 if self.settled else None

    def add(self, status: str) -> None:
        setattr(self, status, getattr(self, status) + 1)

    def summary(self) -> str:
        if not self.settled:
            return f"nothing settled yet ({self.pending} pending)"
        return (
            f"{self.hit}/{self.settled} correct ({self.hit_rate:.0f}%), "
            f"{self.pending} pending, {self.unresolvable} unresolvable"
        )
