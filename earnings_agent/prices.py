"""Price history.

Optional by design. Scoring must not depend on this module: the primary way a
claim gets resolved is against the next quarter's XBRL actuals, which are
exact, free, and officially published. Prices are a secondary check.

The source is Yahoo's chart endpoint - no key, but unofficial and unsupported,
so it can break without notice. Stooq, the usual keyless alternative, now
serves a JavaScript proof-of-work challenge instead of CSV. Every function here
returns None rather than raising when the source misbehaves, so a broken price
feed degrades the scoreboard instead of taking the pipeline down.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone

import httpx

log = logging.getLogger("earnings_agent")

CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
# Yahoo rejects requests without a browser-ish UA.
UA = "Mozilla/5.0 (X11; Linux x86_64) earnings-agent/0.1"


@dataclass
class Quote:
    ticker: str
    price: float
    currency: str
    fifty_two_week_high: float | None
    fifty_two_week_low: float | None
    as_of: date


def _get(ticker: str, params: dict) -> dict | None:
    try:
        r = httpx.get(
            CHART.format(ticker=ticker.upper()),
            params=params,
            headers={"User-Agent": UA},
            timeout=20.0,
            follow_redirects=True,
        )
        r.raise_for_status()
        result = r.json().get("chart", {}).get("result")
        return result[0] if result else None
    except Exception as e:
        log.info("price lookup failed for %s: %s: %s", ticker, type(e).__name__, e)
        return None


def quote(ticker: str) -> Quote | None:
    node = _get(ticker, {"range": "1d", "interval": "1d"})
    if not node:
        return None
    m = node.get("meta", {})
    if m.get("regularMarketPrice") is None:
        return None
    ts = m.get("regularMarketTime")
    return Quote(
        ticker=ticker.upper(),
        price=float(m["regularMarketPrice"]),
        currency=m.get("currency", "USD"),
        fifty_two_week_high=m.get("fiftyTwoWeekHigh"),
        fifty_two_week_low=m.get("fiftyTwoWeekLow"),
        as_of=datetime.fromtimestamp(ts, tz=timezone.utc).date() if ts else date.today(),
    )


def closes(ticker: str, range_: str = "1y") -> list[tuple[date, float]] | None:
    """Daily closes, oldest first. `range_` is a Yahoo range string (1mo/6mo/1y/5y)."""
    node = _get(ticker, {"range": range_, "interval": "1d"})
    if not node:
        return None
    stamps = node.get("timestamp") or []
    quotes = (node.get("indicators", {}).get("quote") or [{}])[0]
    series = quotes.get("close") or []
    out = [
        (datetime.fromtimestamp(t, tz=timezone.utc).date(), float(c))
        for t, c in zip(stamps, series)
        if c is not None
    ]
    return out or None


def return_since(ticker: str, since: date, range_: str = "2y") -> float | None:
    """Percent return from the first close on or after `since` to the latest."""
    hist = closes(ticker, range_)
    if not hist:
        return None
    after = [(d, c) for d, c in hist if d >= since]
    if len(after) < 2:
        return None
    start, end = after[0][1], after[-1][1]
    return None if start == 0 else (end - start) / start * 100.0
