"""Pipeline orchestration.

Three independent stages, each safe to run on its own schedule:

  poll    discover new Item 2.02 8-Ks for the watchlist
  process turn each new filing into a brief and emit it
  enrich  attach transcripts to briefs already emitted

The split is the whole point. `process` never waits on a transcript, so a brief
ships minutes after the filing lands; `enrich` catches up hours later when the
transcript exists. Run `enrich` on a slower cadence than the other two.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

from earnings_agent import llm, peers, prices, scoring, transcripts
from earnings_agent.config import Config
from earnings_agent.edgar import EdgarClient, Filing
from earnings_agent.llm import Brief, ExtractionError, Extractor, RefusalError
from earnings_agent.render import render
from earnings_agent.store import Store
from earnings_agent.xbrl import Financials

log = logging.getLogger("earnings_agent")


def _client(cfg: Config) -> EdgarClient:
    return EdgarClient(cfg.user_agent, cfg.requests_per_second)


def poll(cfg: Config, store: Store) -> list[Filing]:
    """Discover new earnings 8-Ks. Returns only filings not seen before."""
    found: list[Filing] = []
    with _client(cfg) as edgar:
        for ticker in cfg.tickers:
            try:
                filings = edgar.recent_filings(ticker, forms=("8-K",), limit=20)
            except Exception as e:
                log.error("poll %s failed: %s: %s", ticker, type(e).__name__, e)
                continue
            for f in filings:
                if not f.is_earnings:
                    continue
                if store.record_filing({
                    "accession": f.accession, "cik": f.cik, "ticker": f.ticker,
                    "form": f.form, "items": f.items, "filing_date": f.filing_date,
                    "report_date": f.report_date, "acceptance_dt": f.acceptance_dt,
                    "primary_doc": f.primary_doc,
                }):
                    found.append(f)
                    log.info("new earnings filing: %s %s (%s)",
                             f.ticker, f.accession, f.filing_date)
    return found


def process(cfg: Config, store: Store) -> list[Path]:
    """Build and emit a brief for every filing not yet processed."""
    pending = store.pending()
    if not pending:
        return []

    extractor: Extractor | None = None
    if cfg.llm_enabled:
        extractor = llm.build(cfg.provider, cfg.api_key, cfg.model, cfg.max_tokens, cfg.base_url)
        log.info("extraction via %s (%s)", cfg.model, cfg.provider)
    else:
        log.warning("%s unset - emitting figures without narrative extraction", cfg.api_key_env)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    with _client(cfg) as edgar:
        for row in pending:
            try:
                path = _process_one(cfg, store, edgar, extractor, row)
                if path:
                    written.append(path)
            except Exception as e:
                log.error("process %s failed: %s: %s", row["accession"], type(e).__name__, e)
                store.set_status(row["accession"], "error")
    return written


def _process_one(
    cfg: Config, store: Store, edgar: EdgarClient,
    extractor: Extractor | None, row: sqlite3.Row,
) -> Path | None:
    filing = Filing(
        accession=row["accession"], cik=row["cik"], ticker=row["ticker"],
        form=row["form"], items=row["items"] or "", filing_date=row["filing_date"] or "",
        report_date=row["report_date"] or "", acceptance_dt=row["acceptance_dt"] or "",
        primary_doc=row["primary_doc"] or "",
    )

    pr = edgar.press_release(filing)
    url, text = pr if pr else (None, None)
    if text is None:
        log.warning("%s %s has no EX-99 exhibit - figures only",
                    filing.ticker, filing.accession)

    fin = Financials(edgar.company_facts(filing.cik))

    brief: Brief | None = None
    model: str | None = None
    if extractor and text:
        try:
            brief, model = extractor.extract(filing.ticker, filing.filing_date, text)
        except RefusalError as e:
            log.error("extraction refused for %s: %s", filing.accession, e)
        except ExtractionError as e:
            log.error("extraction invalid for %s: %s", filing.accession, e)
        except Exception as e:
            log.error("extraction failed for %s: %s: %s",
                      filing.accession, type(e).__name__, e)

    # A transcript would add Q&A and, for issuers that do not publish written
    # guidance, the guidance itself. Queue it and move on - the brief ships now.
    store.queue_enrichment(
        filing.accession, "transcript",
        detail=f"provider={transcripts.provider().name}",
    )

    markdown = render(filing, fin, url, brief, model, transcript_state="pending")
    store.save_brief(
        filing.accession, filing.ticker, markdown, model,
        brief.model_dump() if brief else None,
    )
    store.set_status(filing.accession, "processed")

    path = cfg.output_dir / f"{filing.ticker}-{filing.filing_date}-{filing.accession}.md"
    path.write_text(markdown)
    log.info("wrote %s", path)
    return path


def enrich(cfg: Config, store: Store) -> int:
    """Attempt to attach transcripts to briefs already emitted.

    A provider returning None means 'not ready yet', so the enrichment stays
    pending and is retried on the next run.
    """
    provider = transcripts.provider()
    pending = store.pending_enrichments("transcript")
    if not pending:
        return 0
    if isinstance(provider, transcripts.NullProvider):
        log.info("%d transcript enrichment(s) pending; no provider configured", len(pending))
        return 0

    attached = 0
    for row in pending:
        filing = store.conn.execute(
            "SELECT * FROM filings WHERE accession = ?", (row["accession"],)
        ).fetchone()
        if filing is None:
            continue
        try:
            t = provider.fetch(filing["ticker"], filing["filing_date"])
        except Exception as e:
            log.error("transcript fetch failed for %s: %s: %s",
                      row["accession"], type(e).__name__, e)
            continue
        if t is None:
            continue
        store.resolve_enrichment(
            row["accession"], "transcript", "done",
            body=json.dumps({"source": t.source, "period": t.period, "text": t.text}),
            detail=f"provider={provider.name}",
        )
        attached += 1
        log.info("attached transcript for %s (%s)", filing["ticker"], row["accession"])
    return attached


# -- thesis layer ----------------------------------------------------------

def _latest_brief_markdown(store: Store, ticker: str) -> sqlite3.Row | None:
    return store.conn.execute(
        """SELECT b.*, f.cik FROM briefs b JOIN filings f ON f.accession = b.accession
           WHERE b.ticker = ? ORDER BY f.filing_date DESC LIMIT 1""",
        (ticker.upper(),),
    ).fetchone()


def generate_thesis(cfg: Config, store: Store, ticker: str,
                    with_prices: bool = True) -> tuple[int, Any]:
    """Build and persist a thesis for one ticker. Returns (thesis_id, Thesis)."""
    from earnings_agent.thesis import SYSTEM, Thesis, build_prompt

    ticker = ticker.upper()
    subject = _latest_brief_markdown(store, ticker)
    if subject is None:
        raise ValueError(f"no brief stored for {ticker} - run `process` first")

    peer_list, groups = peers.resolve(ticker, cfg.peer_groups)
    peer_briefs: dict[str, str] = {}
    for p in peer_list:
        row = _latest_brief_markdown(store, p)
        if row is not None:
            peer_briefs[p] = row["markdown"]
        else:
            log.info("no stored brief for peer %s - skipping", p)
    if peer_list and not peer_briefs:
        log.warning("%s has peers %s but none are briefed yet - thesis will see one "
                    "company in isolation", ticker, ", ".join(peer_list))

    price_context = None
    if with_prices:
        bits = []
        for t in [ticker, *peer_briefs.keys()]:
            q = prices.quote(t)
            if q:
                rng = ""
                if q.fifty_two_week_low and q.fifty_two_week_high:
                    rng = f", 52wk {q.fifty_two_week_low:,.2f}-{q.fifty_two_week_high:,.2f}"
                bits.append(f"{t}: {q.price:,.2f} {q.currency}{rng} (as of {q.as_of})")
        price_context = "\n".join(bits) or None

    with _client(cfg) as edgar:
        fin = Financials(edgar.company_facts(subject["cik"]))
    base_end = fin.period_end("quarter")
    if base_end is None:
        raise ValueError(f"no quarterly XBRL data for {ticker}")

    client = llm.build(cfg.provider, cfg.api_key, cfg.thesis_model, cfg.max_tokens, cfg.base_url)
    prompt = build_prompt(ticker, subject["markdown"], peer_briefs, price_context)
    result, model = client.complete(SYSTEM, prompt, Thesis)

    tid = store.save_thesis(
        ticker, subject["accession"], base_end.isoformat(), model,
        result, list(peer_briefs.keys()),
    )
    log.info("thesis #%d for %s (%s/%s, %d claims, peers: %s)",
             tid, ticker, result.direction, result.confidence, len(result.claims),
             ", ".join(peer_briefs) or "none")
    return tid, result


def score(cfg: Config, store: Store) -> dict[str, int]:
    """Settle every pending claim it is now possible to settle."""
    pending = store.unsettled_claims()
    if not pending:
        return {"settled": 0, "still_pending": 0}

    settled = still = 0
    facts_cache: dict[str, Financials] = {}
    with _client(cfg) as edgar:
        for row in pending:
            t = row["ticker"]
            if t not in facts_cache:
                try:
                    facts_cache[t] = Financials(edgar.company_facts(edgar.ticker_to_cik(t)))
                except Exception as e:
                    log.error("cannot load facts for %s: %s", t, e)
                    continue
            res = scoring.resolve(
                row["check_kind"], row["comparator"], row["threshold"],
                facts_cache[t], date.fromisoformat(row["base_end"]),
                row["horizon_quarters"], t,
                date.fromisoformat(row["thesis_created_at"][:10]),
            )
            if res.status == "pending":
                still += 1
                continue
            store.settle_claim(row["id"], res.status, res.actual, res.note)
            settled += 1
            log.info("claim %d (%s): %s — %s", row["id"], t, res.status, res.note)
    return {"settled": settled, "still_pending": still}


def run(cfg: Config, store: Store) -> dict[str, int]:
    new = poll(cfg, store)
    written = process(cfg, store)
    attached = enrich(cfg, store)
    # Scoring is cheap and has no model in the loop, so it always runs - new
    # filings are exactly what settles outstanding claims.
    scored = score(cfg, store)
    return {
        "discovered": len(new), "briefs": len(written),
        "transcripts": attached, "claims_settled": scored["settled"],
    }
