"""Command line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from earnings_agent import llm, pipeline, scoring
from earnings_agent.config import ConfigError, load
from earnings_agent.edgar import EdgarClient, Filing
from earnings_agent.render import render
from earnings_agent.store import Store
from earnings_agent.xbrl import Financials


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
        stream=sys.stderr,
    )


def cmd_run(args, cfg, store) -> int:
    stats = pipeline.run(cfg, store)
    print(f"discovered {stats['discovered']} · briefs {stats['briefs']} "
          f"· transcripts {stats['transcripts']}")
    return 0


def cmd_poll(args, cfg, store) -> int:
    found = pipeline.poll(cfg, store)
    for f in found:
        print(f"{f.ticker:6s} {f.accession}  {f.filing_date}  items={f.items}")
    print(f"{len(found)} new earnings filing(s)")
    return 0


def cmd_process(args, cfg, store) -> int:
    for p in pipeline.process(cfg, store):
        print(p)
    return 0


def cmd_brief(args, cfg, store) -> int:
    """Build a brief for one ticker's latest earnings 8-K, ignoring dedupe state."""
    ticker = args.ticker.upper()
    with EdgarClient(cfg.user_agent, cfg.requests_per_second) as edgar:
        try:
            earnings = [f for f in edgar.recent_filings(ticker, limit=25) if f.is_earnings]
        except KeyError as e:
            print(f"error: {e.args[0]}", file=sys.stderr)
            return 1
        if not earnings:
            print(f"no Item 2.02 8-K found for {ticker}", file=sys.stderr)
            return 1
        filing: Filing = earnings[0]

        pr = edgar.press_release(filing)
        url, text = pr if pr else (None, None)
        fin = Financials(edgar.company_facts(filing.cik))

    brief = model = None
    if cfg.llm_enabled and text and not args.no_llm:
        extractor = llm.build(
            cfg.provider, cfg.api_key, cfg.model, cfg.max_tokens, cfg.base_url
        )
        brief, model = extractor.extract(filing.ticker, filing.filing_date, text)
    elif not cfg.llm_enabled and not args.no_llm:
        print(f"note: {cfg.api_key_env} unset - figures only", file=sys.stderr)

    print(render(filing, fin, url, brief, model))
    return 0


def cmd_models(args, cfg, store) -> int:
    """List models this key can reach.

    Provider model IDs churn, so discover them rather than trusting the default
    in config.toml.
    """
    if not cfg.llm_enabled:
        print(f"{cfg.api_key_env} is unset", file=sys.stderr)
        return 1
    if cfg.provider != "openai-compat":
        print(f"model listing is only wired for openai-compat (provider is {cfg.provider})",
              file=sys.stderr)
        return 1

    from openai import OpenAI

    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=60.0)
    names = sorted(m.id for m in client.models.list())
    for n in names:
        print(f"  {n}{'   <- configured' if n.endswith(cfg.model) else ''}")
    print(f"\n{len(names)} model(s) reachable at {cfg.base_url}")
    return 0


def cmd_thesis(args, cfg, store) -> int:
    from earnings_agent.peers import describe
    from earnings_agent.render import render_thesis

    if not cfg.llm_enabled:
        print(f"{cfg.api_key_env} is unset — a thesis needs a model", file=sys.stderr)
        return 1
    print(describe(args.ticker.upper(), cfg.peer_groups), file=sys.stderr)
    try:
        tid, thesis = pipeline.generate_thesis(cfg, store, args.ticker,
                                               with_prices=not args.no_prices)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    row = store.get_thesis(tid)
    print(render_thesis(args.ticker.upper(), thesis, json.loads(row["peers"]),
                        row["model"], store.claims_for(tid)))
    print(f"\n_Saved as thesis #{tid}. Run `earnings-agent score` after the next "
          f"quarter to settle its claims._")
    return 0


def cmd_score(args, cfg, store) -> int:
    res = pipeline.score(cfg, store)
    print(f"settled {res['settled']}, still pending {res['still_pending']}\n")

    overall = scoring.Scorecard()
    for ticker in sorted({r["ticker"] for r in store.theses(limit=500)}):
        card = scoring.Scorecard()
        for status, n in store.claim_stats(ticker).items():
            if hasattr(card, status):
                for _ in range(n):
                    card.add(status)
                    overall.add(status)
        print(f"  {ticker:6s} {card.summary()}")
    print(f"\n  {'ALL':6s} {overall.summary()}")
    if overall.settled and overall.settled < 10:
        print(f"\n  Note: {overall.settled} settled claims is far too few to read "
              "anything into the hit rate.")
    return 0


def cmd_serve(args, cfg, store) -> int:
    import uvicorn

    from earnings_agent.web import create_app

    # Bound to loopback deliberately: there is no auth, so it must not be
    # reachable from anywhere but an SSH tunnel.
    print(f"serving on http://127.0.0.1:{args.port}", file=sys.stderr)
    print(f"tunnel from your laptop:  ssh -L {args.port}:localhost:{args.port} "
          f"$(hostname)", file=sys.stderr)
    uvicorn.run(create_app(cfg), host="127.0.0.1", port=args.port, log_level="warning")
    return 0


def cmd_status(args, cfg, store) -> int:
    print(f"watchlist: {', '.join(cfg.tickers)}")
    if cfg.llm_enabled:
        print(f"llm:       {cfg.model} via {cfg.provider}")
        if cfg.base_url:
            print(f"           {cfg.base_url}")
    else:
        print(f"llm:       disabled ({cfg.api_key_env} unset)")
    print(f"db:        {cfg.db_path}")
    print()
    rows = list(store.conn.execute(
        "SELECT status, COUNT(*) n FROM filings GROUP BY status"))
    print("filings:", ", ".join(f"{r['status']}={r['n']}" for r in rows) or "none")
    pend = store.pending_enrichments("transcript")
    print(f"transcript enrichments pending: {len(pend)}")
    print()
    for b in store.recent_briefs(10):
        print(f"  {b['created_at']}  {b['ticker']:6s} {b['accession']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="earnings-agent", description=__doc__)
    p.add_argument("-c", "--config", help="path to config.toml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("run", help="poll, process, and enrich in one pass")
    sub.add_parser("poll", help="discover new earnings filings only")
    sub.add_parser("process", help="build briefs for discovered filings")
    sub.add_parser("status", help="show watchlist and pipeline state")
    sub.add_parser("models", help="list models the configured key can reach")
    b = sub.add_parser("brief", help="build a brief for one ticker on demand")
    b.add_argument("ticker")
    b.add_argument("--no-llm", action="store_true", help="figures only, skip extraction")

    th = sub.add_parser("thesis", help="form a speculative, scoreable view on a ticker")
    th.add_argument("ticker")
    th.add_argument("--no-prices", action="store_true",
                    help="skip the (unofficial) price feed")
    sub.add_parser("score", help="settle outstanding claims and show the scoreboard")
    sv = sub.add_parser("serve", help="run the local web UI")
    sv.add_argument("--port", type=int, default=8000)

    args = p.parse_args(argv)
    _setup_logging(args.verbose)

    try:
        cfg = load(args.config)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    store = Store(cfg.db_path)
    try:
        return {
            "run": cmd_run, "poll": cmd_poll, "process": cmd_process,
            "brief": cmd_brief, "status": cmd_status, "models": cmd_models,
            "thesis": cmd_thesis, "score": cmd_score, "serve": cmd_serve,
        }[args.cmd](args, cfg, store)
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
