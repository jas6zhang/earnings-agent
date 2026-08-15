"""Local web UI.

Binds to localhost only. There is no authentication because there is nothing to
authenticate against — reach it over an SSH tunnel:

    ssh -L 8000:localhost:8000 <devserver>

Read-mostly by design: the pipeline runs from cron or the CLI, and this renders
what it produced. The one exception is thesis generation, which is on-demand by
nature.
"""

from __future__ import annotations

import json

import markdown as md
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse

from earnings_agent import scoring
from earnings_agent.config import Config
from earnings_agent.store import Store

CSS = """
:root{--fg:#1a1a1a;--dim:#6b7280;--line:#e5e7eb;--bg:#fdfdfc;--accent:#8b4513;
--hit:#15803d;--miss:#b91c1c;--pend:#a16207;--spec:#fff7ed;--specline:#fdba74}
@media(prefers-color-scheme:dark){:root{--fg:#e8e6e3;--dim:#9ca3af;--line:#2f2f33;
--bg:#161618;--accent:#d99058;--hit:#4ade80;--miss:#f87171;--pend:#fbbf24;
--spec:#2a1f14;--specline:#7c4a1e}}
*{box-sizing:border-box}
body{max-width:52rem;margin:0 auto;padding:2rem 1.25rem 5rem;background:var(--bg);
color:var(--fg);font:16px/1.65 Georgia,'Iowan Old Style',serif}
h1{font-size:1.75rem;margin:0 0 .25rem}h2{font-size:1.2rem;margin:2rem 0 .5rem;
padding-bottom:.3rem;border-bottom:1px solid var(--line)}h3{font-size:1rem;margin:1.4rem 0 .4rem}
a{color:var(--accent)}code,pre,table{font-family:ui-monospace,'SF Mono',Menlo,monospace}
code{font-size:.85em}
nav{margin-bottom:2rem;padding-bottom:.75rem;border-bottom:2px solid var(--line);
font-family:ui-monospace,monospace;font-size:.85rem}
nav a{margin-right:1.25rem;text-decoration:none}
table{border-collapse:collapse;width:100%;font-size:.85rem;margin:.75rem 0}
th,td{padding:.4rem .6rem;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--dim);font-weight:600;text-transform:uppercase;font-size:.7rem;letter-spacing:.04em}
td:nth-child(n+2){text-align:right}
th:nth-child(n+2){text-align:right}
.meta{color:var(--dim);font-size:.85rem;font-family:ui-monospace,monospace}
blockquote{margin:.3rem 0 .3rem 1rem;padding-left:.8rem;border-left:2px solid var(--line);
color:var(--dim);font-size:.9rem;font-style:italic}
.card{border:1px solid var(--line);border-radius:6px;padding:.75rem 1rem;margin:.6rem 0}
.card a{text-decoration:none;font-weight:600}
.spec{background:var(--spec);border:1px solid var(--specline);border-radius:6px;
padding:.6rem .9rem;margin:1rem 0;font-size:.85rem}
.hit{color:var(--hit);font-weight:600}.miss{color:var(--miss);font-weight:600}
.pending{color:var(--pend)}.unresolvable{color:var(--dim)}
.tag{display:inline-block;padding:.1rem .45rem;border:1px solid var(--line);
border-radius:3px;font-size:.7rem;font-family:ui-monospace,monospace;color:var(--dim)}
ul{padding-left:1.2rem}
"""

NAV = ('<nav><a href="/">briefs</a><a href="/theses">theses</a>'
       '<a href="/scoreboard">scoreboard</a></nav>')


def page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(
        f"<!doctype html><html><head><meta charset=utf-8>"
        f"<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{title}</title><style>{CSS}</style></head>"
        f"<body>{NAV}{body}</body></html>"
    )


def to_html(text: str) -> str:
    return md.markdown(text, extensions=["tables", "fenced_code"])


def create_app(cfg: Config) -> FastAPI:
    app = FastAPI(title="earnings-agent", docs_url=None, redoc_url=None)

    def store() -> Store:
        # SQLite connections are not shareable across threads, so open per
        # request. The database is tiny and local; this costs nothing.
        return Store(cfg.db_path)

    @app.get("/", response_class=HTMLResponse)
    def index():
        s = store()
        try:
            briefs = s.recent_briefs(50)
            counts = s.claim_stats()
            card = scoring.Scorecard()
            for status, n in counts.items():
                if hasattr(card, status):
                    for _ in range(n):
                        card.add(status)

            body = [f"<h1>earnings-agent</h1>",
                    f"<p class=meta>watching {', '.join(cfg.tickers)} · "
                    f"{cfg.model} via {cfg.provider}</p>"]
            if counts:
                body.append(f"<p class=meta>claims: {card.summary()}</p>")
            body.append("<h2>Briefs</h2>")
            if not briefs:
                body.append("<p class=meta>None yet. Run <code>earnings-agent run</code>.</p>")
            for b in briefs:
                body.append(
                    f"<div class=card><a href='/brief/{b['accession']}'>{b['ticker']}</a> "
                    f"<span class=tag>{b['accession']}</span>"
                    f"<div class=meta>{b['created_at']} · {b['model'] or 'figures only'}</div>"
                    f"</div>"
                )
            return page("earnings-agent", "".join(body))
        finally:
            s.close()

    @app.get("/brief/{accession}", response_class=HTMLResponse)
    def brief(accession: str):
        s = store()
        try:
            row = s.get_brief(accession)
            if row is None:
                raise HTTPException(404, "no such brief")
            theses = [t for t in s.theses(row["ticker"]) if t["accession"] == accession]
            extra = ""
            if theses:
                links = " ".join(
                    f"<a href='/thesis/{t['id']}'>#{t['id']} {t['direction']}</a>"
                    for t in theses)
                extra = f"<div class=spec><strong>Theses from this filing:</strong> {links}</div>"
            return page(f"{row['ticker']} brief", to_html(row["markdown"]) + extra)
        finally:
            s.close()

    @app.get("/theses", response_class=HTMLResponse)
    def theses():
        s = store()
        try:
            rows = s.theses(limit=100)
            body = ["<h1>Theses</h1>",
                    "<div class=spec>Speculative. Unlike briefs, nothing here is "
                    "quote-backed — each exists to be scored against what actually "
                    "happens.</div>"]
            if not rows:
                body.append("<p class=meta>None yet. Run "
                            "<code>earnings-agent thesis TICKER</code>.</p>")
            for t in rows:
                claims = s.claims_for(t["id"])
                hits = sum(1 for c in claims if c["status"] == "hit")
                settled = sum(1 for c in claims if c["status"] in ("hit", "miss"))
                tally = f"{hits}/{settled} correct" if settled else f"{len(claims)} pending"
                peers = json.loads(t["peers"] or "[]")
                body.append(
                    f"<div class=card><a href='/thesis/{t['id']}'>{t['ticker']} "
                    f"— {t['direction']}</a> <span class=tag>{t['confidence']}</span>"
                    f"<div class=meta>{t['created_at'][:10]} · {tally} · "
                    f"peers: {', '.join(peers) or 'none'}</div>"
                    f"<div>{t['summary']}</div></div>"
                )
            return page("theses", "".join(body))
        finally:
            s.close()

    @app.get("/thesis/{thesis_id}", response_class=HTMLResponse)
    def thesis(thesis_id: int):
        from earnings_agent.render import render_thesis
        from earnings_agent.thesis import Thesis

        s = store()
        try:
            row = s.get_thesis(thesis_id)
            if row is None:
                raise HTTPException(404, "no such thesis")
            obj = Thesis.model_validate_json(row["payload"])
            html = to_html(render_thesis(
                row["ticker"], obj, json.loads(row["peers"] or "[]"),
                row["model"], s.claims_for(thesis_id),
            ))
            back = f"<p class=meta><a href='/brief/{row['accession']}'>← the brief this came from</a></p>"
            return page(f"{row['ticker']} thesis", html + back)
        finally:
            s.close()

    @app.get("/scoreboard", response_class=HTMLResponse)
    def scoreboard():
        s = store()
        try:
            tickers = sorted({t["ticker"] for t in s.theses(limit=500)})
            rows = ["<h1>Scoreboard</h1>",
                    "<div class=spec>Claims are settled by arithmetic against reported "
                    "XBRL figures — no model grades its own work.</div>",
                    "<table><tr><th>Ticker</th><th>Correct</th><th>Wrong</th>"
                    "<th>Pending</th><th>Manual</th><th>Hit rate</th></tr>"]
            overall = scoring.Scorecard()
            for t in tickers:
                card = scoring.Scorecard()
                for status, n in s.claim_stats(t).items():
                    if hasattr(card, status):
                        for _ in range(n):
                            card.add(status)
                            overall.add(status)
                rate = f"{card.hit_rate:.0f}%" if card.hit_rate is not None else "—"
                rows.append(
                    f"<tr><td>{t}</td><td class=hit>{card.hit}</td>"
                    f"<td class=miss>{card.miss}</td><td class=pending>{card.pending}</td>"
                    f"<td class=unresolvable>{card.unresolvable}</td><td>{rate}</td></tr>")
            orate = f"{overall.hit_rate:.0f}%" if overall.hit_rate is not None else "—"
            rows.append(f"<tr><td><strong>all</strong></td><td class=hit>{overall.hit}</td>"
                        f"<td class=miss>{overall.miss}</td><td class=pending>{overall.pending}</td>"
                        f"<td class=unresolvable>{overall.unresolvable}</td>"
                        f"<td><strong>{orate}</strong></td></tr></table>")
            if 0 < overall.settled < 10:
                rows.append(f"<p class=meta>{overall.settled} settled claims is far too "
                            "few to read anything into that rate.</p>")
            if not tickers:
                rows.append("<p class=meta>No theses yet.</p>")
            return page("scoreboard", "".join(rows))
        finally:
            s.close()

    @app.get("/favicon.ico")
    def favicon():
        return RedirectResponse("/", status_code=302)

    return app
