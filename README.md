# earnings-agent

A filings-first earnings monitor. Watches a list of tickers, and when an issuer
files an earnings 8-K it emits a brief: forward guidance pulled out of the press
release with verbatim citations, alongside exact financials pulled from XBRL.

## Why filings first

The 8-K earnings press release (Item 2.02, exhibit EX-99.1) lands on EDGAR
within minutes of release — usually *before* the earnings call starts, since
issuers file the release and then host the call 30–60 minutes later. Call
transcripts take 1–3 hours even from enterprise vendors.

So the fastest path to the numbers is free and public, and this tool takes it.

The catch, measured across large caps: **only about half of issuers publish
written guidance in the 8-K.** NVIDIA, Meta, Broadcom, Amazon, Salesforce and
Walmart do. Apple, Microsoft and Alphabet do not — they guide verbally on the
call, or not at all.

That's why transcripts are a *queued enrichment* rather than a blocking
dependency. A brief ships minutes after the filing; if a transcript provider is
configured, the Q&A and verbal guidance get attached later. Nothing waits.

```
T+0min    8-K EX-99.1 hits EDGAR    →  brief emitted            [free]
T+30min   call begins
T+90min   call ends
T+3hr     transcript available      →  enrichment attached      [$$$]
```

## What is and isn't model-generated

Kept deliberately separate, because a financial tool that blurs this is not
trustworthy:

| | Source |
|---|---|
| Revenue, margins, EPS, balance sheet, YoY | **SEC XBRL `companyfacts`.** Exact values, no LLM in the path, each traceable to the accession number that reported it. |
| Guidance, key points, watch items, tone | **Claude**, reading the press release prose only. Every claim carries a verbatim quote. The model is explicitly instructed not to compute or restate figures. |

## Setup

```bash
uv venv --python 3.12
uv pip install httpx beautifulsoup4 lxml anthropic pytest
```

Two pieces of configuration:

1. **`edgar.user_agent` in `config.toml`** — required. The SEC blocks requests
   without an identifying User-Agent carrying a contact address, and rate-limits
   at 10 req/sec. Format: `Your Name you@example.com`. Overridable with the
   `EDGAR_USER_AGENT` env var.
2. **An LLM key** — optional. Without one the agent still runs and emits exact
   figures, just no narrative extraction.

Then edit `watchlist.tickers`.

### Choosing a model

Defaults to **Google AI Studio's free tier** — no credit card, ~1,500 req/day,
1M context. Get a key at <https://aistudio.google.com/apikey>, then:

```bash
export GEMINI_API_KEY=...
earnings-agent models      # list what your key can actually reach
```

Model IDs churn, so `models` is the source of truth rather than the default in
`config.toml`.

> On Google's **free** tier, prompts are used to improve their products (paid
> tier is not). Everything sent here is a public SEC filing, so that's fine —
> but don't point this at non-public data without switching tiers.

Any OpenAI-compatible endpoint works by changing three lines — Groq, OpenRouter,
Cerebras, a local Ollama. Claude is also supported via `provider = "anthropic"`.
See the comments in `config.toml`.

Structured-output support varies a lot across free providers, so extraction
walks a ladder: strict `json_schema` → JSON mode with the schema in the prompt →
a retry that feeds the validation error back. A weaker model that fumbles the
shape once usually gets it on the retry.

## Use

```bash
earnings-agent status              # watchlist, credentials, pipeline state
earnings-agent models              # models the configured key can reach
earnings-agent brief NVDA          # one-off brief for a ticker, to stdout
earnings-agent brief AAPL --no-llm # figures only
earnings-agent run                 # poll → process → enrich → score
earnings-agent thesis SNDK         # speculative, scoreable view
earnings-agent score               # settle claims, show the scoreboard
earnings-agent serve               # local web UI on 127.0.0.1:8000
```

### Web UI

```bash
earnings-agent serve                              # on the devserver
ssh -L 8000:localhost:8000 <devserver>            # from your laptop
```

Bound to loopback deliberately — there's no authentication, so it must only be
reachable through the tunnel. Pages: briefs, theses, scoreboard.

## Theses

Separate from briefs, and deliberately so. A brief is quote-backed and checkable
against the filing in seconds. A thesis is an *argument*, and mixing the two
would launder one into the other — so they live in different tables, render
differently, and are labelled speculative.

Three things make it more than a plausible-sounding narrative:

**Peer sets.** A cyclical name is driven by an industry cycle visible in
competitors' filings before its own. SanDisk is the clean example: the stock ran
from roughly \$43 to \$2,354 in a year on a NAND shortage, and almost none of
that signal is in SanDisk's own 8-K. So `[peers]` in config declares the group,
and a thesis reads across all of them.

Peer groups are declared, not inferred — SEC SIC codes file SanDisk under
"Computer Storage Devices" and Micron under "Semiconductors", splitting exactly
the two companies you'd want side by side.

**Machine-checkable claims.** Every claim carries a structured check —
quantity, comparator, threshold, horizon — not just prose. "Margins stay strong"
is rejected; "gross margin above 52% next quarter" is a claim. Settling it is
then arithmetic against XBRL, with no model in the verification loop, for the
same reason no model touches the figures.

**A scoreboard.** `score` resolves outstanding claims against reported actuals
and tracks a hit rate. This is the part that converts theses from noise into
something you can calibrate trust against. Hit rate reads `—` rather than 0%
until something actually settles.

The schema also forces the model to fill in `what_we_cannot_see`. Filings don't
contain spot pricing, channel inventory, or hyperscaler capex, and a thesis that
doesn't say so is overclaiming.

Point the thesis layer at a stronger model than extraction (`[thesis] model`).
Quote-backed extraction is safe on a small free model because every claim is
checkable; an unverifiable argument is not.

`run` is idempotent — filings already briefed are skipped — so it is safe on a
cron. The three stages are also separately invocable (`poll`, `process`) if you
want them on different cadences; `enrich` belongs on a slower one than the rest.

## Adding a transcript provider

Implement `TranscriptProvider` in `transcripts.py` and call `register()`.
Nothing else in the pipeline changes.

```python
class MyProvider:
    name = "my-vendor"
    def fetch(self, ticker: str, filing_date: str) -> Transcript | None:
        ...  # return None for "not ready yet" — it is not an error
```

Before buying one: verify latency and price on a free tier first. Vendor claims
in this space are frequently unverifiable — Aiera's "99% accuracy" has no
published WER benchmark, and at least one low-cost vendor lists three different
coverage numbers on its own homepage. Note also that a **seat licence does not
grant redistribution** — fine for personal use, not for anything you ship.

## Known sharp edges

- **Filers change XBRL tags.** NVIDIA reported revenue as
  `RevenueFromContractWithCustomerExcludingAssessedTax` until 2020 and as
  `Revenues` after. The fallback chains in `xbrl.py` merge per *period*, not per
  tag, so a stale tag can't strand current data. Covered by tests.
- **Statements mix period lengths.** Apple reports cash flow year-to-date in
  10-Qs. Selecting each line independently produces a table that looks right and
  is not, so `income_statement()` returns one coherent period and reports what it
  omitted; `ytd()` surfaces the cumulative figures separately.
- **Not every line exists for every filer.** Alphabet, Amazon and Meta publish
  no `GrossProfit` subtotal; Amazon has no `ResearchAndDevelopmentExpense`.
  These show as explicit omissions rather than zeros.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

Offline, using synthetic `companyfacts` blobs. Both main cases are regressions
from bugs found against live SEC data.
