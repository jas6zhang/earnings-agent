"""Thesis generation — speculative, quarantined, and scored.

This is the one part of the system that reasons rather than extracts, so it is
built to be checked rather than trusted:

  - Every claim carries a *structured numeric check*, not just prose. Scoring
    is then deterministic arithmetic against XBRL actuals - no model in the
    verification loop, for the same reason no model touches the figures.
  - Peer briefs go in alongside the subject's. A cyclical name is driven by an
    industry pricing cycle visible in competitors' filings before its own;
    reasoning from one issuer is how you get a confident, uninformed answer.
  - The schema forces the model to state its blind spots. Filings do not
    contain spot pricing, channel inventory, or hyperscaler capex, and a thesis
    that does not say so is overclaiming.

Output never merges into the brief. Briefs are quote-backed and checkable in
seconds; a thesis is neither, and mixing them would launder one into the other.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Checks a claim can be resolved against. Everything except "qualitative"
# reduces to arithmetic over XBRL facts or a price series.
CheckKind = Literal[
    "revenue_yoy_pct",
    "revenue_usd",
    "gross_margin_pct",
    "operating_margin_pct",
    "net_margin_pct",
    "eps_diluted_usd",
    "price_return_pct",
    "qualitative",
]


class Claim(BaseModel):
    statement: str = Field(
        description="The falsifiable claim in plain English, e.g. "
        "'Revenue growth decelerates below 40% year over year next quarter'"
    )
    check: CheckKind = Field(
        description="Which measurable quantity settles this claim. Use 'qualitative' ONLY "
        "when no number could settle it - prefer a numeric check wherever possible."
    )
    comparator: Literal["above", "below"] = Field(
        description="Whether the claim is that the measured value lands above or below "
        "the threshold"
    )
    threshold: float = Field(
        description="The number to compare against, in the check's natural unit: percent "
        "for *_pct checks, US dollars for *_usd checks. Ignored for 'qualitative'."
    )
    horizon_quarters: int = Field(
        description="How many reported quarters ahead this settles. 1 = the next quarter."
    )
    disconfirms: str = Field(
        description="What observation would show this claim was wrong, stated concretely."
    )


class PeerSignal(BaseModel):
    ticker: str
    signal: str = Field(description="What this peer's filing says that bears on the subject")
    quote: str = Field(description="Verbatim sentence from the peer's release")


class Thesis(BaseModel):
    direction: Literal["bullish", "bearish", "neutral"]
    confidence: Literal["low", "medium", "high"]
    summary: str = Field(description="One or two sentences. The thesis itself.")
    reasoning: str = Field(
        description="The argument, in a short paragraph. Reference only what is in the "
        "supplied filings - do not import outside knowledge as if it were evidence."
    )
    rests_on: list[str] = Field(
        description="The specific facts from the supplied briefs this argument depends on. "
        "If one of these turned out wrong, the thesis would weaken."
    )
    peer_signals: list[PeerSignal] = Field(
        description="Cross-company evidence. Empty list if no peer briefs were supplied."
    )
    claims: list[Claim] = Field(description="2-4 falsifiable claims. Prefer numeric checks.")
    key_uncertainty: str = Field(description="The single thing most likely to make this wrong.")
    what_we_cannot_see: str = Field(
        description="What would be needed to hold this view with real confidence, that is "
        "NOT in the supplied filings. Be specific and honest."
    )


SYSTEM = """You are a buy-side analyst forming a view from SEC filings alone.

WHAT YOU HAVE
Earnings press releases (8-K exhibit 99.1) for a subject company and, usually,
its industry peers. Exact reported figures accompany each one, taken from XBRL.

WHAT YOU DO NOT HAVE, AND MUST NOT PRETEND TO
Spot commodity pricing. Channel and distributor inventory. Hyperscaler capex
plans. Supply agreements. Sell-side estimates or consensus. Anything management
said verbally on the call. Current share price or valuation multiples, unless a
price series is explicitly supplied below.

For many companies - especially commodity and cyclical names - the dominant
driver of the share price lives in exactly those places and not in the filings.
When that is the case, say so plainly in what_we_cannot_see and lower your
confidence. A calibrated "I can see maybe a third of what matters here" is far
more useful than a fluent argument built on a fraction of the picture.

HOW TO REASON
- Peers first. An industry pricing cycle usually shows up in competitors'
  language before it shows up in the subject's reported numbers. Read across
  the filings, not just down one.
- Ground every step in a supplied fact. If you catch yourself reciting general
  market knowledge, stop - that is not evidence you were given, and you cannot
  check whether it is still true.
- Distinguish what changed this quarter from what was already known.

CLAIMS
Each claim must be settleable by someone with no judgement, months from now.
Give a numeric check wherever one exists: pick the measurable quantity, a
comparator, and a threshold. "Margins stay strong" is not a claim. "Gross
margin above 52% next quarter" is.
Use 'qualitative' only when genuinely no number settles it, and expect that
those claims will simply go unresolved on the scoreboard.
Prefer 2-4 sharp claims to a longer, hedged list. A claim you would be
embarrassed to be wrong about is the right kind of claim.

CONFIDENCE
low     - the filings touch the question only indirectly
medium  - the filings support a direction but the main drivers sit outside them
high    - the filings themselves largely settle the question

Most theses about cyclical companies should be low or medium. If you find
yourself writing high, check whether you are relying on something you were not
actually given."""


def build_prompt(
    subject: str,
    subject_brief: str,
    peer_briefs: dict[str, str],
    price_context: str | None,
) -> str:
    parts = [f"Subject: {subject}\n", f"<brief ticker=\"{subject}\">\n{subject_brief}\n</brief>\n"]

    if peer_briefs:
        parts.append("Peer filings, most recent quarter each:\n")
        for t, b in peer_briefs.items():
            parts.append(f'<brief ticker="{t}">\n{b}\n</brief>\n')
    else:
        parts.append(
            "No peer filings were supplied. You are seeing one company in isolation, "
            "which is a serious limitation for any industry-driven name - reflect that "
            "in confidence and in what_we_cannot_see.\n"
        )

    if price_context:
        parts.append(f"\nPrice context (unofficial source, for calibration only):\n{price_context}\n")
    else:
        parts.append(
            "\nNo price data supplied. Do not speculate about valuation or how much of "
            "your view is already priced in.\n"
        )

    parts.append(
        f"\nForm a thesis on {subject}. Ground it in the filings above, state your blind "
        "spots honestly, and give claims that can be settled against future reported "
        "figures."
    )
    return "\n".join(parts)
