"""Guidance extraction.

Scope discipline matters here. The model reads *prose* - it decides whether a
sentence is real forward guidance or safe-harbor boilerplate, and it quotes the
passage it relied on. It does not compute, restate, or estimate financials:
every number in a brief comes from XBRL (see xbrl.py). Keeping those two jobs
apart is what makes the output trustworthy.

Two backends. The OpenAI-compatible one covers every free provider worth using
(Google AI Studio, Groq, OpenRouter, Cerebras, a local Ollama) since they all
speak /chat/completions; the Anthropic one is kept for when quality matters
more than cost. Only the transport differs - schema and prompt are shared, so
output is comparable across providers.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger("earnings_agent")

# Empirically, roughly half of large-cap 8-K releases carry no written guidance
# at all - Apple, Microsoft and Alphabet all guide verbally on the call or not
# at all. "No guidance found" is a real, useful answer, so the schema forces the
# model to state which case it is rather than returning a silently empty list.
GuidanceStatus = Literal[
    "written_guidance_found",
    "no_written_guidance",
    "issuer_does_not_guide",
    "unclear",
]


class GuidanceItem(BaseModel):
    metric: str = Field(description="What is being guided, e.g. 'Total revenue', 'Diluted EPS'")
    period: str = Field(description="Period the guidance covers, e.g. 'Q3 FY2026', 'FY2026'")
    value: str = Field(description="The guided figure exactly as stated, e.g. '$61-64 billion'")
    quote: str = Field(description="Verbatim sentence from the release that states this guidance")
    change_vs_prior: str = Field(
        description="How this compares to prior guidance if the release says so, "
        "else 'not stated in this release'"
    )


class KeyPoint(BaseModel):
    point: str = Field(description="One material item an investor should know")
    quote: str = Field(description="Verbatim supporting sentence from the release")


class Brief(BaseModel):
    guidance_status: GuidanceStatus
    guidance_status_note: str = Field(
        description="One sentence explaining the status. If no written guidance, say whether "
        "the issuer appears to guide verbally on the call or does not guide at all."
    )
    guidance: list[GuidanceItem]
    key_points: list[KeyPoint]
    watch_items: list[str] = Field(
        description="Things a reader should follow up on - unexplained changes, one-time items, "
        "accounting changes, anything the release is vague about. Empty list if none."
    )
    tone_shift: str = Field(
        description="Any notable change in management's framing vs a typical release "
        "(new risk language, hedged phrasing, unusual emphasis), or 'nothing notable'."
    )


SYSTEM = """You are an equity research analyst reading an 8-K earnings press release.

Your single job is to separate signal from boilerplate. Specifically:

FORWARD GUIDANCE vs SAFE-HARBOR BOILERPLATE
Every release contains a "forward-looking statements" legal disclaimer. That is
NOT guidance. Real guidance is a specific forward projection of a financial
metric for a named future period - "we expect Q3 revenue of $61-64 billion",
"full year EPS of $4.20 to $4.40", "gross margin of approximately 50%".

The following are NOT guidance, and you must not report them as such:
  - Safe-harbor disclaimers ("this release contains forward-looking statements
    within the meaning of the Private Securities Litigation Reform Act...")
  - Descriptions of risk factors
  - Backward-looking commentary about the quarter that just ended
  - Non-GAAP reconciliation explanations
  - Generic optimism with no figure ("we are well positioned for growth")

Many issuers publish no written guidance at all - some give it verbally on the
earnings call, others do not guide. If you find no real written guidance, say so
via guidance_status. An empty guidance list with an honest status is a correct
and useful answer. Do not manufacture guidance to fill the field.

RULES
- Quote verbatim. Every guidance item and key point must carry the exact
  sentence from the release that supports it. Never paraphrase into the quote
  field.
- Do not compute anything. Do not restate revenue, margins, or growth rates as
  your own analysis - exact financials are pulled separately from XBRL filings
  and will be shown alongside your output. If you mention a figure, it must be
  inside a verbatim quote.
- If the release does not say something, do not infer it. "not stated in this
  release" is the correct value for an unknown.
- Prefer fewer, higher-signal key points over an exhaustive list."""


def _user_prompt(ticker: str, filing_date: str, press_release: str) -> str:
    return (
        f"8-K earnings press release for {ticker}, filed {filing_date}.\n\n"
        f"<press_release>\n{press_release}\n</press_release>"
    )


class RefusalError(RuntimeError):
    """The model declined the request."""


class ExtractionError(RuntimeError):
    """The model returned something that is not a valid Brief."""


class Extractor(Protocol):
    model: str

    def complete(self, system: str, user: str, model_cls: type[BaseModel]) -> tuple[Any, str]:
        """Return a validated instance of `model_cls`, plus the model id that produced it."""
        ...

    def extract(self, ticker: str, filing_date: str, press_release: str) -> tuple[Brief, str]:
        ...


# -- JSON schema helpers ---------------------------------------------------

def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve $ref/$defs into a self-contained schema.

    Pydantic emits nested models as $defs + $ref. Several providers' structured
    output implementations reject or silently mishandle refs, so flatten them.
    Our schema is a shallow tree with no recursion, which makes this safe.
    """
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                return walk({k: v for k, v in defs[name].items()})
            return {k: walk(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def schema_for(model_cls: type[BaseModel]) -> dict[str, Any]:
    """A model's JSON schema, flattened and closed for strict-mode providers."""
    def close(node: Any) -> Any:
        if isinstance(node, dict):
            node = {k: close(v) for k, v in node.items()}
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"].keys())
            return node
        if isinstance(node, list):
            return [close(v) for v in node]
        return node

    return close(_inline_refs(model_cls.model_json_schema()))


def _strict_schema() -> dict[str, Any]:
    return schema_for(Brief)


# -- backends --------------------------------------------------------------

class AnthropicExtractor:
    """Anthropic native. Validated structured output via messages.parse()."""

    def __init__(self, api_key: str, model: str = "claude-opus-5", max_tokens: int = 16000):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str, model_cls: type[BaseModel]) -> tuple[Any, str]:
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            # The system prompt is byte-identical across every call of a given
            # kind, so it caches. Opus 5's minimum cacheable prefix is 512 tokens.
            system=[{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }],
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            messages=[{"role": "user", "content": user}],
            output_format=model_cls,
        )
        # Opus 5 runs safety classifiers that can decline a request outright.
        # Unlikely here, but reading .parsed_output unconditionally would raise
        # something unhelpful.
        if response.stop_reason == "refusal":
            raise RefusalError(f"model declined ({getattr(response.stop_details, 'category', None)})")
        return response.parsed_output, response.model

    def extract(self, ticker: str, filing_date: str, press_release: str) -> tuple[Brief, str]:
        return self.complete(SYSTEM, _user_prompt(ticker, filing_date, press_release), Brief)


class OpenAICompatExtractor:
    """Any /chat/completions endpoint: Gemini, Groq, OpenRouter, Ollama, ...

    Structured-output support varies a lot across free providers, so this walks
    a ladder rather than assuming: strict json_schema first, then plain JSON
    mode with the schema in the prompt, then a validation-feedback retry. A
    weaker free model that fumbles the shape once usually gets it on the retry.
    """

    def __init__(self, api_key: str, base_url: str, model: str, max_tokens: int = 8192):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=180.0)
        self.model = model
        self.max_tokens = max_tokens
        self._supports_json_schema = True

    @staticmethod
    def _messages(system: str, user: str, schema: dict | None):
        if schema is not None:
            system += (
                "\n\nRespond with a single JSON object and nothing else - no prose, no "
                "markdown fence. It must validate against this JSON Schema:\n"
                + json.dumps(schema, indent=2)
            )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _call(self, messages: list[dict], response_format: dict | None) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format
        completion = self.client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        if choice.finish_reason == "content_filter":
            raise RefusalError("provider content filter blocked the request")
        if choice.finish_reason == "length":
            raise ExtractionError(
                f"response hit the {self.max_tokens}-token cap before completing - "
                "raise llm.max_tokens"
            )
        return choice.message.content or ""

    @staticmethod
    def _parse(raw: str, model_cls: type[BaseModel]) -> Any:
        text = raw.strip()
        # Some models fence JSON despite being told not to.
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        return model_cls.model_validate_json(text)

    def complete(self, system: str, user: str, model_cls: type[BaseModel]) -> tuple[Any, str]:
        """Get a validated `model_cls` back, walking the structured-output ladder."""
        schema = schema_for(model_cls)
        name = model_cls.__name__.lower()

        # 1. strict json_schema, if this provider honours it
        if self._supports_json_schema:
            try:
                raw = self._call(
                    self._messages(system, user, None),
                    {"type": "json_schema", "json_schema": {
                        "name": name, "schema": schema, "strict": True}},
                )
                return self._parse(raw, model_cls), self.model
            except (RefusalError, ExtractionError):
                raise
            except Exception as e:
                log.info("%s rejected json_schema mode (%s) - falling back to JSON mode",
                         self.model, type(e).__name__)
                self._supports_json_schema = False

        # 2. plain JSON mode with the schema stated in the prompt
        messages = self._messages(system, user, schema)
        try:
            raw = self._call(messages, {"type": "json_object"})
        except (RefusalError, ExtractionError):
            raise
        except Exception as e:
            log.info("%s rejected json_object mode (%s) - falling back to prompt only",
                     self.model, type(e).__name__)
            raw = self._call(messages, None)

        try:
            return self._parse(raw, model_cls), self.model
        except (ValidationError, ValueError) as e:
            # 3. one retry, showing the model exactly what was wrong
            log.info("%s returned invalid %s - retrying with the validation error",
                     self.model, model_cls.__name__)
            messages += [
                {"role": "assistant", "content": raw[:4000]},
                {"role": "user", "content":
                    f"That did not validate against the schema:\n\n{e}\n\n"
                    "Return the corrected JSON object only."},
            ]
            retry = self._call(messages, {"type": "json_object"})
            try:
                return self._parse(retry, model_cls), self.model
            except (ValidationError, ValueError) as e2:
                raise ExtractionError(f"{self.model} produced invalid output twice: {e2}") from e2

    def extract(self, ticker: str, filing_date: str, press_release: str) -> tuple[Brief, str]:
        return self.complete(SYSTEM, _user_prompt(ticker, filing_date, press_release), Brief)


def build(provider: str, api_key: str, model: str, max_tokens: int,
          base_url: str | None) -> Extractor:
    if provider == "anthropic":
        return AnthropicExtractor(api_key, model, max_tokens)
    if provider == "openai-compat":
        if not base_url:
            raise ValueError("llm.base_url is required for provider 'openai-compat'")
        return OpenAICompatExtractor(api_key, base_url, model, max_tokens)
    raise ValueError(f"unknown llm.provider {provider!r} (use 'openai-compat' or 'anthropic')")
