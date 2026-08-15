"""Config loading and validation."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PLACEHOLDER_UA = "CHANGE ME"


class ConfigError(Exception):
    pass


@dataclass
class Config:
    user_agent: str
    requests_per_second: float
    provider: str
    base_url: str | None
    model: str
    max_tokens: int
    api_key_env: str
    tickers: list[str]
    peer_groups: dict[str, list[str]]
    thesis_model: str
    output_dir: Path
    root: Path
    api_key: str | None = field(default=None, repr=False)

    @property
    def db_path(self) -> Path:
        return self.root / "earnings.db"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.api_key)


def load(path: str | Path | None = None) -> Config:
    root = Path(path).parent if path else Path(__file__).resolve().parent.parent
    cfg_path = Path(path) if path else root / "config.toml"
    if not cfg_path.exists():
        raise ConfigError(f"No config at {cfg_path}")

    with cfg_path.open("rb") as fh:
        raw = tomllib.load(fh)

    ua = os.environ.get("EDGAR_USER_AGENT") or raw.get("edgar", {}).get("user_agent", "")
    if not ua or PLACEHOLDER_UA in ua:
        raise ConfigError(
            f"Set a real edgar.user_agent in {cfg_path} (or the EDGAR_USER_AGENT env var).\n"
            "SEC requires an identifying User-Agent with a contact address and "
            "will block requests without one. Format: 'Your Name you@example.com'"
        )

    tickers = [t.strip().upper() for t in raw.get("watchlist", {}).get("tickers", []) if t.strip()]
    if not tickers:
        raise ConfigError(f"watchlist.tickers is empty in {cfg_path}")

    out = raw.get("output", {}).get("dir", "briefs")
    out_dir = Path(out) if Path(out).is_absolute() else root / out

    llm = raw.get("llm", {})
    provider = llm.get("provider", "openai-compat")
    if provider not in ("openai-compat", "anthropic"):
        raise ConfigError(
            f"llm.provider must be 'openai-compat' or 'anthropic', got {provider!r}"
        )
    key_env = llm.get("api_key_env") or (
        "ANTHROPIC_API_KEY" if provider == "anthropic" else "GEMINI_API_KEY"
    )

    return Config(
        user_agent=ua,
        requests_per_second=float(raw.get("edgar", {}).get("requests_per_second", 8.0)),
        provider=provider,
        base_url=llm.get("base_url"),
        model=llm.get("model", "gemini-2.5-flash"),
        max_tokens=int(llm.get("max_tokens", 8192)),
        api_key_env=key_env,
        tickers=tickers,
        peer_groups={
            k: [t.strip().upper() for t in v if t.strip()]
            for k, v in raw.get("peers", {}).items()
        },
        # Extraction is quote-backed and cheap to verify, so a small free model
        # is fine. A thesis is neither, so it defaults to the same model but is
        # separately configurable - point it at something stronger.
        thesis_model=raw.get("thesis", {}).get("model") or llm.get("model", "gemini-2.5-flash"),
        output_dir=out_dir,
        root=root,
        api_key=os.environ.get(key_env),
    )
