"""Transcript enrichment.

Deliberately a stub with a real interface. The measured reason: the 8-K press
release lands on EDGAR within minutes of release - typically before the call
even starts - while transcripts take 1-3 hours even from enterprise vendors
(Aiera's own claim is 99% accuracy "within one to three hours of event
completion"). So a brief must never wait on a transcript.

But transcripts are not optional polish either. Apple, Microsoft and Alphabet
publish no written guidance; for those issuers the transcript is the *only*
path to forward guidance. Hence: a queued enrichment that upgrades a brief
after the fact, not a blocking dependency that delays it.

To wire a real vendor, implement TranscriptProvider and register it. Nothing
else in the pipeline changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class Transcript:
    ticker: str
    period: str
    text: str
    source: str
    retrieved_at: str


@runtime_checkable
class TranscriptProvider(Protocol):
    name: str

    def fetch(self, ticker: str, filing_date: str) -> Transcript | None:
        """Return the call transcript, or None if not yet available.

        Returning None must mean "not ready, ask again later" - it is not an
        error. Raise only for genuine failures (auth, network, bad request) so
        the pipeline can tell a pending transcript from a broken provider.
        """
        ...


class NullProvider:
    """Default provider: no transcripts configured.

    Every enrichment stays pending, which is the honest state - the brief is
    complete on the filings path and the transcript is simply unavailable.
    """

    name = "null"

    def fetch(self, ticker: str, filing_date: str) -> Transcript | None:
        return None


_PROVIDER: TranscriptProvider = NullProvider()


def register(provider: TranscriptProvider) -> None:
    global _PROVIDER
    _PROVIDER = provider


def provider() -> TranscriptProvider:
    return _PROVIDER
