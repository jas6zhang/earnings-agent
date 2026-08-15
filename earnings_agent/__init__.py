"""Filings-first earnings monitor.

The design premise: the 8-K earnings press release (Item 2.02, exhibit EX-99.1)
lands on EDGAR within minutes of release - typically *before* the earnings call
even starts - and it contains the forward guidance. Transcripts take hours even
from paid vendors. So filings are the primary path and transcripts are a
late-arriving enrichment that must never block a brief from being emitted.
"""

__version__ = "0.1.0"
