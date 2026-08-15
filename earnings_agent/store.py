"""SQLite persistence.

Two jobs: dedupe (never brief the same filing twice) and enrichment tracking
(remember that a transcript is still owed for a filing we already briefed).
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS filings (
    accession     TEXT PRIMARY KEY,
    cik           TEXT NOT NULL,
    ticker        TEXT NOT NULL,
    form          TEXT NOT NULL,
    items         TEXT,
    filing_date   TEXT,
    report_date   TEXT,
    acceptance_dt TEXT,
    primary_doc   TEXT,
    discovered_at TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'new'
);
CREATE INDEX IF NOT EXISTS idx_filings_status ON filings(status);
CREATE INDEX IF NOT EXISTS idx_filings_ticker ON filings(ticker);

CREATE TABLE IF NOT EXISTS briefs (
    accession  TEXT PRIMARY KEY REFERENCES filings(accession),
    ticker     TEXT NOT NULL,
    created_at TEXT NOT NULL,
    model      TEXT,
    markdown   TEXT NOT NULL,
    extraction TEXT
);

-- An enrichment is something we know we want but cannot have yet, e.g. the
-- call transcript. The brief ships without it; this row is the reminder.
CREATE TABLE IF NOT EXISTS enrichments (
    accession  TEXT NOT NULL REFERENCES filings(accession),
    kind       TEXT NOT NULL,
    status     TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    detail     TEXT,
    body       TEXT,
    PRIMARY KEY (accession, kind)
);
CREATE INDEX IF NOT EXISTS idx_enrich_status ON enrichments(status);

-- Theses are kept apart from briefs on purpose. A brief is quote-backed and
-- checkable against the filing in seconds; a thesis is an argument. Storing
-- them in one table would invite rendering them as one thing.
CREATE TABLE IF NOT EXISTS theses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT NOT NULL,
    accession   TEXT NOT NULL,
    base_end    TEXT NOT NULL,   -- period end of the brief it was built from
    created_at  TEXT NOT NULL,
    model       TEXT,
    direction   TEXT,
    confidence  TEXT,
    summary     TEXT,
    peers       TEXT,            -- json list of tickers actually supplied
    payload     TEXT             -- full Thesis json
);
CREATE INDEX IF NOT EXISTS idx_theses_ticker ON theses(ticker);

-- One row per falsifiable claim. status starts 'pending' and is settled by
-- arithmetic in scoring.py, never by a model.
CREATE TABLE IF NOT EXISTS claims (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    thesis_id        INTEGER NOT NULL REFERENCES theses(id),
    statement        TEXT NOT NULL,
    check_kind       TEXT NOT NULL,
    comparator       TEXT NOT NULL,
    threshold        REAL NOT NULL,
    horizon_quarters INTEGER NOT NULL,
    disconfirms      TEXT,
    status           TEXT NOT NULL DEFAULT 'pending',
    actual           REAL,
    note             TEXT,
    resolved_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(status);
CREATE INDEX IF NOT EXISTS idx_claims_thesis ON claims(thesis_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # -- filings ----------------------------------------------------------

    def seen(self, accession: str) -> bool:
        cur = self.conn.execute("SELECT 1 FROM filings WHERE accession = ?", (accession,))
        return cur.fetchone() is not None

    def record_filing(self, f: dict[str, Any]) -> bool:
        """Insert a filing. Returns True if newly recorded, False if already known."""
        if self.seen(f["accession"]):
            return False
        with self.tx() as c:
            c.execute(
                """INSERT INTO filings
                   (accession, cik, ticker, form, items, filing_date, report_date,
                    acceptance_dt, primary_doc, discovered_at, status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,'new')""",
                (
                    f["accession"], f["cik"], f["ticker"], f["form"], f.get("items"),
                    f.get("filing_date"), f.get("report_date"), f.get("acceptance_dt"),
                    f.get("primary_doc"), _now(),
                ),
            )
        return True

    def set_status(self, accession: str, status: str) -> None:
        with self.tx() as c:
            c.execute("UPDATE filings SET status = ? WHERE accession = ?", (status, accession))

    def pending(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM filings WHERE status = 'new' ORDER BY filing_date"))

    # -- briefs -----------------------------------------------------------

    def save_brief(self, accession: str, ticker: str, markdown: str,
                   model: str | None, extraction: dict | None) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT OR REPLACE INTO briefs
                   (accession, ticker, created_at, model, markdown, extraction)
                   VALUES (?,?,?,?,?,?)""",
                (accession, ticker, _now(), model, markdown,
                 json.dumps(extraction) if extraction else None),
            )

    def get_brief(self, accession: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM briefs WHERE accession = ?", (accession,)).fetchone()

    def recent_briefs(self, limit: int = 20) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM briefs ORDER BY created_at DESC LIMIT ?", (limit,)))

    # -- enrichments ------------------------------------------------------

    def queue_enrichment(self, accession: str, kind: str, detail: str = "") -> None:
        with self.tx() as c:
            c.execute(
                """INSERT OR IGNORE INTO enrichments
                   (accession, kind, status, created_at, updated_at, detail)
                   VALUES (?,?,'pending',?,?,?)""",
                (accession, kind, _now(), _now(), detail),
            )

    def resolve_enrichment(self, accession: str, kind: str, status: str,
                           body: str | None = None, detail: str = "") -> None:
        with self.tx() as c:
            c.execute(
                """UPDATE enrichments
                   SET status = ?, body = ?, detail = ?, updated_at = ?
                   WHERE accession = ? AND kind = ?""",
                (status, body, detail, _now(), accession, kind),
            )

    # -- theses -----------------------------------------------------------

    def save_thesis(self, ticker: str, accession: str, base_end: str, model: str | None,
                    thesis: Any, peers: list[str]) -> int:
        with self.tx() as c:
            cur = c.execute(
                """INSERT INTO theses
                   (ticker, accession, base_end, created_at, model, direction,
                    confidence, summary, peers, payload)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (ticker, accession, base_end, _now(), model, thesis.direction,
                 thesis.confidence, thesis.summary, json.dumps(peers),
                 thesis.model_dump_json()),
            )
            tid = cur.lastrowid
            for cl in thesis.claims:
                c.execute(
                    """INSERT INTO claims
                       (thesis_id, statement, check_kind, comparator, threshold,
                        horizon_quarters, disconfirms)
                       VALUES (?,?,?,?,?,?,?)""",
                    (tid, cl.statement, cl.check, cl.comparator, cl.threshold,
                     cl.horizon_quarters, cl.disconfirms),
                )
        return tid

    def theses(self, ticker: str | None = None, limit: int = 50) -> list[sqlite3.Row]:
        if ticker:
            return list(self.conn.execute(
                "SELECT * FROM theses WHERE ticker = ? ORDER BY created_at DESC LIMIT ?",
                (ticker.upper(), limit)))
        return list(self.conn.execute(
            "SELECT * FROM theses ORDER BY created_at DESC LIMIT ?", (limit,)))

    def get_thesis(self, thesis_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM theses WHERE id = ?", (thesis_id,)).fetchone()

    def claims_for(self, thesis_id: int) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM claims WHERE thesis_id = ? ORDER BY id", (thesis_id,)))

    def unsettled_claims(self) -> list[sqlite3.Row]:
        """Claims still worth re-checking. Excludes 'unresolvable' - those need a human."""
        return list(self.conn.execute(
            """SELECT c.*, t.ticker, t.base_end, t.created_at AS thesis_created_at
               FROM claims c JOIN theses t ON t.id = c.thesis_id
               WHERE c.status = 'pending' ORDER BY t.ticker"""))

    def settle_claim(self, claim_id: int, status: str,
                     actual: float | None, note: str) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE claims SET status=?, actual=?, note=?, resolved_at=? WHERE id=?",
                (status, actual, note, _now(), claim_id),
            )

    def claim_stats(self, ticker: str | None = None) -> dict[str, int]:
        q = ("SELECT c.status, COUNT(*) n FROM claims c JOIN theses t ON t.id=c.thesis_id"
             + (" WHERE t.ticker = ?" if ticker else "") + " GROUP BY c.status")
        rows = self.conn.execute(q, (ticker.upper(),) if ticker else ())
        return {r["status"]: r["n"] for r in rows}

    def pending_enrichments(self, kind: str | None = None) -> list[sqlite3.Row]:
        if kind:
            return list(self.conn.execute(
                "SELECT * FROM enrichments WHERE status = 'pending' AND kind = ?", (kind,)))
        return list(self.conn.execute("SELECT * FROM enrichments WHERE status = 'pending'"))
