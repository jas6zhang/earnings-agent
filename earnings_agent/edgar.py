"""EDGAR client.

Everything here hits SEC endpoints that are free and public. Two rules the SEC
enforces and we must respect:
  1. Send a User-Agent that identifies you with a contact address.
  2. Stay under 10 requests/second.
Violating either gets you blocked, so the rate limiter is not optional.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable

import httpx
from bs4 import BeautifulSoup

SEC_WWW = "https://www.sec.gov"
SEC_DATA = "https://data.sec.gov"

# 8-K Item 2.02 is "Results of Operations and Financial Condition" - the
# earnings release. This is the signal we key on.
EARNINGS_ITEM = "2.02"

# How much of the full submission text to pull when hunting for exhibit types.
# The SEC-HEADER block carries a manifest of every document in the filing, so
# a small prefix is enough; we never download the whole (often multi-MB) file.
HEADER_PREFIX_BYTES = 65_536


class RateLimiter:
    """Simple thread-safe spacing limiter."""

    def __init__(self, per_second: float):
        self._interval = 1.0 / per_second if per_second > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
                now = time.monotonic()
            self._next = now + self._interval


@dataclass
class Exhibit:
    type: str
    sequence: str
    filename: str
    description: str


@dataclass
class Filing:
    accession: str      # dashed form, e.g. 0000320193-26-000018
    cik: str            # zero-padded 10 digits
    ticker: str
    form: str
    items: str
    filing_date: str
    report_date: str
    acceptance_dt: str
    primary_doc: str

    @property
    def accession_nodash(self) -> str:
        return self.accession.replace("-", "")

    @property
    def is_earnings(self) -> bool:
        return self.form.startswith("8-K") and EARNINGS_ITEM in (self.items or "")

    @property
    def dir_url(self) -> str:
        return f"{SEC_WWW}/Archives/edgar/data/{int(self.cik)}/{self.accession_nodash}"

    @property
    def index_url(self) -> str:
        return f"{self.dir_url}/{self.accession}-index.html"


class EdgarClient:
    def __init__(self, user_agent: str, requests_per_second: float = 8.0):
        self._limiter = RateLimiter(requests_per_second)
        self._http = httpx.Client(
            headers={
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=30.0,
            follow_redirects=True,
        )
        self._ticker_map: dict[str, str] | None = None
        # companyfacts is multi-MB per issuer (Apple's is ~3.8MB) and does not
        # change between filings in a single run, so fetch it at most once.
        self._facts_cache: dict[str, dict[str, Any]] = {}

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "EdgarClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _get(self, url: str, **kw: Any) -> httpx.Response:
        self._limiter.wait()
        r = self._http.get(url, **kw)
        r.raise_for_status()
        return r

    # -- ticker resolution ------------------------------------------------

    def ticker_to_cik(self, ticker: str) -> str:
        """Resolve a ticker to a zero-padded 10-digit CIK."""
        if self._ticker_map is None:
            data = self._get(f"{SEC_WWW}/files/company_tickers.json").json()
            self._ticker_map = {
                row["ticker"].upper(): str(row["cik_str"]).zfill(10)
                for row in data.values()
            }
        cik = self._ticker_map.get(ticker.upper())
        if cik is None:
            raise KeyError(f"Unknown ticker {ticker!r} - not in SEC company_tickers.json")
        return cik

    # -- filings ----------------------------------------------------------

    def recent_filings(self, ticker: str, forms: Iterable[str] = ("8-K",),
                       limit: int = 40) -> list[Filing]:
        """Most recent filings for a ticker, newest first."""
        cik = self.ticker_to_cik(ticker)
        data = self._get(f"{SEC_DATA}/submissions/CIK{cik}.json").json()
        recent = data["filings"]["recent"]
        wanted = tuple(forms)

        out: list[Filing] = []
        for i in range(len(recent["accessionNumber"])):
            form = recent["form"][i]
            if wanted and not form.startswith(wanted):
                continue
            out.append(Filing(
                accession=recent["accessionNumber"][i],
                cik=cik,
                ticker=ticker.upper(),
                form=form,
                items=recent.get("items", [""] * (i + 1))[i],
                filing_date=recent["filingDate"][i],
                report_date=recent.get("reportDate", [""] * (i + 1))[i],
                acceptance_dt=recent.get("acceptanceDateTime", [""] * (i + 1))[i],
                primary_doc=recent["primaryDocument"][i],
            ))
            if len(out) >= limit:
                break
        return out

    # -- exhibits ---------------------------------------------------------

    def exhibits(self, filing: Filing) -> list[Exhibit]:
        """List documents in a filing with their exhibit types.

        Primary path reads the SGML manifest in the SEC-HEADER via a ranged
        request - stable machine-readable format, a few KB. Falls back to
        scraping the index page if that comes back empty.
        """
        try:
            ex = self._exhibits_from_sgml(filing)
            if ex:
                return ex
        except httpx.HTTPError:
            pass
        return self._exhibits_from_index(filing)

    def _exhibits_from_sgml(self, filing: Filing) -> list[Exhibit]:
        url = f"{filing.dir_url}/{filing.accession}.txt"
        r = self._get(url, headers={"Range": f"bytes=0-{HEADER_PREFIX_BYTES - 1}"})
        text = r.text
        # Stop at the end of the header manifest if we can see it, so we do not
        # mistake inline document bodies for manifest entries.
        end = text.find("</SEC-HEADER>")
        if end != -1:
            text = text[:end]

        out: list[Exhibit] = []
        # Manifest entries are consecutive <TYPE>/<SEQUENCE>/<FILENAME>/<DESCRIPTION>
        # lines. DESCRIPTION is sometimes absent.
        pattern = re.compile(
            r"^<TYPE>(?P<type>[^\r\n]+)\s*"
            r"^<SEQUENCE>(?P<seq>[^\r\n]+)\s*"
            r"^<FILENAME>(?P<fn>[^\r\n]+)\s*"
            r"(?:^<DESCRIPTION>(?P<desc>[^\r\n]*)\s*)?",
            re.M,
        )
        for m in pattern.finditer(text):
            out.append(Exhibit(
                type=m.group("type").strip(),
                sequence=m.group("seq").strip(),
                filename=m.group("fn").strip(),
                description=(m.group("desc") or "").strip(),
            ))
        return out

    def _exhibits_from_index(self, filing: Filing) -> list[Exhibit]:
        html = self._get(filing.index_url).text
        soup = BeautifulSoup(html, "lxml")
        out: list[Exhibit] = []
        for tr in soup.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if len(cells) >= 4 and cells[0].isdigit():
                link = tr.find("a")
                fn = link.get_text(strip=True) if link else cells[2]
                out.append(Exhibit(
                    type=cells[3], sequence=cells[0],
                    filename=fn.split()[0] if fn else "", description=cells[1],
                ))
        return out

    def press_release(self, filing: Filing) -> tuple[str, str] | None:
        """Find and fetch the earnings press release exhibit as plain text.

        Returns (url, text) or None when the filing carries no EX-99 exhibit.
        """
        ex = self.exhibits(filing)
        # EX-99.1 is the overwhelming convention for the earnings release, but
        # some filers use EX-99 or EX-99.2, so widen the net in priority order.
        def rank(e: Exhibit) -> int:
            t = e.type.upper().replace(" ", "")
            if t == "EX-99.1":
                return 0
            if t == "EX-99":
                return 1
            if t.startswith("EX-99"):
                return 2
            return 99

        candidates = sorted([e for e in ex if rank(e) < 99], key=rank)
        if not candidates:
            return None
        target = candidates[0]
        url = f"{filing.dir_url}/{target.filename}"
        text = html_to_text(self._get(url).text)
        return url, strip_edgar_wrapper(text, target)

    def company_facts(self, cik: str) -> dict[str, Any]:
        if cik not in self._facts_cache:
            self._facts_cache[cik] = self._get(
                f"{SEC_DATA}/api/xbrl/companyfacts/CIK{cik}.json"
            ).json()
        return self._facts_cache[cik]


def strip_edgar_wrapper(text: str, exhibit: Exhibit) -> str:
    """Drop the boilerplate banner EDGAR prepends to archived exhibit documents.

    Served exhibits open with a few lines echoing the exhibit type, sequence
    number, filename and the literal word "Document" before the real content.
    Feeding that to the model wastes tokens and invites confusion about what
    the document is.
    """
    noise = {
        exhibit.type.upper(),
        exhibit.filename.lower(),
        exhibit.sequence,
        "DOCUMENT",
    }
    lines = text.split("\n")
    start = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        if s.upper() in noise or s.lower() in noise or re.fullmatch(r"EX-[\d.]+", s.upper()):
            start = i + 1
            continue
        break
    return "\n".join(lines[start:]).strip()


def html_to_text(html: str) -> str:
    """Flatten filing HTML to text, keeping tables legible.

    Press releases put the financial statements in <table>, and a naive
    get_text() smears every cell into one run-on line. Rendering rows as
    pipe-delimited keeps the columns aligned to their labels.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()

    for table in soup.find_all("table"):
        lines = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c not in ("", "$", ")", "(")]
            if cells:
                lines.append(" | ".join(cells))
        table.replace_with("\n" + "\n".join(lines) + "\n")

    text = soup.get_text("\n")
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()
