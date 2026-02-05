"""
SEC Filing Auto-Scraper
========================
Discovers US microcap companies ($20-100M market cap) on NYSE, NASDAQ,
AMEX, OTCQX, and OTCQB, then downloads their latest 10-K/10-Q filings
from SEC EDGAR.

Universe source:
  FMP Company Screener (https://financialmodelingprep.com/stable/company-screener)
  - Requires FMP Starter plan ($22/mo) or higher
  - Hard server-side market cap filter: $20M-$100M
  - No proxies, no float estimates, no fallbacks

Rate limits:
  - FMP: 299 requests/min (Starter plan allows 300/min)
  - SEC EDGAR: 299 requests/min (SEC allows 10/sec = 600/min, we stay well under)

All EDGAR endpoints are free, no key needed (just User-Agent header).
"""

import os
import json
import time
import re
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "SECFilingAnalyzer/1.0 (contact@example.com)"
)

# Target exchanges — must be US, listed on these venues
TARGET_EXCHANGES = ["NYSE", "NASDAQ", "AMEX", "OTCQX", "OTCQB"]

# Hard market cap bounds (USD). No results outside this range.
MIN_MARKET_CAP = 20_000_000    # $20M
MAX_MARKET_CAP = 100_000_000   # $100M

# Filing types to download
TARGET_FORM_TYPES = ["10-K", "10-Q"]

# How far back to look for filings (days)
FILING_LOOKBACK_DAYS = 365

# ------------------------------------------------------------------
# RATE LIMITING — 299 requests per minute max for both FMP and EDGAR
# ------------------------------------------------------------------
RATE_LIMIT_PER_MIN = 299
_MIN_INTERVAL = 60.0 / RATE_LIMIT_PER_MIN  # ~0.2007s between requests

# Storage paths
DATA_DIR = "scraper_data"
UNIVERSE_FILE = "company_universe.json"
FILINGS_DIR = "filings"
FILINGS_INDEX = "filings_index.json"


# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    """
    Token-bucket rate limiter. Enforces max N requests per minute
    across all calls sharing the same limiter instance.
    """
    def __init__(self, max_per_minute: int = RATE_LIMIT_PER_MIN):
        self.min_interval = 60.0 / max_per_minute
        self._last_call = 0.0
        self._call_count = 0
        self._window_start = time.monotonic()

    def wait(self):
        """Block until it's safe to make the next request."""
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()
        self._call_count += 1

        # Log throughput every 100 calls
        if self._call_count % 100 == 0:
            window = time.monotonic() - self._window_start
            rate = self._call_count / (window / 60) if window > 0 else 0
            logger.info(f"Rate limiter: {self._call_count} calls, {rate:.0f}/min actual")

    @property
    def stats(self) -> Dict[str, Any]:
        window = time.monotonic() - self._window_start
        return {
            "total_calls": self._call_count,
            "elapsed_seconds": round(window, 1),
            "actual_rate_per_min": round(self._call_count / (window / 60), 1) if window > 0 else 0,
        }


# Global rate limiters — one for FMP, one for EDGAR
_fmp_limiter = RateLimiter(RATE_LIMIT_PER_MIN)
_edgar_limiter = RateLimiter(RATE_LIMIT_PER_MIN)


# ============================================================================
# HTTP HELPERS
# ============================================================================

def _get_session():
    """Create a requests session with proper headers."""
    import requests
    session = requests.Session()
    session.headers.update({
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    })
    return session


def _safe_request(session, url: str, params: dict = None,
                  limiter: RateLimiter = None, retries: int = 3) -> Optional[Any]:
    """
    Make a rate-limited HTTP request with retries.
    Returns parsed JSON, text, or None on failure.
    """
    import requests

    if limiter is None:
        limiter = _edgar_limiter

    for attempt in range(retries):
        try:
            limiter.wait()
            resp = session.get(url, params=params, timeout=30)

            if resp.status_code == 200:
                content_type = resp.headers.get("Content-Type", "")
                if "json" in content_type:
                    return resp.json()
                return resp.text

            elif resp.status_code == 429:
                wait = min(2 ** (attempt + 2), 30)
                logger.warning(f"Rate limited (429) on {url}, waiting {wait}s...")
                time.sleep(wait)
                continue

            elif resp.status_code == 404:
                logger.debug(f"Not found (404): {url}")
                return None

            elif resp.status_code in (402, 403):
                msg = f"HTTP {resp.status_code} — access denied for {url}"
                try:
                    body = resp.json()
                    msg += f" | {body}"
                except Exception:
                    pass
                logger.error(msg)
                raise PermissionError(msg)

            else:
                logger.warning(f"HTTP {resp.status_code} for {url}")
                if attempt < retries - 1:
                    time.sleep(2)

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on {url} (attempt {attempt + 1}/{retries})")
            time.sleep(2)
        except PermissionError:
            raise  # Don't retry auth failures
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            if attempt < retries - 1:
                time.sleep(2)

    return None


# ============================================================================
# STEP 1: BUILD COMPANY UNIVERSE (FMP SCREENER — PAID)
# ============================================================================

def build_universe_fmp(fmp_api_key: str,
                       progress_callback=None) -> List[Dict[str, Any]]:
    """
    Use FMP Company Screener to find US companies with $20-100M market cap
    on target exchanges. Requires FMP Starter plan ($22/mo) or higher.

    The screener does server-side filtering — every result is guaranteed
    to be in the $20-100M range. No float proxies, no guessing.

    Queries each exchange separately to ensure complete coverage,
    including OTC tiers.
    """
    session = _get_session()
    all_companies = []

    # FMP exchange codes for the screener endpoint
    # We query each separately to ensure nothing is missed
    fmp_exchanges = {
        "NYSE": ["NYSE"],
        "NASDAQ": ["NASDAQ"],
        "AMEX": ["AMEX"],
        "OTCQX": ["OTCQX"],
        "OTCQB": ["OTCQB"],
    }

    base_url = "https://financialmodelingprep.com/stable/company-screener"

    for our_exchange, fmp_codes in fmp_exchanges.items():
        for code in fmp_codes:
            if progress_callback:
                progress_callback(f"Screening {our_exchange} ({code})...")

            logger.info(f"FMP screener: {code} | $20M-$100M market cap...")

            params = {
                "marketCapMoreThan": MIN_MARKET_CAP,
                "marketCapLowerThan": MAX_MARKET_CAP,
                "exchange": code,
                "country": "US",
                "isActivelyTrading": "true",
                "isEtf": "false",
                "isFund": "false",
                "limit": 10000,
                "apikey": fmp_api_key,
            }

            try:
                data = _safe_request(session, base_url, params=params,
                                     limiter=_fmp_limiter)
            except PermissionError as e:
                raise RuntimeError(
                    f"FMP access denied. The Company Screener requires the Starter "
                    f"plan ($22/mo) or higher. Your current plan does not include "
                    f"this endpoint. Upgrade at: "
                    f"https://site.financialmodelingprep.com/developer/docs/pricing\n"
                    f"Error: {e}"
                )

            if data and isinstance(data, list):
                for company in data:
                    mkt_cap = company.get("marketCap", 0)

                    # Double-check server-side filter (belt and suspenders)
                    if not (MIN_MARKET_CAP <= mkt_cap <= MAX_MARKET_CAP):
                        continue

                    symbol = company.get("symbol", "")
                    name = company.get("companyName", "")

                    # Skip warrants, units, rights, etc.
                    if any(c in symbol for c in ["-", ".", "^", "+"]):
                        continue
                    if any(kw in name.lower() for kw in
                           ["warrant", "unit ", "rights", " right",
                            "acquisition corp", "blank check"]):
                        continue

                    all_companies.append({
                        "ticker": symbol,
                        "company_name": name,
                        "exchange": our_exchange,
                        "market_cap": mkt_cap,
                        "sector": company.get("sector", ""),
                        "industry": company.get("industry", ""),
                        "price": company.get("price", 0),
                        "volume": company.get("volume", 0),
                        "country": company.get("country", "US"),
                        "is_actively_trading": True,
                        "source": "fmp_screener",
                    })

                logger.info(f"  {code}: {len(data)} raw results → "
                           f"{sum(1 for c in all_companies if c['exchange'] == our_exchange)} kept")
            else:
                logger.warning(f"  {code}: no data returned (empty list or error)")

    # Deduplicate by ticker (some may appear in multiple exchange codes)
    seen = set()
    unique = []
    for c in all_companies:
        t = c["ticker"]
        if t and t not in seen:
            seen.add(t)
            unique.append(c)

    logger.info(f"FMP universe complete: {len(unique)} unique companies in $20-100M range")
    return unique


# ============================================================================
# CIK RESOLUTION
# ============================================================================

def resolve_ciks(companies: List[Dict], session=None,
                 progress_callback=None) -> List[Dict]:
    """
    Resolve SEC CIK numbers for each company using EDGAR's company
    tickers file. Companies without a CIK are dropped (can't search
    EDGAR for their filings).
    """
    if session is None:
        session = _get_session()

    if progress_callback:
        progress_callback("Downloading CIK mapping from SEC EDGAR...")

    logger.info("Downloading CIK-ticker mapping from EDGAR...")
    url = "https://www.sec.gov/files/company_tickers.json"
    data = _safe_request(session, url, limiter=_edgar_limiter)

    if not data:
        raise RuntimeError("Failed to download CIK mapping from SEC EDGAR. "
                          "Check your network connection.")

    # Build ticker -> CIK map (zero-padded to 10 digits)
    ticker_to_cik = {}
    for key, entry in data.items():
        ticker = entry.get("ticker", "").upper()
        cik = str(entry.get("cik_str", "")).zfill(10)
        if ticker:
            ticker_to_cik[ticker] = cik

    resolved = 0
    for company in companies:
        ticker = company["ticker"].upper()
        cik = ticker_to_cik.get(ticker)
        if cik:
            company["cik"] = cik
            resolved += 1

    with_cik = [c for c in companies if c.get("cik")]
    without_cik = [c for c in companies if not c.get("cik")]

    if without_cik:
        logger.warning(f"Dropped {len(without_cik)} companies without CIK: "
                      f"{[c['ticker'] for c in without_cik[:20]]}...")

    logger.info(f"CIK resolution: {resolved}/{len(companies)} resolved, "
               f"{len(with_cik)} companies ready for EDGAR lookup")

    return with_cik


# ============================================================================
# STEP 2: FIND FILINGS ON EDGAR
# ============================================================================

def get_company_filings(cik: str, session=None,
                        form_types: List[str] = None) -> List[Dict]:
    """
    Get recent filings for a company from SEC EDGAR Submissions API.
    Returns list of filing metadata dicts.
    """
    if session is None:
        session = _get_session()
    if form_types is None:
        form_types = TARGET_FORM_TYPES

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = _safe_request(session, url, limiter=_edgar_limiter)

    if not data:
        return []

    filings = []
    recent = data.get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    descriptions = recent.get("primaryDocDescription", [])

    cutoff = (datetime.now() - timedelta(days=FILING_LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    for i in range(len(forms)):
        form_type = forms[i] if i < len(forms) else ""
        filing_date = dates[i] if i < len(dates) else ""
        accession = accessions[i] if i < len(accessions) else ""
        primary_doc = primary_docs[i] if i < len(primary_docs) else ""
        description = descriptions[i] if i < len(descriptions) else ""

        if form_type in form_types and filing_date >= cutoff:
            accession_clean = accession.replace("-", "")
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik.lstrip('0')}/{accession_clean}/{primary_doc}"
            )

            filings.append({
                "form_type": form_type,
                "filing_date": filing_date,
                "accession_number": accession,
                "primary_document": primary_doc,
                "description": description,
                "filing_url": filing_url,
                "cik": cik,
            })

    return filings


# ============================================================================
# STEP 3: DOWNLOAD FILINGS
# ============================================================================

def download_filing(filing: Dict, output_dir: str,
                    session=None) -> Optional[str]:
    """Download a single filing document from EDGAR."""
    if session is None:
        session = _get_session()

    url = filing["filing_url"]
    ticker = filing.get("ticker", "UNKNOWN")
    form_type = filing["form_type"].replace("/", "-")
    date = filing["filing_date"]

    filename = f"{ticker}_{form_type}_{date}.html"
    filepath = os.path.join(output_dir, filename)

    # Skip if already downloaded
    if os.path.exists(filepath) and os.path.getsize(filepath) > 100:
        logger.debug(f"Already downloaded: {filename}")
        return filepath

    content = _safe_request(session, url, limiter=_edgar_limiter)
    if content and isinstance(content, str) and len(content) > 100:
        os.makedirs(output_dir, exist_ok=True)
        with open(filepath, "w", encoding="utf-8", errors="replace") as f:
            f.write(content)
        logger.debug(f"Downloaded: {filename} ({len(content)} chars)")
        return filepath
    else:
        logger.warning(f"Failed to download: {url}")
        return None


# ============================================================================
# TEXT EXTRACTION
# ============================================================================

def extract_text_from_html(html_content: str) -> str:
    """Extract readable text from an HTML filing."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)
    except ImportError:
        text = re.sub(r"<style[^>]*>.*?</style>", "", html_content,
                       flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<script[^>]*>.*?</script>", "", text,
                       flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


# ============================================================================
# ORCHESTRATOR
# ============================================================================

class SECScraper:
    """
    Orchestrates the full pipeline:
    1. Build company universe via FMP screener ($20-100M hard filter)
    2. Resolve CIKs from EDGAR
    3. Find filings on EDGAR for each company
    4. Download filing documents
    5. Track everything in a persistent index
    """

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.filings_dir = os.path.join(data_dir, FILINGS_DIR)
        self.universe_path = os.path.join(data_dir, UNIVERSE_FILE)
        self.index_path = os.path.join(data_dir, FILINGS_INDEX)
        self.universe: List[Dict] = []
        self.filings_index: List[Dict] = []

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.filings_dir, exist_ok=True)

        self._load_state()

    def _load_state(self):
        """Load saved universe and filings index."""
        if os.path.exists(self.universe_path):
            with open(self.universe_path) as f:
                self.universe = json.load(f)
            logger.info(f"Loaded universe: {len(self.universe)} companies")

        if os.path.exists(self.index_path):
            with open(self.index_path) as f:
                self.filings_index = json.load(f)
            logger.info(f"Loaded filings index: {len(self.filings_index)} filings")

    def _save_state(self):
        """Persist universe and index to disk."""
        with open(self.universe_path, "w") as f:
            json.dump(self.universe, f, indent=2)
        with open(self.index_path, "w") as f:
            json.dump(self.filings_index, f, indent=2)

    def step1_build_universe(self, fmp_api_key: str = None,
                              progress_callback=None) -> int:
        """
        Build company universe using FMP screener.
        Requires FMP API key (Starter plan or higher).
        Returns count of companies found.
        """
        if not fmp_api_key:
            raise RuntimeError(
                "FMP API key is required. The Company Screener endpoint "
                "requires the Starter plan ($22/mo) or higher. "
                "Sign up at: https://financialmodelingprep.com/register"
            )

        if progress_callback:
            progress_callback("Querying FMP screener for $20-100M US companies...")

        self.universe = build_universe_fmp(fmp_api_key, progress_callback)

        if not self.universe:
            raise RuntimeError(
                "FMP screener returned 0 companies. This usually means:\n"
                "1. Your plan doesn't include the screener endpoint (need Starter at $22/mo)\n"
                "2. Your API key is invalid\n"
                "3. FMP is temporarily down\n"
                "Check your key and plan at: https://site.financialmodelingprep.com/developer/docs/pricing"
            )

        # Resolve CIKs
        if progress_callback:
            progress_callback(f"Found {len(self.universe)} companies. Resolving CIKs...")

        session = _get_session()
        self.universe = resolve_ciks(self.universe, session, progress_callback)

        self._save_state()
        logger.info(f"Universe built: {len(self.universe)} companies with CIK")
        return len(self.universe)

    def step2_find_filings(self, max_companies: int = None,
                            progress_callback=None) -> int:
        """
        Query EDGAR for latest 10-K/10-Q for each company.
        Returns count of filings found.
        """
        if not self.universe:
            raise RuntimeError("No companies in universe. Run Step 1 first.")

        session = _get_session()
        companies = self.universe[:max_companies] if max_companies else self.universe
        total = len(companies)
        found = 0

        indexed_ciks = {f["cik"] for f in self.filings_index}

        for i, company in enumerate(companies):
            cik = company["cik"]
            ticker = company["ticker"]

            if cik in indexed_ciks:
                logger.debug(f"Skipping {ticker} (already indexed)")
                if progress_callback:
                    progress_callback(i + 1, total, ticker, 0)
                continue

            filings = get_company_filings(cik, session)

            for filing in filings:
                filing["ticker"] = ticker
                filing["company_name"] = company.get("company_name", "")
                filing["exchange"] = company.get("exchange", "")
                filing["market_cap"] = company.get("market_cap")
                filing["downloaded"] = False
                filing["local_path"] = None
                filing["analyzed"] = False
                self.filings_index.append(filing)
                found += 1

            if progress_callback:
                progress_callback(i + 1, total, ticker, len(filings))

            # Save every 50 companies
            if (i + 1) % 50 == 0:
                self._save_state()
                logger.info(f"Progress: {i + 1}/{total} companies, {found} filings found")

        self._save_state()
        logger.info(f"Filing search complete: {found} new filings across {total} companies")
        return found

    def step3_download_filings(self, max_downloads: int = None,
                                progress_callback=None) -> int:
        """
        Download filing documents that haven't been downloaded yet.
        Returns count of successful downloads.
        """
        session = _get_session()
        pending = [f for f in self.filings_index if not f.get("downloaded")]

        if max_downloads:
            pending = pending[:max_downloads]

        total = len(pending)
        downloaded = 0

        for i, filing in enumerate(pending):
            path = download_filing(filing, self.filings_dir, session)

            if path:
                filing["downloaded"] = True
                filing["local_path"] = path
                downloaded += 1

            if progress_callback:
                progress_callback(i + 1, total, filing.get("ticker", ""),
                                path is not None)

            # Save every 25 downloads
            if (i + 1) % 25 == 0:
                self._save_state()

        self._save_state()
        logger.info(f"Downloaded {downloaded}/{total} filings")
        return downloaded

    def get_stats(self) -> Dict[str, Any]:
        """Return current pipeline statistics."""
        total_companies = len(self.universe)
        total_filings = len(self.filings_index)
        downloaded = sum(1 for f in self.filings_index if f.get("downloaded"))
        analyzed = sum(1 for f in self.filings_index if f.get("analyzed"))

        by_exchange = {}
        for c in self.universe:
            ex = c.get("exchange", "Unknown")
            by_exchange[ex] = by_exchange.get(ex, 0) + 1

        by_form = {}
        for f in self.filings_index:
            ft = f.get("form_type", "Unknown")
            by_form[ft] = by_form.get(ft, 0) + 1

        # Market cap distribution
        cap_ranges = {"$20-40M": 0, "$40-60M": 0, "$60-80M": 0, "$80-100M": 0}
        for c in self.universe:
            mc = c.get("market_cap", 0) or 0
            if mc < 40_000_000:
                cap_ranges["$20-40M"] += 1
            elif mc < 60_000_000:
                cap_ranges["$40-60M"] += 1
            elif mc < 80_000_000:
                cap_ranges["$60-80M"] += 1
            else:
                cap_ranges["$80-100M"] += 1

        return {
            "total_companies": total_companies,
            "total_filings": total_filings,
            "filings_downloaded": downloaded,
            "filings_pending_download": total_filings - downloaded,
            "filings_analyzed": analyzed,
            "filings_pending_analysis": downloaded - analyzed,
            "companies_by_exchange": by_exchange,
            "filings_by_type": by_form,
            "market_cap_distribution": cap_ranges,
            "rate_limiter_fmp": _fmp_limiter.stats,
            "rate_limiter_edgar": _edgar_limiter.stats,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def get_unanalyzed_filings(self) -> List[Dict]:
        """Return filings that are downloaded but not yet analyzed."""
        return [
            f for f in self.filings_index
            if f.get("downloaded") and not f.get("analyzed")
        ]

    def mark_analyzed(self, accession_number: str, result_summary: dict = None):
        """Mark a filing as analyzed and store summary."""
        for f in self.filings_index:
            if f.get("accession_number") == accession_number:
                f["analyzed"] = True
                if result_summary:
                    f["score"] = result_summary.get("final_score")
                    f["risk_rating"] = result_summary.get("risk_rating")
                    f["red_flags"] = result_summary.get("red_flag_count", 0)
                    f["yellow_flags"] = result_summary.get("yellow_flag_count", 0)
                    f["green_flags"] = result_summary.get("green_flag_count", 0)
                    f["sentiment_trajectory"] = result_summary.get("sentiment_trajectory")
                    f["key_concerns"] = result_summary.get("key_concerns", [])
                    f["result_file"] = result_summary.get("result_file")
                break
        self._save_state()

    def get_results_ranked(self, sort_by: str = "score_asc",
                           risk_filter: str = "",
                           exchange_filter: str = "",
                           form_type_filter: str = "") -> List[Dict]:
        """Get analyzed filings ranked by score or other criteria."""
        results = [f for f in self.filings_index if f.get("analyzed")]

        if risk_filter:
            results = [f for f in results if f.get("risk_rating", "") == risk_filter]
        if exchange_filter:
            results = [f for f in results if f.get("exchange", "") == exchange_filter]
        if form_type_filter:
            results = [f for f in results if f.get("form_type", "") == form_type_filter]

        if sort_by == "score_asc":
            results.sort(key=lambda x: x.get("score", 100))
        elif sort_by == "score_desc":
            results.sort(key=lambda x: x.get("score", 0), reverse=True)
        elif sort_by == "red_flags":
            results.sort(key=lambda x: x.get("red_flags", 0), reverse=True)
        elif sort_by == "ticker":
            results.sort(key=lambda x: x.get("ticker", ""))
        elif sort_by == "date":
            results.sort(key=lambda x: x.get("filing_date", ""), reverse=True)

        return results

    def clear_universe(self):
        """Reset universe and filings index."""
        self.universe = []
        self.filings_index = []
        self._save_state()
        logger.info("Universe and filings index cleared")
