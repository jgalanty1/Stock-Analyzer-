"""
SEC Filing Auto-Scraper
========================
Discovers US microcap companies ($20-100M market cap) on NYSE, NASDAQ,
AMEX, OTCQX, and OTCQB, then downloads their latest 10-K/10-Q filings
from SEC EDGAR.

Universe source:
  FMP Company Screener (tries stable endpoint, falls back to legacy v3)
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
_MIN_INTERVAL = 60.0 / RATE_LIMIT_PER_MIN

# Storage paths
DATA_DIR = "scraper_data"
UNIVERSE_FILE = "company_universe.json"
FILINGS_DIR = "filings"
FILINGS_INDEX = "filings_index.json"


# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    def __init__(self, max_per_minute: int = RATE_LIMIT_PER_MIN):
        self.min_interval = 60.0 / max_per_minute
        self._last_call = 0.0
        self._call_count = 0
        self._window_start = time.monotonic()

    def wait(self):
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()
        self._call_count += 1
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


_fmp_limiter = RateLimiter(RATE_LIMIT_PER_MIN)
_edgar_limiter = RateLimiter(RATE_LIMIT_PER_MIN)


# ============================================================================
# HTTP HELPERS
# ============================================================================

def _get_session():
    import requests
    session = requests.Session()
    session.headers.update({
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    })
    return session


def _safe_request(session, url: str, params: dict = None,
                  limiter: RateLimiter = None, retries: int = 3) -> Optional[Any]:
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
                try:
                    return resp.json()
                except Exception:
                    return resp.text

            elif resp.status_code == 429:
                wait = min(2 ** (attempt + 2), 30)
                logger.warning(f"Rate limited (429) on {url}, waiting {wait}s...")
                time.sleep(wait)
                continue

            elif resp.status_code == 404:
                logger.debug(f"Not found (404): {url}")
                return None

            elif resp.status_code in (401, 402, 403):
                body_text = ""
                try:
                    body = resp.json()
                    body_text = json.dumps(body)
                except Exception:
                    body_text = resp.text[:500]
                msg = f"HTTP {resp.status_code} from {url}\nResponse: {body_text}"
                logger.error(msg)
                raise PermissionError(msg)

            else:
                body_text = ""
                try:
                    body_text = resp.text[:500]
                except Exception:
                    pass
                logger.warning(f"HTTP {resp.status_code} for {url}: {body_text}")
                if attempt < retries - 1:
                    time.sleep(2)

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on {url} (attempt {attempt + 1}/{retries})")
            time.sleep(2)
        except PermissionError:
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            if attempt < retries - 1:
                time.sleep(2)

    return None


# ============================================================================
# FMP API DIAGNOSTIC TEST
# ============================================================================

def test_fmp_api(fmp_api_key: str) -> Dict[str, Any]:
    """
    Run diagnostics against FMP to verify key validity and screener access.
    Returns detailed results for display to user.
    """
    import requests

    results = {
        "key_provided": bool(fmp_api_key),
        "key_length": len(fmp_api_key) if fmp_api_key else 0,
        "tests": [],
        "working_endpoint": None,
        "sample_count": 0,
        "error": None,
    }

    if not fmp_api_key:
        results["error"] = "No API key provided"
        return results

    session = _get_session()

    # Test 1: Stable screener endpoint
    test1 = {"name": "Stable endpoint (/stable/company-screener)", "status": "pending"}
    try:
        url = "https://financialmodelingprep.com/stable/company-screener"
        params = {
            "marketCapMoreThan": MIN_MARKET_CAP,
            "marketCapLowerThan": MAX_MARKET_CAP,
            "country": "US",
            "isActivelyTrading": "true",
            "isEtf": "false",
            "isFund": "false",
            "limit": 5,
            "apikey": fmp_api_key,
        }
        resp = session.get(url, params=params, timeout=15)
        test1["http_status"] = resp.status_code
        test1["response_preview"] = resp.text[:800]

        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                test1["status"] = "PASS"
                test1["count"] = len(data)
                test1["sample"] = {k: data[0].get(k) for k in
                    ["symbol", "companyName", "marketCap", "exchangeShortName", "exchange"]
                    if k in data[0]} if data else None
                results["working_endpoint"] = "stable"
                results["sample_count"] = len(data)
            elif isinstance(data, list) and len(data) == 0:
                test1["status"] = "WARN"
                test1["detail"] = "Returned empty list — endpoint works but no matching data"
            elif isinstance(data, dict):
                test1["status"] = "FAIL"
                test1["detail"] = f"API error: {data}"
            else:
                test1["status"] = "WARN"
                test1["detail"] = f"Unexpected type: {type(data).__name__}"
        elif resp.status_code in (401, 402, 403):
            test1["status"] = "FAIL"
            test1["detail"] = f"Access denied (HTTP {resp.status_code})"
        else:
            test1["status"] = "FAIL"
            test1["detail"] = f"HTTP {resp.status_code}"
    except Exception as e:
        test1["status"] = "ERROR"
        test1["detail"] = str(e)
    results["tests"].append(test1)

    # Test 2: Legacy v3 screener endpoint
    test2 = {"name": "Legacy endpoint (/api/v3/stock-screener)", "status": "pending"}
    try:
        url = "https://financialmodelingprep.com/api/v3/stock-screener"
        params = {
            "marketCapMoreThan": MIN_MARKET_CAP,
            "marketCapLowerThan": MAX_MARKET_CAP,
            "country": "US",
            "isActivelyTrading": "true",
            "isEtf": "false",
            "limit": 5,
            "apikey": fmp_api_key,
        }
        resp = session.get(url, params=params, timeout=15)
        test2["http_status"] = resp.status_code
        test2["response_preview"] = resp.text[:800]

        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                test2["status"] = "PASS"
                test2["count"] = len(data)
                test2["sample"] = {k: data[0].get(k) for k in
                    ["symbol", "companyName", "marketCap", "exchangeShortName", "exchange"]
                    if k in data[0]} if data else None
                if not results["working_endpoint"]:
                    results["working_endpoint"] = "legacy"
                    results["sample_count"] = len(data)
            elif isinstance(data, list) and len(data) == 0:
                test2["status"] = "WARN"
                test2["detail"] = "Returned empty list"
            elif isinstance(data, dict):
                test2["status"] = "FAIL"
                test2["detail"] = f"API error: {data}"
            else:
                test2["status"] = "WARN"
        elif resp.status_code in (401, 402, 403):
            test2["status"] = "FAIL"
            test2["detail"] = f"Access denied (HTTP {resp.status_code})"
        else:
            test2["status"] = "FAIL"
            test2["detail"] = f"HTTP {resp.status_code}"
    except Exception as e:
        test2["status"] = "ERROR"
        test2["detail"] = str(e)
    results["tests"].append(test2)

    # Test 3: Basic profile (to verify key validity independent of plan)
    test3 = {"name": "Profile check (key validity)", "status": "pending"}
    try:
        url = "https://financialmodelingprep.com/stable/profile"
        params = {"symbol": "AAPL", "apikey": fmp_api_key}
        resp = session.get(url, params=params, timeout=15)
        test3["http_status"] = resp.status_code
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                test3["status"] = "PASS"
                test3["detail"] = "API key is VALID"
            elif isinstance(data, dict) and ("Error" in data or "error" in data):
                test3["status"] = "FAIL"
                test3["detail"] = f"Key invalid: {data}"
            else:
                test3["status"] = "PASS"
                test3["detail"] = "Key appears valid"
        elif resp.status_code in (401, 403):
            test3["status"] = "FAIL"
            test3["detail"] = "API key is INVALID"
        else:
            test3["status"] = "WARN"
            test3["detail"] = f"HTTP {resp.status_code}"
    except Exception as e:
        test3["status"] = "ERROR"
        test3["detail"] = str(e)
    results["tests"].append(test3)

    # Summary
    if not results["working_endpoint"]:
        key_valid = any(t["status"] == "PASS" for t in results["tests"] if "profile" in t["name"].lower())
        if key_valid:
            results["error"] = (
                "Your API key is valid, but the screener endpoint is NOT accessible. "
                "Your FMP plan does not include the stock screener. "
                "You need the Starter plan ($22/mo) or higher. "
                "Upgrade: https://site.financialmodelingprep.com/developer/docs/pricing"
            )
        else:
            results["error"] = (
                "Could not connect to FMP. Your API key may be invalid or there's "
                "a network issue. Check: https://financialmodelingprep.com/dashboard"
            )

    return results


# ============================================================================
# STEP 1: BUILD COMPANY UNIVERSE (FMP SCREENER)
# ============================================================================

def build_universe_fmp(fmp_api_key: str,
                       progress_callback=None,
                       min_market_cap: int = None,
                       max_market_cap: int = None) -> List[Dict[str, Any]]:
    """
    Use FMP Company Screener to find US companies within a market cap range.

    Strategy:
    1. Try the stable endpoint first (/stable/company-screener)
    2. If that fails, try the legacy v3 endpoint (/api/v3/stock-screener)
    3. Do NOT filter by exchange at the API level — FMP may not support
       OTC exchange codes. Filter by country=US only, keep all exchanges.
    4. The screener does server-side market cap filtering.
    """
    min_cap = int(min_market_cap) if min_market_cap else MIN_MARKET_CAP
    max_cap = int(max_market_cap) if max_market_cap else MAX_MARKET_CAP
    session = _get_session()
    all_companies = []
    working_endpoint = None

    endpoints = [
        ("stable", "https://financialmodelingprep.com/stable/company-screener"),
        ("legacy", "https://financialmodelingprep.com/api/v3/stock-screener"),
    ]

    for ep_name, ep_url in endpoints:
        if progress_callback:
            progress_callback(f"Trying FMP {ep_name} screener...")

        logger.info(f"Trying FMP {ep_name}: {ep_url}")

        params = {
            "marketCapMoreThan": min_cap,
            "marketCapLowerThan": max_cap,
            "country": "US",
            "isActivelyTrading": "true",
            "isEtf": "false",
            "limit": 10000,
            "apikey": fmp_api_key,
        }
        if ep_name == "stable":
            params["isFund"] = "false"

        try:
            data = _safe_request(session, ep_url, params=params, limiter=_fmp_limiter)
        except PermissionError as e:
            logger.warning(f"FMP {ep_name} access denied: {e}")
            if progress_callback:
                progress_callback(f"FMP {ep_name}: access denied, trying next...")
            continue

        if data and isinstance(data, list) and len(data) > 0:
            working_endpoint = ep_name
            logger.info(f"FMP {ep_name}: {len(data)} results")
            if progress_callback:
                progress_callback(f"FMP {ep_name}: {len(data)} raw results")

            for company in data:
                mkt_cap = company.get("marketCap", 0) or 0
                if not (min_cap <= mkt_cap <= max_cap):
                    continue

                symbol = company.get("symbol", "")
                name = company.get("companyName", "")
                exchange = company.get("exchangeShortName",
                           company.get("exchange", ""))

                if any(c in symbol for c in ["-", ".", "^", "+"]):
                    continue
                if any(kw in name.lower() for kw in
                       ["warrant", "unit ", "rights", " right",
                        "acquisition corp", "blank check"]):
                    continue

                all_companies.append({
                    "ticker": symbol,
                    "company_name": name,
                    "exchange": exchange,
                    "market_cap": mkt_cap,
                    "sector": company.get("sector", ""),
                    "industry": company.get("industry", ""),
                    "price": company.get("price", 0),
                    "volume": company.get("volume", 0),
                    "country": company.get("country", "US"),
                    "is_actively_trading": True,
                    "source": f"fmp_{ep_name}",
                })
            break
        elif data and isinstance(data, dict):
            err_msg = data.get("Error Message",
                      data.get("message",
                      data.get("error", str(data))))
            logger.warning(f"FMP {ep_name} error: {err_msg}")
            if progress_callback:
                progress_callback(f"FMP {ep_name}: {err_msg}")
            continue
        else:
            logger.warning(f"FMP {ep_name}: empty response")
            if progress_callback:
                progress_callback(f"FMP {ep_name}: empty, trying next...")
            continue

    if not working_endpoint:
        raise RuntimeError(
            "BOTH FMP screener endpoints failed.\n\n"
            "Possible causes:\n"
            "1. Invalid API key — check at https://financialmodelingprep.com/dashboard\n"
            "2. Plan doesn't include screener — need Starter ($22/mo)+\n"
            "3. FMP is temporarily down\n\n"
            "Use 'Test API' button for exact error details.\n"
            "Upgrade: https://site.financialmodelingprep.com/developer/docs/pricing"
        )

    # Deduplicate
    seen = set()
    unique = []
    for c in all_companies:
        t = c["ticker"]
        if t and t not in seen:
            seen.add(t)
            unique.append(c)

    # Log exchange distribution
    exc = {}
    for c in unique:
        ex = c.get("exchange", "Unknown")
        exc[ex] = exc.get(ex, 0) + 1
    logger.info(f"FMP universe: {len(unique)} companies, exchanges: {exc}")

    if progress_callback:
        min_m = int(min_cap / 1_000_000)
        max_m = int(max_cap / 1_000_000)
        progress_callback(
            f"Found {len(unique)} US microcaps (${min_m}-${max_m}M). "
            f"Exchanges: {', '.join(f'{k}:{v}' for k,v in sorted(exc.items()))}"
        )

    return unique


# ============================================================================
# CIK RESOLUTION
# ============================================================================

def resolve_ciks(companies: List[Dict], session=None,
                 progress_callback=None) -> List[Dict]:
    if session is None:
        session = _get_session()

    if progress_callback:
        progress_callback("Downloading CIK mapping from SEC EDGAR...")

    url = "https://www.sec.gov/files/company_tickers.json"
    data = _safe_request(session, url, limiter=_edgar_limiter)

    if not data:
        raise RuntimeError("Failed to download CIK mapping from SEC EDGAR.")

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
        logger.warning(f"Dropped {len(without_cik)} without CIK: "
                      f"{[c['ticker'] for c in without_cik[:20]]}")

    logger.info(f"CIK: {resolved}/{len(companies)} resolved, {len(with_cik)} ready")
    return with_cik


# ============================================================================
# STEP 2: FIND FILINGS ON EDGAR
# ============================================================================

def get_company_filings(cik: str, session=None,
                        form_types: List[str] = None) -> List[Dict]:
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
    if session is None:
        session = _get_session()

    url = filing["filing_url"]
    ticker = filing.get("ticker", "UNKNOWN")
    form_type = filing["form_type"].replace("/", "-")
    date = filing["filing_date"]

    filename = f"{ticker}_{form_type}_{date}.html"
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath) and os.path.getsize(filepath) > 100:
        return filepath

    content = _safe_request(session, url, limiter=_edgar_limiter)
    if content and isinstance(content, str) and len(content) > 100:
        os.makedirs(output_dir, exist_ok=True)
        with open(filepath, "w", encoding="utf-8", errors="replace") as f:
            f.write(content)
        return filepath
    return None


# ============================================================================
# TEXT EXTRACTION
# ============================================================================

def extract_text_from_html(html_content: str) -> str:
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
        if os.path.exists(self.universe_path):
            with open(self.universe_path) as f:
                self.universe = json.load(f)
            logger.info(f"Loaded universe: {len(self.universe)} companies")
        if os.path.exists(self.index_path):
            with open(self.index_path) as f:
                self.filings_index = json.load(f)
            logger.info(f"Loaded filings index: {len(self.filings_index)} filings")

    def _save_state(self):
        with open(self.universe_path, "w") as f:
            json.dump(self.universe, f, indent=2)
        with open(self.index_path, "w") as f:
            json.dump(self.filings_index, f, indent=2)

    def step1_build_universe(self, fmp_api_key: str = None,
                              progress_callback=None,
                              min_market_cap: int = None,
                              max_market_cap: int = None,
                              cancel_event=None) -> int:
        if not fmp_api_key:
            raise RuntimeError(
                "FMP API key is required. The Company Screener endpoint "
                "requires the Starter plan ($22/mo) or higher. "
                "Sign up at: https://financialmodelingprep.com/register"
            )

        if progress_callback:
            progress_callback("Querying FMP screener...")

        # Clear old filings — universe is changing so old filings are stale
        old_count = len(self.filings_index)
        if old_count > 0:
            if progress_callback:
                progress_callback(f"Clearing {old_count} filings from previous universe...")
            self.filings_index = []
            logger.info(f"Cleared {old_count} stale filings (universe rebuild)")

        self.universe = build_universe_fmp(fmp_api_key, progress_callback,
                                          min_market_cap=min_market_cap,
                                          max_market_cap=max_market_cap)

        if not self.universe:
            raise RuntimeError(
                "FMP screener returned 0 companies.\n"
                "1. Need Starter plan ($22/mo) — screener not on free plan\n"
                "2. API key may be invalid\n"
                "3. FMP may be down\n\n"
                "Use 'Test API' button for diagnostics."
            )

        if progress_callback:
            progress_callback(f"Found {len(self.universe)} companies. Resolving CIKs...")

        session = _get_session()
        self.universe = resolve_ciks(self.universe, session, progress_callback)
        self._save_state()
        logger.info(f"Universe built: {len(self.universe)} companies with CIK")
        return len(self.universe)

    def step2_find_filings(self, max_companies: int = None,
                            progress_callback=None,
                            cancel_event=None) -> int:
        if not self.universe:
            raise RuntimeError("No companies in universe. Run Step 1 first.")

        session = _get_session()
        companies = self.universe[:max_companies] if max_companies else self.universe
        total = len(companies)
        found = 0
        indexed_ciks = {f["cik"] for f in self.filings_index}

        for i, company in enumerate(companies):
            if cancel_event and cancel_event.is_set():
                logger.info(f"step2_find_filings cancelled at {i}/{total}")
                self._save_state()
                raise InterruptedError(f"Cancelled after {i}/{total} companies ({found} filings found)")

            cik = company["cik"]
            ticker = company["ticker"]

            if cik in indexed_ciks:
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
            if (i + 1) % 50 == 0:
                self._save_state()

        self._save_state()
        logger.info(f"Filing search: {found} new filings across {total} companies")
        return found

    def step3_download_filings(self, max_downloads: int = None,
                                progress_callback=None,
                                cancel_event=None) -> int:
        session = _get_session()
        pending = [f for f in self.filings_index if not f.get("downloaded")]
        if max_downloads:
            pending = pending[:max_downloads]

        total = len(pending)
        downloaded = 0

        for i, filing in enumerate(pending):
            if cancel_event and cancel_event.is_set():
                logger.info(f"step3_download_filings cancelled at {i}/{total}")
                self._save_state()
                raise InterruptedError(f"Cancelled after {i}/{total} downloads ({downloaded} succeeded)")

            path = download_filing(filing, self.filings_dir, session)
            if path:
                filing["downloaded"] = True
                filing["local_path"] = path
                downloaded += 1
            if progress_callback:
                progress_callback(i + 1, total, filing.get("ticker", ""),
                                path is not None)
            if (i + 1) % 25 == 0:
                self._save_state()

        self._save_state()
        logger.info(f"Downloaded {downloaded}/{total} filings")
        return downloaded

    def get_stats(self) -> Dict[str, Any]:
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
        return [f for f in self.filings_index
                if f.get("downloaded") and not f.get("analyzed")]

    def mark_analyzed(self, accession_number: str, result_summary: dict = None):
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
                           form_type_filter: str = "",
                           tier_filter: str = "") -> List[Dict]:
        """Return all analyzed filings with unified tier info."""
        results = []
        for f in self.filings_index:
            # Determine analysis tier(s) for this filing
            has_batch = f.get("analyzed", False)
            has_t1 = f.get("llm_analyzed", False)
            has_t2 = f.get("tier2_analyzed", False)

            if not (has_batch or has_t1 or has_t2):
                continue

            # Determine best tier and unified score
            if has_t2:
                tier = "tier2"
                score = f.get("final_gem_score") if f.get("final_gem_score") is not None else f.get("tier1_score", 50)
            elif has_t1:
                tier = "tier1"
                score = f.get("tier1_score", 50)
            else:
                tier = "batch"
                score = f.get("score", 50)

            entry = {
                "ticker": f.get("ticker", ""),
                "company_name": f.get("company_name", ""),
                "form_type": f.get("form_type", ""),
                "filing_date": f.get("filing_date", ""),
                "exchange": f.get("exchange", ""),
                "accession_number": f.get("accession_number", ""),
                "tier": tier,
                "score": score,
                # Batch analysis fields
                "risk_rating": f.get("risk_rating", ""),
                "red_flags": f.get("red_flag_count", 0) or 0,
                "yellow_flags": f.get("yellow_flag_count", 0) or 0,
                "green_flags": f.get("green_flag_count", 0) or 0,
                "sentiment_trajectory": f.get("sentiment_trajectory", ""),
                "key_concerns": f.get("key_concerns", []),
                "result_file": f.get("result_file", ""),
                # LLM Tier 1 fields
                "tier1_score": f.get("tier1_score"),
                "gem_potential": f.get("gem_potential", ""),
                "tier1_result_file": f.get("tier1_result_file", ""),
                # LLM Tier 2 fields
                "tier2_analyzed": has_t2,
                "final_gem_score": f.get("final_gem_score"),
                "conviction": f.get("conviction", ""),
                "recommendation": f.get("recommendation", ""),
                "tier2_result_file": f.get("tier2_result_file", ""),
                # Diff fields
                "has_prior_diff": f.get("has_prior_diff", False),
                "diff_signal": f.get("diff_signal", ""),
            }
            results.append(entry)

        # Filters
        if risk_filter:
            results = [r for r in results if r.get("risk_rating", "") == risk_filter]
        if exchange_filter:
            results = [r for r in results if r.get("exchange", "") == exchange_filter]
        if form_type_filter:
            results = [r for r in results if r.get("form_type", "") == form_type_filter]
        if tier_filter:
            results = [r for r in results if r.get("tier", "") == tier_filter]

        # Sort
        if sort_by == "score_asc":
            results.sort(key=lambda x: x.get("score") if x.get("score") is not None else 100)
        elif sort_by == "score_desc":
            results.sort(key=lambda x: x.get("score") if x.get("score") is not None else 0, reverse=True)
        elif sort_by == "red_flags":
            results.sort(key=lambda x: x.get("red_flags", 0), reverse=True)
        elif sort_by == "ticker":
            results.sort(key=lambda x: x.get("ticker", ""))
        elif sort_by == "date":
            results.sort(key=lambda x: x.get("filing_date", ""), reverse=True)
        elif sort_by == "gem_score_desc":
            results.sort(key=lambda x: x.get("final_gem_score") or x.get("tier1_score") or 0, reverse=True)
        return results

    def _normalize_score(self, score):
        """Normalize any score to 0-100 scale. Detects 1-10 scale and multiplies."""
        if score is None:
            return None
        try:
            s = float(score)
        except (TypeError, ValueError):
            return None
        if s <= 10:
            s = s * 10
        return round(min(100, max(0, s)), 1)

    def _get_filing_score(self, f):
        """Get the best available score for a filing, normalized to 0-100."""
        # Prefer tier2 > tier1 > batch
        if f.get('final_gem_score') is not None:
            return self._normalize_score(f['final_gem_score']), 'tier2'
        if f.get('tier1_score') is not None:
            return self._normalize_score(f['tier1_score']), 'tier1'
        if f.get('score') is not None:
            return self._normalize_score(f['score']), 'batch'
        return None, None

    def get_stocks_ranked(self, sort_by: str = "score_desc",
                          exchange_filter: str = "",
                          min_score: float = 0) -> List[Dict]:
        """
        Aggregate filings by ticker into one entry per stock.
        Computes recency-weighted composite score across all filings.
        """
        from collections import defaultdict

        # Group analyzed filings by ticker
        ticker_filings = defaultdict(list)
        for f in self.filings_index:
            score, tier = self._get_filing_score(f)
            if score is None:
                continue
            ticker_filings[f.get('ticker', '')].append({
                'filing_date': f.get('filing_date', ''),
                'form_type': f.get('form_type', ''),
                'score': score,
                'tier': tier,
                'accession_number': f.get('accession_number', ''),
                'final_gem_score': self._normalize_score(f.get('final_gem_score')),
                'tier1_score': self._normalize_score(f.get('tier1_score')),
                'batch_score': self._normalize_score(f.get('score')),
                'conviction': f.get('conviction', ''),
                'recommendation': f.get('recommendation', ''),
                'gem_potential': f.get('gem_potential', ''),
                'diff_signal': f.get('diff_signal', ''),
                'has_prior_diff': f.get('has_prior_diff', False),
                'insider_signal': f.get('insider_signal', ''),
                'accumulation_score': f.get('accumulation_score'),
                'inflection_signal': f.get('inflection_signal', ''),
                'inflection_score': f.get('inflection_score'),
                'risk_rating': f.get('risk_rating', ''),
                'tier2_result_file': f.get('tier2_result_file', ''),
                'tier1_result_file': f.get('tier1_result_file', ''),
                'result_file': f.get('result_file', ''),
                'company_name': f.get('company_name', ''),
                'exchange': f.get('exchange', ''),
            })

        stocks = []
        for ticker, filings in ticker_filings.items():
            if not filings:
                continue

            # Sort by date descending (newest first)
            filings.sort(key=lambda x: x['filing_date'], reverse=True)
            latest = filings[0]

            # Company info
            company = next((c for c in self.universe
                            if c.get('ticker', '').upper() == ticker.upper()), None)
            company_name = latest.get('company_name') or (company or {}).get('company_name', '')
            exchange = latest.get('exchange') or (company or {}).get('exchange', '')

            # ---- Recency-weighted composite score ----
            # Exponential decay: most recent = weight 1.0, each prior filing halves
            weighted_sum = 0
            weight_total = 0
            for i, fl in enumerate(filings):
                weight = 0.5 ** i  # 1.0, 0.5, 0.25, 0.125, ...
                weighted_sum += fl['score'] * weight
                weight_total += weight
            weighted_score = round(weighted_sum / weight_total, 1) if weight_total > 0 else 50

            # ---- Best tier achieved ----
            tiers = [fl['tier'] for fl in filings]
            if 'tier2' in tiers:
                best_tier = 'tier2'
            elif 'tier1' in tiers:
                best_tier = 'tier1'
            else:
                best_tier = 'batch'

            # ---- Trajectory from score trend ----
            trajectory = 'stable'
            if len(filings) >= 2:
                oldest_score = filings[-1]['score']
                newest_score = filings[0]['score']
                change = newest_score - oldest_score
                if change >= 15:
                    trajectory = 'strongly_improving'
                elif change >= 5:
                    trajectory = 'improving'
                elif change <= -15:
                    trajectory = 'strongly_deteriorating'
                elif change <= -5:
                    trajectory = 'deteriorating'

            # ---- Latest filing's qualitative fields ----
            conviction = ''
            recommendation = ''
            gem_potential = ''
            diff_signal = ''
            insider_signal = ''
            accumulation_score = None
            inflection_signal = ''
            inflection_score = None
            one_liner = ''
            for fl in filings:
                if fl.get('conviction') and not conviction:
                    conviction = fl['conviction']
                if fl.get('recommendation') and not recommendation:
                    recommendation = fl['recommendation']
                if fl.get('gem_potential') and not gem_potential:
                    gem_potential = fl['gem_potential']
                if fl.get('diff_signal') and not diff_signal:
                    diff_signal = fl['diff_signal']
                if fl.get('insider_signal') and not insider_signal:
                    insider_signal = fl['insider_signal']
                if fl.get('accumulation_score') is not None and accumulation_score is None:
                    accumulation_score = fl['accumulation_score']
                if fl.get('inflection_signal') and not inflection_signal:
                    inflection_signal = fl['inflection_signal']
                if fl.get('inflection_score') is not None and inflection_score is None:
                    inflection_score = fl['inflection_score']

            entry = {
                'ticker': ticker,
                'company_name': company_name,
                'exchange': exchange,
                'market_cap': (company or {}).get('market_cap'),
                'sector': (company or {}).get('sector', ''),
                'score': weighted_score,
                'best_tier': best_tier,
                'filing_count': len(filings),
                'latest_filing_date': filings[0]['filing_date'],
                'latest_form_type': filings[0]['form_type'],
                'latest_score': filings[0]['score'],
                'trajectory': trajectory,
                'conviction': conviction,
                'recommendation': recommendation,
                'gem_potential': gem_potential,
                'diff_signal': diff_signal,
                'insider_signal': insider_signal,
                'accumulation_score': accumulation_score,
                'inflection_signal': inflection_signal,
                'inflection_score': inflection_score,
                'filings': filings,  # all filing entries for detail view
            }
            stocks.append(entry)

        # ---- Filters ----
        if exchange_filter:
            stocks = [s for s in stocks if s.get('exchange', '') == exchange_filter]
        if min_score:
            stocks = [s for s in stocks if s['score'] >= float(min_score)]

        # ---- Sort ----
        if sort_by == 'score_desc':
            stocks.sort(key=lambda x: x['score'], reverse=True)
        elif sort_by == 'score_asc':
            stocks.sort(key=lambda x: x['score'])
        elif sort_by == 'ticker':
            stocks.sort(key=lambda x: x['ticker'])
        elif sort_by == 'date':
            stocks.sort(key=lambda x: x['latest_filing_date'], reverse=True)
        elif sort_by == 'trajectory':
            traj_order = {'strongly_improving': 5, 'improving': 4, 'stable': 3,
                          'deteriorating': 2, 'strongly_deteriorating': 1}
            stocks.sort(key=lambda x: traj_order.get(x['trajectory'], 3), reverse=True)
        elif sort_by == 'filing_count':
            stocks.sort(key=lambda x: x['filing_count'], reverse=True)

        return stocks

    def clear_universe(self):
        self.universe = []
        self.filings_index = []
        self._save_state()
