"""
SEC Filing Analyzer - Web Application
=====================================
A web-based tool for analyzing microcap and OTC stock SEC filings
using the Microcap Scoring and Flagging System.

Supports two analysis modes:
  - Regex (default): Fast pattern matching, no API key needed
  - LLM (Claude API): Deep qualitative analysis with sentiment across
    5 categories, nuanced flag detection, and change tracking

Upload 10-K, 10-Q, and 8-K filings to get:
- Risk scores (0-100)
- Change-focused flag detection
- Sentiment analysis (5 categories in LLM mode)
- Investment thesis

Run with: python app.py
Then open: http://localhost:5000
"""

import os
import json
import re
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import pdfplumber
from analyzer import SECFilingAnalyzer

logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'html', 'htm'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(filepath):
    """Extract text from PDF file"""
    text = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return None
    return text


def extract_text_from_file(filepath):
    """Extract text from various file formats"""
    ext = filepath.rsplit('.', 1)[1].lower()
    
    if ext == 'pdf':
        return extract_text_from_pdf(filepath)
    elif ext in ['txt', 'html', 'htm']:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    return None


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    """Check if LLM analysis is available"""
    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    has_key = bool(api_key)
    has_sdk = False
    try:
        import anthropic
        has_sdk = True
    except ImportError:
        pass
    return jsonify({
        'llm_available': has_key and has_sdk,
        'has_api_key': has_key,
        'has_sdk': has_sdk,
    })


@app.route('/api/set-key', methods=['POST'])
def set_api_key():
    """Set the Anthropic API key at runtime (stored in env only, not persisted)"""
    data = request.get_json()
    key = data.get('api_key', '').strip()
    if not key:
        return jsonify({'error': 'No API key provided'}), 400
    os.environ['ANTHROPIC_API_KEY'] = key
    return jsonify({'success': True})


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded SEC filings"""
    
    # Get form data
    ticker = request.form.get('ticker', 'UNKNOWN').upper()
    company_name = request.form.get('company_name', 'Unknown Company')
    use_llm = request.form.get('use_llm', 'false').lower() == 'true'
    
    # Check for files
    if 'current_filing' not in request.files:
        return jsonify({'error': 'No current filing uploaded'}), 400
    
    current_file = request.files['current_filing']
    prior_file = request.files.get('prior_filing')
    
    if current_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(current_file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PDF, TXT, HTML'}), 400
    
    # Save and extract current filing
    current_filename = secure_filename(f"{ticker}_current_{current_file.filename}")
    current_path = os.path.join(app.config['UPLOAD_FOLDER'], current_filename)
    current_file.save(current_path)
    
    current_text = extract_text_from_file(current_path)
    if not current_text:
        return jsonify({'error': 'Could not extract text from current filing'}), 400
    
    # Save and extract prior filing if provided
    prior_text = None
    if prior_file and prior_file.filename != '':
        if allowed_file(prior_file.filename):
            prior_filename = secure_filename(f"{ticker}_prior_{prior_file.filename}")
            prior_path = os.path.join(app.config['UPLOAD_FOLDER'], prior_filename)
            prior_file.save(prior_path)
            prior_text = extract_text_from_file(prior_path)
    
    # Run analysis
    analyzer = SECFilingAnalyzer(use_llm=use_llm)
    
    try:
        results = analyzer.analyze_filing(
            ticker=ticker,
            company_name=company_name,
            current_text=current_text,
            prior_text=prior_text
        )
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    
    # Save results
    result_filename = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Add result filename for download
    results['result_file'] = result_filename
    
    return jsonify(results)


@app.route('/download/<filename>')
def download(filename):
    """Download analysis results"""
    filepath = os.path.join(app.config['RESULTS_FOLDER'], secure_filename(filename))
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404


@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    """API endpoint for analyzing raw text (for programmatic use)"""
    data = request.get_json()
    
    if not data or 'current_text' not in data:
        return jsonify({'error': 'current_text required'}), 400
    
    use_llm = data.get('use_llm', False)
    analyzer = SECFilingAnalyzer(use_llm=use_llm)
    
    try:
        results = analyzer.analyze_filing(
            ticker=data.get('ticker', 'UNKNOWN'),
            company_name=data.get('company_name', 'Unknown Company'),
            current_text=data['current_text'],
            prior_text=data.get('prior_text')
        )
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    
    return jsonify(results)


# =========================================================================
# SCRAPER ROUTES
# =========================================================================

from scraper import SECScraper
from monitor import MonitorState, DailyScheduler, run_daily_check
import threading

# Global scraper state
_scraper = None
_monitor_state = None
_scheduler = None
_scraper_status = {
    "running": False,
    "step": "",
    "progress": 0,
    "total": 0,
    "current_ticker": "",
    "message": "",
    "error": None,
}
_scraper_lock = threading.Lock()


def _get_scraper():
    global _scraper
    if _scraper is None:
        _scraper = SECScraper()
    return _scraper


def _get_monitor_state():
    global _monitor_state
    if _monitor_state is None:
        _monitor_state = MonitorState()
    return _monitor_state


def _get_scheduler():
    global _scheduler
    if _scheduler is None:
        _scheduler = DailyScheduler(_get_scraper(), _get_monitor_state())
    return _scheduler


@app.route('/scraper')
def scraper_page():
    """Scraper dashboard page"""
    return render_template('scraper.html')


@app.route('/api/scraper/status')
def scraper_status():
    """Get scraper status and stats"""
    scraper = _get_scraper()
    stats = scraper.get_stats()
    stats["pipeline_status"] = _scraper_status.copy()
    stats["fmp_key_set"] = bool(os.environ.get("FMP_API_KEY", ""))
    return jsonify(stats)


@app.route('/api/scraper/set-fmp-key', methods=['POST'])
def set_fmp_key():
    """Set the FMP API key"""
    data = request.get_json()
    key = data.get('api_key', '').strip()
    if not key:
        return jsonify({'error': 'No API key provided'}), 400
    os.environ['FMP_API_KEY'] = key
    return jsonify({'success': True})


@app.route('/api/scraper/run', methods=['POST'])
def run_scraper():
    """Start the scraper pipeline in a background thread"""
    global _scraper_status

    with _scraper_lock:
        if _scraper_status["running"]:
            return jsonify({'error': 'Scraper is already running'}), 409

    data = request.get_json() or {}
    steps = data.get('steps', ['universe', 'find', 'download'])
    max_companies = data.get('max_companies', None)
    max_downloads = data.get('max_downloads', None)

    def run_pipeline():
        global _scraper_status
        scraper = _get_scraper()

        try:
            _scraper_status["running"] = True
            _scraper_status["error"] = None

            if 'universe' in steps:
                _scraper_status["step"] = "Building company universe"
                _scraper_status["progress"] = 0
                _scraper_status["total"] = 0

                def universe_progress(msg):
                    _scraper_status["message"] = msg

                fmp_key = os.environ.get("FMP_API_KEY", "")
                if not fmp_key:
                    raise RuntimeError(
                        "FMP API key not set. Go to Settings and enter your "
                        "FMP API key. Requires Starter plan ($22/mo) or higher. "
                        "Sign up: https://financialmodelingprep.com/register"
                    )

                _scraper_status["message"] = "Querying FMP screener for $20-100M US companies..."
                count = scraper.step1_build_universe(
                    fmp_api_key=fmp_key,
                    progress_callback=universe_progress
                )
                _scraper_status["message"] = f"Universe: {count} companies ($20-100M market cap)"

            if 'find' in steps:
                _scraper_status["step"] = "Finding filings on EDGAR"
                _scraper_status["progress"] = 0

                def find_progress(current, total, ticker, num_filings):
                    _scraper_status["progress"] = current
                    _scraper_status["total"] = total
                    _scraper_status["current_ticker"] = ticker
                    _scraper_status["message"] = (
                        f"{current}/{total} companies — {ticker} ({num_filings} filings)"
                    )

                count = scraper.step2_find_filings(
                    max_companies=max_companies,
                    progress_callback=find_progress
                )
                _scraper_status["message"] = f"Found {count} filings total"

            if 'download' in steps:
                _scraper_status["step"] = "Downloading filings"
                _scraper_status["progress"] = 0

                def dl_progress(current, total, ticker, success):
                    _scraper_status["progress"] = current
                    _scraper_status["total"] = total
                    _scraper_status["current_ticker"] = ticker
                    status = "✓" if success else "✗"
                    _scraper_status["message"] = (
                        f"{current}/{total} — {ticker} {status}"
                    )

                count = scraper.step3_download_filings(
                    max_downloads=max_downloads,
                    progress_callback=dl_progress
                )
                _scraper_status["message"] = f"Downloaded {count} filings"

            _scraper_status["step"] = "Complete"
            _scraper_status["message"] = "Pipeline finished successfully"

        except Exception as e:
            _scraper_status["error"] = str(e)
            _scraper_status["step"] = "Error"
            _scraper_status["message"] = str(e)
            logger.error(f"Pipeline error: {e}", exc_info=True)
        finally:
            _scraper_status["running"] = False

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    return jsonify({'success': True, 'message': 'Scraper pipeline started'})


@app.route('/api/scraper/stop', methods=['POST'])
def stop_scraper():
    """Request scraper stop (currently just updates status)"""
    global _scraper_status
    _scraper_status["message"] = "Stop requested... will halt after current operation"
    # Note: full graceful shutdown would need a cancel token passed into scraper
    return jsonify({'success': True})


@app.route('/api/scraper/universe')
def get_universe():
    """Get the company universe"""
    scraper = _get_scraper()
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 50))
    exchange = request.args.get('exchange', '')

    companies = scraper.universe
    if exchange:
        companies = [c for c in companies if c.get('exchange') == exchange]

    start = (page - 1) * per_page
    end = start + per_page

    return jsonify({
        'total': len(companies),
        'page': page,
        'per_page': per_page,
        'companies': companies[start:end],
    })


@app.route('/api/scraper/filings')
def get_filings():
    """Get the filings index with optional filters"""
    scraper = _get_scraper()
    status_filter = request.args.get('status', '')  # downloaded, pending, analyzed
    form_type = request.args.get('form_type', '')
    ticker = request.args.get('ticker', '')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 50))

    filings = scraper.filings_index

    if status_filter == 'downloaded':
        filings = [f for f in filings if f.get('downloaded') and not f.get('analyzed')]
    elif status_filter == 'pending':
        filings = [f for f in filings if not f.get('downloaded')]
    elif status_filter == 'analyzed':
        filings = [f for f in filings if f.get('analyzed')]

    if form_type:
        filings = [f for f in filings if f.get('form_type') == form_type]

    if ticker:
        filings = [f for f in filings if f.get('ticker', '').upper() == ticker.upper()]

    start = (page - 1) * per_page
    end = start + per_page

    return jsonify({
        'total': len(filings),
        'page': page,
        'per_page': per_page,
        'filings': filings[start:end],
    })


@app.route('/api/scraper/analyze-filing', methods=['POST'])
def analyze_scraped_filing():
    """Analyze a specific scraped filing by accession number"""
    data = request.get_json()
    accession = data.get('accession_number', '')
    use_llm = data.get('use_llm', False)

    scraper = _get_scraper()
    filing = None
    for f in scraper.filings_index:
        if f.get('accession_number') == accession:
            filing = f
            break

    if not filing:
        return jsonify({'error': 'Filing not found'}), 404

    if not filing.get('downloaded') or not filing.get('local_path'):
        return jsonify({'error': 'Filing not downloaded yet'}), 400

    filepath = filing['local_path']
    if not os.path.exists(filepath):
        return jsonify({'error': f'File not found at {filepath}'}), 404

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    if filepath.endswith('.html') or filepath.endswith('.htm'):
        from scraper import extract_text_from_html
        content = extract_text_from_html(content)

    analyzer = SECFilingAnalyzer(use_llm=use_llm)
    results = analyzer.analyze_filing(
        ticker=filing.get('ticker', 'UNKNOWN'),
        company_name=filing.get('company_name', 'Unknown'),
        current_text=content,
    )

    result_filename = (
        f"{filing.get('ticker', 'UNK')}_"
        f"{filing.get('form_type', '').replace('/', '-')}_"
        f"{filing.get('filing_date', '')}.json"
    )
    result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
    with open(result_path, 'w') as rf:
        json.dump(results, rf, indent=2)

    scraper.mark_analyzed(accession, {
        "final_score": results.get("final_score"),
        "risk_rating": results.get("risk_rating"),
        "red_flag_count": results.get("score_breakdown", {}).get("red_flag_count", 0),
        "yellow_flag_count": results.get("score_breakdown", {}).get("yellow_flag_count", 0),
        "green_flag_count": results.get("score_breakdown", {}).get("green_flag_count", 0),
        "sentiment_trajectory": results.get("sentiment_trajectory"),
        "key_concerns": results.get("key_concerns", []),
        "result_file": result_filename,
    })

    results['result_file'] = result_filename
    return jsonify(results)


# =========================================================================
# BATCH ANALYSIS
# =========================================================================

_batch_status = {
    "running": False, "progress": 0, "total": 0,
    "current_ticker": "", "message": "",
    "errors": [], "completed": 0, "skipped": 0,
}


@app.route('/api/scraper/batch-analyze', methods=['POST'])
def batch_analyze():
    """Run analysis on all downloaded but unanalyzed filings"""
    global _batch_status
    if _batch_status["running"]:
        return jsonify({'error': 'Batch analysis already running'}), 409

    data = request.get_json() or {}
    use_llm = data.get('use_llm', False)
    max_filings = data.get('max_filings', None)
    reanalyze = data.get('reanalyze', False)

    def run_batch():
        global _batch_status
        scraper = _get_scraper()
        try:
            _batch_status = {
                "running": True, "progress": 0, "total": 0,
                "current_ticker": "", "message": "Starting batch analysis...",
                "errors": [], "completed": 0, "skipped": 0,
            }
            if reanalyze:
                pending = [f for f in scraper.filings_index if f.get('downloaded')]
            else:
                pending = [f for f in scraper.filings_index
                           if f.get('downloaded') and not f.get('analyzed')]
            if max_filings:
                pending = pending[:int(max_filings)]

            _batch_status["total"] = len(pending)
            _batch_status["message"] = f"Analyzing {len(pending)} filings..."

            analyzer = SECFilingAnalyzer(use_llm=use_llm)

            for i, filing in enumerate(pending):
                ticker = filing.get('ticker', 'UNKNOWN')
                form_type = filing.get('form_type', '')
                _batch_status["progress"] = i + 1
                _batch_status["current_ticker"] = ticker
                _batch_status["message"] = f"{i+1}/{len(pending)} — {ticker} {form_type}"

                filepath = filing.get('local_path', '')
                if not filepath or not os.path.exists(filepath):
                    _batch_status["skipped"] += 1
                    _batch_status["errors"].append(f"{ticker}: File not found")
                    continue

                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    if filepath.endswith('.html') or filepath.endswith('.htm'):
                        from scraper import extract_text_from_html
                        content = extract_text_from_html(content)
                    if not content or len(content) < 200:
                        _batch_status["skipped"] += 1
                        _batch_status["errors"].append(f"{ticker}: Filing too short")
                        continue

                    results = analyzer.analyze_filing(
                        ticker=ticker,
                        company_name=filing.get('company_name', 'Unknown'),
                        current_text=content,
                    )

                    result_filename = (
                        f"{ticker}_{form_type.replace('/', '-')}_"
                        f"{filing.get('filing_date', '')}.json"
                    )
                    result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
                    with open(result_path, 'w') as rf:
                        json.dump(results, rf, indent=2)

                    scraper.mark_analyzed(filing.get('accession_number', ''), {
                        "final_score": results.get("final_score"),
                        "risk_rating": results.get("risk_rating"),
                        "red_flag_count": results.get("score_breakdown", {}).get("red_flag_count", 0),
                        "yellow_flag_count": results.get("score_breakdown", {}).get("yellow_flag_count", 0),
                        "green_flag_count": results.get("score_breakdown", {}).get("green_flag_count", 0),
                        "sentiment_trajectory": results.get("sentiment_trajectory"),
                        "key_concerns": results.get("key_concerns", []),
                        "result_file": result_filename,
                    })
                    _batch_status["completed"] += 1
                except Exception as e:
                    _batch_status["errors"].append(f"{ticker}: {str(e)}")
                    _batch_status["skipped"] += 1

            _batch_status["message"] = (
                f"Complete — {_batch_status['completed']} analyzed, "
                f"{_batch_status['skipped']} skipped"
            )
        except Exception as e:
            _batch_status["message"] = f"Batch error: {str(e)}"
            _batch_status["errors"].append(str(e))
        finally:
            _batch_status["running"] = False

    thread = threading.Thread(target=run_batch, daemon=True)
    thread.start()
    return jsonify({'success': True, 'message': 'Batch analysis started'})


@app.route('/api/scraper/batch-status')
def batch_status():
    """Get batch analysis progress"""
    return jsonify(_batch_status)


@app.route('/api/scraper/results')
def get_results():
    """Get analyzed filings ranked by risk score"""
    scraper = _get_scraper()
    sort_by = request.args.get('sort', 'score_asc')
    risk_filter = request.args.get('risk', '')
    exchange_filter = request.args.get('exchange', '')
    form_type_filter = request.args.get('form_type', '')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 50))

    results = scraper.get_results_ranked(
        sort_by=sort_by, risk_filter=risk_filter,
        exchange_filter=exchange_filter, form_type_filter=form_type_filter,
    )
    start = (page - 1) * per_page
    end = start + per_page

    scores = [r.get('score', 50) for r in results]
    risk_dist = {}
    for r in results:
        rating = r.get('risk_rating', 'UNKNOWN')
        risk_dist[rating] = risk_dist.get(rating, 0) + 1

    return jsonify({
        'total': len(results), 'page': page, 'per_page': per_page,
        'results': results[start:end],
        'summary': {
            'avg_score': round(sum(scores) / len(scores), 1) if scores else 0,
            'min_score': min(scores) if scores else 0,
            'max_score': max(scores) if scores else 0,
            'risk_distribution': risk_dist,
        }
    })


@app.route('/api/scraper/result/<filename>')
def get_result_detail(filename):
    """Get full analysis result for a specific filing"""
    filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'Result not found'}), 404
    with open(filepath) as f:
        return jsonify(json.load(f))


# =========================================================================
# DAILY MONITOR ROUTES
# =========================================================================

@app.route('/monitor')
def monitor_page():
    """Daily monitor dashboard"""
    return render_template('monitor.html')


@app.route('/api/monitor/status')
def monitor_status():
    """Get monitor state, scheduler status, and recent log"""
    state = _get_monitor_state()
    scheduler = _get_scheduler()
    return jsonify({
        "state": state.state,
        "scheduler_running": scheduler.is_running,
        "last_result": scheduler.last_result,
        "recent_log": state.log[-30:],
    })


@app.route('/api/monitor/configure', methods=['POST'])
def monitor_configure():
    """Update monitor configuration"""
    data = request.get_json() or {}
    state = _get_monitor_state()

    if 'run_time' in data:
        rt = data['run_time'].strip()
        try:
            h, m = map(int, rt.split(":"))
            if 0 <= h <= 23 and 0 <= m <= 59:
                state.state["run_time"] = rt
        except (ValueError, AttributeError):
            return jsonify({'error': 'Invalid time format. Use HH:MM (24h)'}), 400

    if 'auto_analyze' in data:
        state.state["auto_analyze"] = bool(data['auto_analyze'])
    if 'use_llm_for_auto' in data:
        state.state["use_llm_for_auto"] = bool(data['use_llm_for_auto'])

    state.save()
    return jsonify({'success': True, 'state': state.state})


@app.route('/api/monitor/start', methods=['POST'])
def monitor_start():
    """Start the daily scheduler"""
    scraper = _get_scraper()
    if not scraper.universe:
        return jsonify({
            'error': 'No company universe loaded. Run the scraper pipeline first.'
        }), 400
    scheduler = _get_scheduler()
    scheduler.start()
    return jsonify({'success': True, 'message': 'Daily monitor started'})


@app.route('/api/monitor/stop', methods=['POST'])
def monitor_stop():
    """Stop the daily scheduler"""
    scheduler = _get_scheduler()
    scheduler.stop()
    return jsonify({'success': True, 'message': 'Daily monitor stopped'})


@app.route('/api/monitor/run-now', methods=['POST'])
def monitor_run_now():
    """Trigger an immediate check"""
    scraper = _get_scraper()
    if not scraper.universe:
        return jsonify({
            'error': 'No company universe loaded. Run the scraper pipeline first.'
        }), 400

    scheduler = _get_scheduler()

    def do_run():
        try:
            scheduler.run_now()
        except Exception as e:
            logger.error(f"Manual check failed: {e}")

    thread = threading.Thread(target=do_run, daemon=True)
    thread.start()
    return jsonify({'success': True, 'message': 'Check started. Poll /api/monitor/status for results.'})


@app.route('/api/monitor/log')
def monitor_log():
    """Get the monitor run log"""
    state = _get_monitor_state()
    log = list(reversed(state.log))
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 30))
    start = (page - 1) * per_page
    return jsonify({
        'total': len(log),
        'page': page,
        'entries': log[start:start + per_page],
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 60)
    print("SEC Filing Analyzer")
    print("=" * 60)
    print(f"Starting server on port {port}")
    print("Upload SEC filings (PDF, TXT, HTML) to analyze")
    print("=" * 60)
    app.run(debug=False, host='0.0.0.0', port=port)
