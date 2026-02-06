# SEC Filing Analyzer

A web-based tool for analyzing SEC filings (10-K, 10-Q, 8-K) for microcap and OTC stocks.

## Features

- **Risk Scoring (0-100)**: Comprehensive risk assessment based on 20+ flag patterns
- **Change Detection**: Compares current filing to prior filing to detect NEW and WORSENING issues
- **Sentiment Analysis**: Analyzes management tone and language shifts
- **Flag Categories**: Liquidity, Debt, Auditor, Governance, Accounting, Legal, Operations, and more

## Installation

### Option 1: Local Installation

```bash
# Clone or download the sec_analyzer folder

# Install dependencies
cd sec_analyzer
pip install -r requirements.txt

# Run the application
python app.py

# Open in browser
# http://localhost:5000
```

### Option 2: Docker (Optional)

```bash
# Build
docker build -t sec-analyzer .

# Run
docker run -p 5000:5000 sec-analyzer
```

## Usage

### Web Interface

1. Open http://localhost:5000 in your browser
2. Enter the company's ticker symbol and name
3. Upload the current SEC filing (PDF, TXT, or HTML)
4. Optionally upload a prior filing for change detection
5. Click "Analyze Filing"
6. Review the results:
   - Risk score and rating
   - Detected flags with impact scores
   - Sentiment analysis
   - Key concerns and positives

### API Usage

You can also use the API programmatically:

```python
import requests

# Analyze with file upload
files = {
    'current_filing': open('filing.pdf', 'rb'),
    'prior_filing': open('prior_filing.pdf', 'rb')  # optional
}
data = {
    'ticker': 'RGS',
    'company_name': 'Regis Corporation'
}

response = requests.post('http://localhost:5000/analyze', files=files, data=data)
results = response.json()

print(f"Score: {results['final_score']}/100")
print(f"Rating: {results['risk_rating']}")
```

Or with raw text:

```python
import requests

data = {
    'ticker': 'RGS',
    'company_name': 'Regis Corporation',
    'current_text': '... filing text ...',
    'prior_text': '... prior filing text ...'  # optional
}

response = requests.post(
    'http://localhost:5000/api/analyze-text',
    json=data
)
results = response.json()
```

## Scoring System

### Risk Levels

| Score | Rating |
|-------|--------|
| 70-100 | LOW RISK |
| 55-69 | MODERATE RISK |
| 40-54 | ELEVATED RISK |
| 25-39 | HIGH RISK |
| 0-24 | CRITICAL RISK |

### Change Multipliers

| Change Type | Multiplier | Effect |
|-------------|-----------|--------|
| NEW | 2.0x | New problems penalized heavily |
| WORSENING | 1.75x | Deteriorating trends penalized |
| UNCHANGED | 1.0x | Persistent issues = base weight |
| IMPROVING | 0.5x | Improving problems = reduced penalty |

### Flag Categories

**Red Flags (Negative)**
- Going Concern Warning
- Auditor Resignation
- Financial Restatement
- SEC Investigation
- Covenant Violation
- Material Weakness
- Refinancing Risk
- Negative Equity
- Related Party Transactions
- CEO/CFO Departure
- Goodwill Impairment

**Yellow Flags (Caution)**
- Operating Losses
- Share Dilution Risk
- Material Litigation

**Green Flags (Positive)**
- Positive Cash Flow
- Debt Reduction
- Same-Store Sales Growth
- Covenant Compliance
- Effective Internal Controls

### Sentiment Levels

| Level | Score Impact |
|-------|-------------|
| VERY_CONFIDENT | +10 pts |
| CONFIDENT | +5 pts |
| NEUTRAL | 0 pts |
| CAUTIOUS | -5 pts |
| DEFENSIVE | -10 pts |
| ALARMING | -20 pts |

## File Structure

```
sec_analyzer/
├── app.py              # Flask web application
├── analyzer.py         # Core analysis module
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Web interface
├── uploads/            # Uploaded files (auto-created)
└── results/            # Analysis results (auto-created)
```

## Customization

### Adding New Flag Rules

Edit `analyzer.py` and add to the `FLAG_RULES` dictionary:

```python
"my_new_flag": {
    "category": Category.OPERATIONS,
    "signal_type": SignalType.YELLOW_FLAG,
    "risk_level": RiskLevel.MODERATE,
    "title": "My New Flag",
    "patterns": [
        r"pattern to match",
        r"another pattern"
    ],
    "description": "What this flag means"
}
```

### Adjusting Weights

Modify `CATEGORY_WEIGHTS` and `CHANGE_MULTIPLIERS` in `analyzer.py` to change how different categories and change types affect the final score.

## Limitations

- Pattern matching is regex-based and may miss nuanced language
- PDF extraction quality depends on document formatting
- Sentiment analysis is keyword-based, not contextual
- For best results, upload clean text files or well-formatted PDFs

## License

MIT License - Free for personal and commercial use.
