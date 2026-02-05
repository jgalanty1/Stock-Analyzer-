"""
LLM Analysis Module - Claude API Integration
=============================================
Provides deep qualitative analysis of SEC filings using Claude.
Handles sentiment analysis, red flag detection, and management tone
across 5 categories with change tracking.
"""

import os
import json
import re
from typing import Optional, Dict, Any, List


def _call_claude(prompt: str, system: str = "", max_tokens: int = 4096) -> Optional[str]:
    """
    Call the Anthropic Claude API.
    Returns the text response or None on failure.
    """
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package not installed. Run: pip install anthropic")
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in environment.")
        return None

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            system=system if system else "You are a forensic financial analyst specializing in microcap and OTC stocks. You analyze SEC filings for risk signals, sentiment shifts, and red flags. Always respond in valid JSON.",
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        print(f"Claude API error: {e}")
        return None


def _parse_json_response(text: str) -> Optional[Dict]:
    """Safely parse JSON from Claude's response, stripping markdown fences."""
    if not text:
        return None
    cleaned = re.sub(r"```json\s*", "", text)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def _truncate_text(text: str, max_chars: int = 80000) -> str:
    """Truncate text to fit within token limits while preserving structure."""
    if len(text) <= max_chars:
        return text
    # Keep first and last portions
    half = max_chars // 2
    return text[:half] + "\n\n[...TRUNCATED...]\n\n" + text[-half:]


# ============================================================================
# LLM SENTIMENT ANALYSIS
# ============================================================================

SENTIMENT_PROMPT_SINGLE = """Analyze the management tone and sentiment in this SEC filing.

Evaluate these 5 categories and rate each from 0-5:
  0 = ALARMING (going concern, strategic alternatives, liquidity crisis language)
  1 = DEFENSIVE (blaming external factors, "despite challenges", "unprecedented")
  2 = CAUTIOUS (headwinds, uncertainty, monitoring closely, pressure on)
  3 = NEUTRAL (as expected, in line with, maintained, stable)
  4 = CONFIDENT (pleased with, on track, good progress, gaining traction)
  5 = VERY_CONFIDENT (exceeded expectations, record results, strong momentum)

Categories:
1. overall_tone - General management sentiment across the entire filing
2. forward_guidance - How specific/confident are forward-looking statements? Vague = lower, specific targets = higher
3. liquidity_discussion - Tone when discussing cash, debt, financing. Defensive/alarming = lower
4. operational_outlook - Confidence about core business operations, revenue, growth
5. competitive_position - How management frames the company vs. competitors and market

For each category also provide:
- key_phrases: 2-3 direct short quotes (under 15 words each) that support your rating
- reasoning: 1-2 sentence explanation

Respond ONLY with a JSON object in this exact format:
{{
  "categories": {{
    "overall_tone": {{
      "score": <0-5>,
      "level": "<ALARMING|DEFENSIVE|CAUTIOUS|NEUTRAL|CONFIDENT|VERY_CONFIDENT>",
      "key_phrases": ["phrase1", "phrase2"],
      "reasoning": "..."
    }},
    "forward_guidance": {{ ... }},
    "liquidity_discussion": {{ ... }},
    "operational_outlook": {{ ... }},
    "competitive_position": {{ ... }}
  }}
}}

FILING TEXT:
{filing_text}"""


SENTIMENT_PROMPT_COMPARISON = """Analyze the management tone shift between these two SEC filings for the SAME company.

Rate each category from 0-5 for BOTH filings:
  0 = ALARMING, 1 = DEFENSIVE, 2 = CAUTIOUS, 3 = NEUTRAL, 4 = CONFIDENT, 5 = VERY_CONFIDENT

Categories:
1. overall_tone - General management sentiment
2. forward_guidance - Specificity and confidence of forward-looking statements
3. liquidity_discussion - Tone around cash, debt, financing
4. operational_outlook - Confidence about core business operations
5. competitive_position - Company vs. competitors framing

For each category, pay special attention to:
- WORD SUBSTITUTIONS: Did "expect" become "hope"? Did "will" become "may"?
- BLAME ATTRIBUTION: Is management blaming external factors more than before?
- METRIC EMPHASIS: Have they stopped highlighting metrics they previously featured?
- HEDGING LANGUAGE: More qualifiers, caveats, or "subject to" phrases?
- SPECIFICITY: Are projections becoming vaguer or more specific?

Respond ONLY with a JSON object in this exact format:
{{
  "categories": {{
    "overall_tone": {{
      "prior_score": <0-5>,
      "prior_level": "<level>",
      "current_score": <0-5>,
      "current_level": "<level>",
      "change_direction": "<improving|worsening|stable>",
      "key_shifts": ["specific shift observation 1", "shift observation 2"],
      "key_phrases_current": ["phrase1", "phrase2"],
      "key_phrases_prior": ["phrase1", "phrase2"],
      "reasoning": "..."
    }},
    "forward_guidance": {{ ... }},
    "liquidity_discussion": {{ ... }},
    "operational_outlook": {{ ... }},
    "competitive_position": {{ ... }}
  }},
  "overall_trajectory": "<improving|worsening|stable>",
  "most_significant_shift": "One sentence describing the single most important tone change",
  "word_substitutions": ["'X' replaced by 'Y' in context of Z", ...]
}}

PRIOR FILING:
{prior_text}

CURRENT FILING:
{current_text}"""


# ============================================================================
# LLM RED FLAG DEEP ANALYSIS
# ============================================================================

RED_FLAG_PROMPT_SINGLE = """You are a forensic accountant analyzing a microcap company's SEC filing.

Perform a deep-dive analysis screening for ALL of the following risks. For each risk found, provide evidence. For risks NOT found, confirm they were checked.

CRITICAL RISKS (check all):
1. Going concern language from auditors or management
2. Auditor changes or qualified opinions
3. Financial restatements or material misstatements
4. SEC investigations or enforcement actions
5. Debt covenant violations or waivers

HIGH RISKS:
6. Material weakness in internal controls
7. Debt refinancing uncertainty (maturing debt with no clear plan)
8. Negative shareholders' equity
9. Material related-party transactions (especially vague ones)
10. Revenue recognition policy changes

MODERATE RISKS:
11. CEO/CFO departure or interim leadership
12. Goodwill/asset impairment charges
13. Operating losses or accumulated deficit
14. Share dilution (shelf registrations, ATM offerings, warrants, convertibles)
15. Material litigation or contingent liabilities

POSITIVE SIGNALS (also flag these):
16. Positive operating cash flow
17. Debt reduction or paydown
18. Same-store or organic revenue growth
19. Covenant compliance confirmed
20. Effective internal controls confirmed

Respond ONLY with a JSON object:
{{
  "flags_detected": [
    {{
      "id": "<short_id>",
      "title": "<descriptive title>",
      "category": "<liquidity|debt|auditor|accounting|legal|management|operations|governance|dilution|related_party|revenue|profitability|insider>",
      "signal_type": "<red_flag|yellow_flag|green_flag>",
      "risk_level": "<CRITICAL|HIGH|MODERATE|LOW>",
      "evidence": "<direct quote or specific reference from filing, under 15 words>",
      "context": "<2-3 sentence explanation of why this matters>",
      "score_impact": <integer, negative for risks, positive for good signals>
    }}
  ],
  "risks_checked_not_found": ["risk description 1", "risk description 2"],
  "overall_assessment": "<1-2 sentence summary>"
}}

FILING TEXT:
{filing_text}"""


RED_FLAG_PROMPT_COMPARISON = """You are a forensic accountant comparing two consecutive SEC filings for the SAME microcap company.

Your PRIMARY job is detecting CHANGES between filings. For every flag, you MUST classify the change type.

Change types:
- NEW: Risk appears in current filing but NOT in prior filing (2.0x weight)
- WORSENING: Risk existed before but has gotten worse (1.75x weight)  
- UNCHANGED: Risk exists in both filings at similar severity (1.0x weight)
- IMPROVING: Risk existed before but is getting better (0.5x weight)
- RESOLVED: Risk was in prior filing but is gone from current (0.0x weight)

For GREEN flags, the change logic inverts:
- NEW positive = 1.5x weight (reward)
- IMPROVING positive = 1.75x weight (strong reward)

Screen for all risks (critical through positive signals) and report what you find.

Respond ONLY with a JSON object:
{{
  "flags_detected": [
    {{
      "id": "<short_id>",
      "title": "<descriptive title>",
      "category": "<category>",
      "signal_type": "<red_flag|yellow_flag|green_flag>",
      "risk_level": "<CRITICAL|HIGH|MODERATE|LOW>",
      "change_type": "<new|worsening|unchanged|improving|resolved>",
      "evidence_current": "<quote from current filing, under 15 words>",
      "evidence_prior": "<quote from prior filing or 'not present', under 15 words>",
      "context": "<2-3 sentence explanation focusing on WHAT CHANGED>",
      "score_impact": <integer>
    }}
  ],
  "resolved_risks": [
    {{
      "title": "<what was resolved>",
      "evidence_prior": "<what it said before>",
      "significance": "<why this matters>"
    }}
  ],
  "change_summary": {{
    "new_risks": <count>,
    "worsening": <count>,
    "unchanged": <count>,
    "improving": <count>,
    "resolved": <count>,
    "net_trajectory": "<deteriorating|stable|improving>"
  }},
  "overall_assessment": "<2-3 sentence summary focusing on trajectory>"
}}

PRIOR FILING:
{prior_text}

CURRENT FILING:
{current_text}"""


# ============================================================================
# PUBLIC API
# ============================================================================

def llm_analyze_sentiment(
    current_text: str,
    prior_text: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Run LLM-powered sentiment analysis across 5 categories.
    Returns structured sentiment results or None if API unavailable.
    """
    current_truncated = _truncate_text(current_text)

    if prior_text:
        prior_truncated = _truncate_text(prior_text)
        prompt = SENTIMENT_PROMPT_COMPARISON.format(
            prior_text=prior_truncated,
            current_text=current_truncated,
        )
    else:
        prompt = SENTIMENT_PROMPT_SINGLE.format(
            filing_text=current_truncated,
        )

    raw = _call_claude(prompt)
    parsed = _parse_json_response(raw)
    if not parsed:
        return None

    # Normalize into a consistent list format for the UI
    results = []
    categories = parsed.get("categories", {})

    level_scores = {
        "VERY_CONFIDENT": 10, "CONFIDENT": 5, "NEUTRAL": 0,
        "CAUTIOUS": -5, "DEFENSIVE": -10, "ALARMING": -20,
    }

    for cat_name, cat_data in categories.items():
        if prior_text:
            current_level = cat_data.get("current_level", "NEUTRAL")
            prior_level = cat_data.get("prior_level", "NEUTRAL")
            change_dir = cat_data.get("change_direction", "stable")
            key_phrases = cat_data.get("key_phrases_current", [])
            key_shifts = cat_data.get("key_shifts", [])
        else:
            current_level = cat_data.get("level", "NEUTRAL")
            prior_level = None
            change_dir = "first_filing"
            key_phrases = cat_data.get("key_phrases", [])
            key_shifts = []

        base = level_scores.get(current_level, 0)
        if change_dir == "worsening":
            mult = 2.0
        elif change_dir == "improving":
            mult = 1.5
        else:
            mult = 1.0

        if base < 0:
            impact = int(base * mult)
        else:
            impact = int(base * (mult if change_dir == "improving" else 1.0))

        results.append({
            "category": cat_name,
            "current_level": current_level,
            "prior_level": prior_level,
            "change_direction": change_dir,
            "score_impact": impact,
            "key_phrases": key_phrases[:5],
            "key_shifts": key_shifts[:3],
            "reasoning": cat_data.get("reasoning", ""),
        })

    # Attach top-level metadata from comparison
    meta = {
        "overall_trajectory": parsed.get("overall_trajectory", "stable"),
        "most_significant_shift": parsed.get("most_significant_shift", ""),
        "word_substitutions": parsed.get("word_substitutions", []),
    }

    return {"sentiment_results": results, "sentiment_meta": meta}


def llm_analyze_flags(
    current_text: str,
    prior_text: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run LLM-powered deep flag detection.
    Returns structured flag results or None if API unavailable.
    """
    current_truncated = _truncate_text(current_text)

    if prior_text:
        prior_truncated = _truncate_text(prior_text)
        prompt = RED_FLAG_PROMPT_COMPARISON.format(
            prior_text=prior_truncated,
            current_text=current_truncated,
        )
    else:
        prompt = RED_FLAG_PROMPT_SINGLE.format(
            filing_text=current_truncated,
        )

    raw = _call_claude(prompt)
    parsed = _parse_json_response(raw)
    if not parsed:
        return None

    # Normalize flags into the same shape the UI expects
    flags = []
    for f in parsed.get("flags_detected", []):
        flags.append({
            "rule_id": f.get("id", "llm_flag"),
            "title": f.get("title", "Unknown Flag"),
            "category": f.get("category", "operations"),
            "signal_type": f.get("signal_type", "yellow_flag"),
            "risk_level": f.get("risk_level", "MODERATE"),
            "change_type": f.get("change_type", "first_filing"),
            "score_impact": f.get("score_impact", 0),
            "description": f.get("context", ""),
            "evidence": f.get("evidence", f.get("evidence_current", "")),
            "evidence_prior": f.get("evidence_prior", ""),
            "source": "llm",
        })

    return {
        "flags": sorted(flags, key=lambda x: abs(x["score_impact"]), reverse=True),
        "resolved_risks": parsed.get("resolved_risks", []),
        "change_summary": parsed.get("change_summary", {}),
        "overall_assessment": parsed.get("overall_assessment", ""),
        "risks_checked_not_found": parsed.get("risks_checked_not_found", []),
    }


def is_llm_available() -> bool:
    """Check if the Claude API is configured and reachable."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return False
    try:
        import anthropic
        return True
    except ImportError:
        return False
