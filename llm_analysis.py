"""
LLM Analysis Module - Dual Model Support (OpenAI + Anthropic)
==============================================================
Two-tier analysis system for finding hidden gem microcap stocks:
  Tier 1: GPT-4o-mini (cheap bulk screening)
  Tier 2: Claude Haiku (deep analysis on top candidates)

Focus: Finding IMPROVING companies with authentic management tone,
not just avoiding disasters.
"""

import os
import json
import re
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODELS = {
    "gpt-4o-mini": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "input_cost_per_1m": 0.15,
        "output_cost_per_1m": 0.60,
        "max_context": 128000,
        "tier": 1,
    },
    "claude-haiku": {
        "provider": "anthropic",
        "model_id": "claude-haiku-4-5-20251001",
        "input_cost_per_1m": 1.00,
        "output_cost_per_1m": 5.00,
        "max_context": 200000,
        "tier": 2,
    },
    "claude-sonnet": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-20250514",
        "input_cost_per_1m": 3.00,
        "output_cost_per_1m": 15.00,
        "max_context": 200000,
        "tier": 3,
    },
}

DEFAULT_TIER1_MODEL = "gpt-4o-mini"
DEFAULT_TIER2_MODEL = "claude-haiku"


# ============================================================================
# API CLIENTS
# ============================================================================

def _call_openai(prompt: str, system: str, model: str = "gpt-4o-mini",
                 max_tokens: int = 2000) -> Optional[str]:
    """Call OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise RuntimeError(f"OpenAI API call failed: {e}")


def _call_anthropic(prompt: str, system: str, model: str = "claude-haiku-4-5-20241022",
                    max_tokens: int = 2000) -> Optional[str]:
    """Call Anthropic Claude API."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        logger.error(f"Anthropic API error (model={model}): {e}")
        raise RuntimeError(f"Anthropic API call failed: {e}")


def _call_llm(prompt: str, system: str, model_name: str = "gpt-4o-mini",
              max_tokens: int = 2000) -> Optional[str]:
    """Route to appropriate API based on model."""
    config = MODELS.get(model_name)
    if not config:
        logger.error(f"Unknown model: {model_name}")
        return None

    if config["provider"] == "openai":
        return _call_openai(prompt, system, config["model_id"], max_tokens)
    elif config["provider"] == "anthropic":
        return _call_anthropic(prompt, system, config["model_id"], max_tokens)
    return None


def _parse_json_response(text: str) -> Optional[Dict]:
    """Parse JSON from LLM response, handling markdown fences."""
    if not text:
        return None
    cleaned = re.sub(r"```json\s*", "", text)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    logger.warning(f"Failed to parse JSON: {text[:200]}...")
    return None


def _truncate_text(text: str, max_chars: int = 100000) -> str:
    """Truncate text while preserving key sections."""
    if len(text) <= max_chars:
        return text
    # Keep beginning and end (often most important)
    half = max_chars // 2
    return text[:half] + "\n\n[...CONTENT TRUNCATED FOR LENGTH...]\n\n" + text[-half:]


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

SYSTEM_PROMPT_GEM_FINDER = """You are an expert microcap stock analyst searching for HIDDEN GEMS - 
undervalued companies with improving fundamentals and authentic, confident management.

Your job is NOT just to avoid disasters. You're looking for:
- Companies quietly executing well
- Management that's confident but not promotional  
- Improving trends that the market may be missing
- Understated positives and conservative guidance

You analyze SEC filings with a focus on TONE and TRAJECTORY.
Always respond with valid JSON only."""


SYSTEM_PROMPT_DEEP_ANALYSIS = """You are a forensic financial analyst specializing in microcap stocks.
You perform deep qualitative analysis of SEC filings, grounded in QUANTITATIVE linguistic forensics.

You don't just summarize filings. You:
- INTERPRET concrete metrics (hedging ratios, confidence scores, specificity)
- EXPLAIN what filing-over-filing changes mean for investors
- IDENTIFY contradictions between management narrative and forensic signals
- FIND what management is trying to hide or downplay

When forensic diff data shows risk factors were removed, that could mean the risk was
resolved (bullish) or that they're hiding it (bearish) - your job is to determine which.

When hedging increases but management claims confidence, that's a contradiction worth flagging.
When specificity decreases, ask why they stopped giving concrete numbers.

Always respond with valid JSON only."""


# ============================================================================
# TIER 1: SCREENING PROMPT (GPT-4o-mini)
# ============================================================================

TIER1_SCREENING_PROMPT = """Analyze this SEC filing to identify if this company could be a hidden gem.

Score each dimension from 1-10:

TONE DIMENSIONS:
1. confidence (1=desperate/promotional, 5=neutral, 10=quietly assured)
2. transparency (1=vague/evasive, 5=standard, 10=unusually specific/honest)
3. consistency (1=changing story, 5=neutral, 10=same strategy, steady execution)
4. operational_focus (1=excuses/blame, 5=neutral, 10=execution-focused, metrics-driven)
5. capital_discipline (1=dilutive/wasteful, 5=neutral, 10=ROIC-focused, smart buybacks)

FINANCIAL SIGNALS:
6. liquidity_health (1=going concern, 5=adequate, 10=strong/improving)
7. growth_quality (1=declining, 5=flat, 10=organic growth with good unit economics)
8. profitability_trend (1=worsening losses, 5=stable, 10=margin expansion)

For each dimension provide:
- score (1-10)
- evidence (one short quote, max 12 words)

Also identify:
- top_3_positives: Best things about this company (be specific)
- top_3_concerns: Biggest risks (be specific)
- composite_score: Overall score from 1 to 100 (NOT 1-10. Calculate as average of all dimension scores multiplied by 10)
- gem_potential: "high", "medium", "low", or "avoid"
- gem_reasoning: One sentence explaining gem potential

Respond with JSON:
{{
  "ticker": "{ticker}",
  "dimensions": {{
    "confidence": {{"score": N, "evidence": "..."}},
    "transparency": {{"score": N, "evidence": "..."}},
    "consistency": {{"score": N, "evidence": "..."}},
    "operational_focus": {{"score": N, "evidence": "..."}},
    "capital_discipline": {{"score": N, "evidence": "..."}},
    "liquidity_health": {{"score": N, "evidence": "..."}},
    "growth_quality": {{"score": N, "evidence": "..."}},
    "profitability_trend": {{"score": N, "evidence": "..."}}
  }},
  "composite_score": 1-100,
  "top_3_positives": ["...", "...", "..."],
  "top_3_concerns": ["...", "...", "..."],
  "gem_potential": "high|medium|low|avoid",
  "gem_reasoning": "..."
}}

FILING ({form_type}) for {ticker} - {company_name}:
{filing_text}"""


# ============================================================================
# TIER 1: TRAJECTORY COMPARISON PROMPT
# ============================================================================

TIER1_TRAJECTORY_PROMPT = """Compare these two SEC filings for {ticker} to assess IMPROVEMENT trajectory.

You're looking for companies getting BETTER - not just avoiding getting worse.

For each dimension, score BOTH filings (1-10) and determine trajectory:

DIMENSIONS:
1. confidence - Management's tone (promotional=bad, quietly assured=good)
2. transparency - Specificity and honesty
3. operational_focus - Execution vs excuses
4. capital_discipline - Smart capital allocation
5. liquidity_health - Cash and debt position
6. growth_quality - Revenue trend and quality
7. profitability_trend - Margin direction

TRAJECTORY SCORING:
- "strongly_improving" = +3 or more points
- "improving" = +1 to +2 points  
- "stable" = no change
- "declining" = -1 to -2 points
- "deteriorating" = -3 or more points

Look specifically for:
- Word substitutions ("will" → "may", "expect" → "hope")
- Metrics no longer highlighted
- New blame language
- OR the opposite: increasing confidence, better metrics, less hedging

Respond with JSON:
{{
  "ticker": "{ticker}",
  "prior_filing_date": "{prior_date}",
  "current_filing_date": "{current_date}",
  "dimensions": {{
    "confidence": {{
      "prior_score": N,
      "current_score": N,
      "trajectory": "strongly_improving|improving|stable|declining|deteriorating",
      "key_shift": "what changed (max 15 words)"
    }},
    ... (all 7 dimensions)
  }},
  "overall_trajectory": "strongly_improving|improving|stable|declining|deteriorating",
  "trajectory_score": N,
  "improvement_signals": ["specific positive change 1", "...", "..."],
  "warning_signals": ["specific negative change 1", "...", "..."],
  "notable_quote_current": "most telling quote from current filing",
  "notable_quote_prior": "comparable quote from prior filing",
  "gem_potential": "high|medium|low|avoid",
  "summary": "2-3 sentence trajectory summary"
}}

PRIOR FILING ({prior_form_type}, {prior_date}):
{prior_text}

CURRENT FILING ({current_form_type}, {current_date}):
{current_text}"""


# ============================================================================
# TIER 2: DEEP ANALYSIS PROMPT (Claude Haiku) - Forensic-Enhanced
# ============================================================================

TIER2_DEEP_ANALYSIS_PROMPT = """Perform deep qualitative analysis on this potential hidden gem.

This company scored well in initial screening. You have been provided with:
1. The SEC filing text
2. QUANTITATIVE forensic language metrics (hedging ratios, confidence patterns, etc.)
3. Filing-over-filing DIFF DATA showing what changed vs the prior filing (if available)
4. INSIDER & INSTITUTIONAL ACTIVITY data (Form 4 insider buys/sells, institutional filings)
5. FINANCIAL INFLECTION ANALYSIS from XBRL data (revenue trends, margin shifts, profitability crossings)

YOUR JOB IS NOT TO SUMMARIZE. Your job is to INTERPRET all signals holistically and explain
what they mean for this company's investment potential. Focus on:

FORENSIC INTERPRETATION:
- What does the confidence-to-hedge ratio tell us? Is management genuinely confident?
- Is the specificity score high (they give real numbers) or low (vague hand-waving)?
- Is blame language present? Who are they blaming and is it legitimate?
- Are the forensic scores consistent with what management claims?

FILING DIFF INTERPRETATION (if prior filing data provided):
- What risk factors were ADDED or REMOVED? What does each change signal?
- What new statements appeared in MD&A? Are they positive or concerning?
- What was REMOVED from MD&A? Did they stop talking about something important?
- Did hedging increase or decrease? Did specificity change?
- Do the metric shifts tell a different story than management's narrative?

INSIDER ACTIVITY INTERPRETATION (if insider data provided):
- Are insiders buying with their own money? This is the strongest bullish signal for microcaps.
- Is there CLUSTER buying (multiple insiders buying in a short window)? Even stronger signal.
- Are C-suite executives (CEO/CFO) buying? They have the most information.
- Is there an activist investor (SC 13D)? What does this mean for the stock?
- Is there net selling? Could be routine or could signal problems — context matters.
- Does insider behavior CONFIRM or CONTRADICT management's optimistic language in the filing?

FINANCIAL INFLECTION INTERPRETATION (if XBRL data provided):
- Is revenue ACCELERATING (growth rate increasing)? This is the most important forward indicator.
- Is the company approaching profitability for the first time? This is a major re-rating catalyst.
- Are margins expanding? This suggests pricing power or operating leverage kicking in.
- Is there heavy dilution? This can destroy shareholder value even if the business improves.
- Is cash runway adequate? Low runway means potential dilutive financing ahead.
- Does the financial trajectory support or contradict the filing's qualitative narrative?

DEEP QUALITATIVE ANALYSIS:
1. MANAGEMENT AUTHENTICITY
   - Does forensic data CONFIRM or CONTRADICT management's narrative?
   - Are they getting more or less specific over time?
   - Is guidance conservative (sandbagging) or aggressive?

2. COMPETITIVE POSITION
   - What's the actual moat (if any)?
   - Pricing power evidence in the language?

3. CAPITAL ALLOCATION QUALITY
   - Are they investing at good returns?
   - Buyback timing and discipline

4. HIDDEN STRENGTHS
   - Understated positives buried in footnotes
   - New positive language that wasn't in prior filing
   - Removed risk factors suggesting resolved concerns

5. SUBTLE RED FLAGS
   - New hedging or blame language
   - Decreased specificity (hiding something?)
   - Related party transactions
   - Revenue recognition nuances

Respond with JSON:
{{
  "ticker": "{ticker}",
  "deep_analysis": {{
    "management_authenticity": {{
      "score": 1-10,
      "genuine_confidence_level": "high|medium|low",
      "guidance_style": "conservative|balanced|aggressive",
      "forensic_confirmation": "confirmed|mixed|contradicted",
      "evidence": ["quote1", "quote2"],
      "assessment": "2-3 sentences interpreting the forensic data"
    }},
    "competitive_position": {{
      "score": 1-10,
      "moat_type": "none|weak|moderate|strong",
      "share_trend": "gaining|stable|losing",
      "evidence": ["quote1", "quote2"],
      "assessment": "2-3 sentences"
    }},
    "capital_allocation": {{
      "score": 1-10,
      "quality": "poor|mediocre|good|excellent",
      "evidence": ["quote1", "quote2"],
      "assessment": "2-3 sentences"
    }},
    "hidden_strengths": [
      {{"finding": "...", "significance": "high|medium|low", "evidence": "..."}}
    ],
    "subtle_red_flags": [
      {{"finding": "...", "severity": "high|medium|low", "evidence": "..."}}
    ]
  }},
  "filing_diff_insights": {{
    "tone_shift": "improving|stable|deteriorating",
    "key_changes": ["most significant change 1", "change 2", "change 3"],
    "what_they_stopped_saying": ["removed statement or topic 1"],
    "what_they_started_saying": ["new statement or topic 1"],
    "red_flag_changes": ["any concerning additions/removals"],
    "positive_changes": ["any encouraging additions/removals"]
  }},
  "insider_signal": {{
    "assessment": "2-3 sentences interpreting insider activity in context of the filing",
    "confirms_thesis": true/false,
    "key_observation": "most important insider activity finding"
  }},
  "financial_trajectory": {{
    "assessment": "2-3 sentences interpreting financial inflection data",
    "revenue_momentum": "accelerating|stable|decelerating|declining",
    "profitability_trajectory": "improving|stable|worsening",
    "key_inflection": "most significant financial inflection detected (or 'none')",
    "dilution_risk": "none|low|moderate|high"
  }},
  "investment_thesis": {{
    "bull_case": "2-3 sentences grounded in specific forensic, insider, AND financial evidence",
    "bear_case": "2-3 sentences grounded in specific forensic, insider, AND financial evidence",
    "key_catalyst": "what could unlock value (reference insider/financial data if relevant)",
    "key_risk": "what could go wrong (reference dilution, cash runway, insider selling if relevant)"
  }},
  "final_gem_score": 1-100,
  "conviction_level": "high|medium|low",
  "recommendation": "strong_buy|buy|hold|avoid",
  "one_liner": "One sentence pitch referencing specific evidence"
}}

{forensic_data}

{insider_data}

{inflection_data}

FILING ({form_type}) for {ticker} - {company_name}:
{filing_text}"""


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_filing_tier1(
    ticker: str,
    company_name: str,
    filing_text: str,
    form_type: str = "10-K",
    model: str = DEFAULT_TIER1_MODEL,
) -> Optional[Dict[str, Any]]:
    """
    Tier 1 screening analysis using GPT-4o-mini.
    Fast and cheap - run on all filings.
    """
    prompt = TIER1_SCREENING_PROMPT.format(
        ticker=ticker,
        company_name=company_name,
        form_type=form_type,
        filing_text=_truncate_text(filing_text, 80000),
    )

    raw = _call_llm(prompt, SYSTEM_PROMPT_GEM_FINDER, model)
    result = _parse_json_response(raw)

    if not result:
        preview = (raw or "")[:200]
        raise RuntimeError(f"JSON parse failed. Raw response: {preview}...")

    result["analysis_tier"] = 1
    result["model_used"] = model
    result["analyzed_at"] = datetime.now().isoformat()

    # Calculate composite if not provided
    if "composite_score" not in result and "dimensions" in result:
        scores = [d.get("score", 5) for d in result["dimensions"].values()]
        result["composite_score"] = round(sum(scores) / len(scores) * 10, 1) if scores else 50

    # Normalize composite_score to 0-100 scale
    # LLM sometimes returns 1-10 despite being asked for 1-100
    cs = result.get("composite_score")
    if cs is not None and cs <= 10:
        result["composite_score"] = round(cs * 10, 1)

    return result


def analyze_trajectory_tier1(
    ticker: str,
    current_text: str,
    prior_text: str,
    current_date: str,
    prior_date: str,
    current_form_type: str = "10-K",
    prior_form_type: str = "10-K",
    model: str = DEFAULT_TIER1_MODEL,
) -> Optional[Dict[str, Any]]:
    """
    Tier 1 trajectory analysis - compare two filings.
    Identifies improving vs declining companies.
    """
    prompt = TIER1_TRAJECTORY_PROMPT.format(
        ticker=ticker,
        current_text=_truncate_text(current_text, 50000),
        prior_text=_truncate_text(prior_text, 50000),
        current_date=current_date,
        prior_date=prior_date,
        current_form_type=current_form_type,
        prior_form_type=prior_form_type,
    )

    raw = _call_llm(prompt, SYSTEM_PROMPT_GEM_FINDER, model)
    result = _parse_json_response(raw)

    if result:
        result["analysis_tier"] = 1
        result["analysis_type"] = "trajectory"
        result["model_used"] = model
        result["analyzed_at"] = datetime.now().isoformat()

        # Calculate trajectory score if not provided
        if "trajectory_score" not in result and "dimensions" in result:
            traj_map = {
                "strongly_improving": 2,
                "improving": 1,
                "stable": 0,
                "declining": -1,
                "deteriorating": -2,
            }
            traj_scores = [
                traj_map.get(d.get("trajectory", "stable"), 0)
                for d in result["dimensions"].values()
            ]
            result["trajectory_score"] = sum(traj_scores)

    return result


def analyze_filing_tier2(
    ticker: str,
    company_name: str,
    filing_text: str,
    form_type: str = "10-K",
    model: str = DEFAULT_TIER2_MODEL,
    forensic_data: str = "",
    insider_data: str = "",
    inflection_data: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Tier 2 deep analysis using Claude Haiku.
    Run on top 25% from Tier 1 screening.

    Args:
        forensic_data: Pre-formatted forensic metrics + diff data string
        insider_data: Pre-formatted insider transaction analysis string
        inflection_data: Pre-formatted financial inflection analysis string
    """
    if not forensic_data:
        forensic_data = "(No forensic data available)"
    if not insider_data:
        insider_data = "(No insider activity data available)"
    if not inflection_data:
        inflection_data = "(No financial inflection data available)"

    prompt = TIER2_DEEP_ANALYSIS_PROMPT.format(
        ticker=ticker,
        company_name=company_name,
        form_type=form_type,
        filing_text=_truncate_text(filing_text, 90000),
        forensic_data=forensic_data,
        insider_data=insider_data,
        inflection_data=inflection_data,
    )

    raw = _call_llm(prompt, SYSTEM_PROMPT_DEEP_ANALYSIS, model, max_tokens=4096)
    result = _parse_json_response(raw)

    if not result:
        preview = (raw or "<no response>")[:300]
        logger.error(f"Tier 2 {ticker}: JSON parse failed. len={len(raw or '')} preview={preview}")
        raise RuntimeError(f"JSON parse failed (response len={len(raw or '')}). Preview: {preview}...")

    result["analysis_tier"] = 2
    result["model_used"] = model
    result["analyzed_at"] = datetime.now().isoformat()

    # Normalize final_gem_score to 0-100 scale (safety check)
    fgs = result.get("final_gem_score")
    if fgs is not None and fgs <= 10:
        result["final_gem_score"] = round(fgs * 10)

    return result


# ============================================================================
# BATCH ANALYSIS ORCHESTRATOR
# ============================================================================

def run_tiered_analysis(
    filings: List[Dict],
    tier1_model: str = DEFAULT_TIER1_MODEL,
    tier2_model: str = DEFAULT_TIER2_MODEL,
    tier2_percentile: float = 0.25,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Run two-tier analysis on a list of filings.

    Args:
        filings: List of dicts with keys: ticker, company_name, text, form_type, filing_date
        tier1_model: Model for initial screening
        tier2_model: Model for deep analysis
        tier2_percentile: Top X% to analyze with Tier 2 (0.25 = top 25%)
        progress_callback: fn(current, total, ticker, tier, status)

    Returns:
        Dict with all results and summary stats
    """
    results = {
        "tier1_results": [],
        "tier2_results": [],
        "errors": [],
        "stats": {
            "total_filings": len(filings),
            "tier1_completed": 0,
            "tier2_completed": 0,
            "tier2_eligible": 0,
        },
    }

    total = len(filings)

    # ---- TIER 1: Screen all filings ----
    if progress_callback:
        progress_callback(0, total, "", 1, "Starting Tier 1 screening...")

    for i, filing in enumerate(filings):
        ticker = filing.get("ticker", "UNKNOWN")
        try:
            t1_result = analyze_filing_tier1(
                ticker=ticker,
                company_name=filing.get("company_name", ""),
                filing_text=filing.get("text", ""),
                form_type=filing.get("form_type", "10-K"),
                model=tier1_model,
            )
            if t1_result:
                t1_result["filing_date"] = filing.get("filing_date", "")
                t1_result["accession_number"] = filing.get("accession_number", "")
                results["tier1_results"].append(t1_result)
                results["stats"]["tier1_completed"] += 1
            else:
                results["errors"].append(f"{ticker}: Tier 1 analysis returned None")
        except Exception as e:
            results["errors"].append(f"{ticker}: Tier 1 error - {str(e)}")
            logger.error(f"Tier 1 error for {ticker}: {e}")

        if progress_callback:
            progress_callback(i + 1, total, ticker, 1, "Tier 1 screening")

    # ---- Sort by composite score and select top percentile for Tier 2 ----
    tier1_sorted = sorted(
        results["tier1_results"],
        key=lambda x: x.get("composite_score", 0),
        reverse=True,
    )

    tier2_count = max(1, int(len(tier1_sorted) * tier2_percentile))
    tier2_candidates = tier1_sorted[:tier2_count]
    results["stats"]["tier2_eligible"] = tier2_count

    # Build lookup for filing text
    filing_lookup = {f.get("ticker"): f for f in filings}

    # ---- TIER 2: Deep analysis on top candidates ----
    if progress_callback:
        progress_callback(0, tier2_count, "", 2, "Starting Tier 2 deep analysis...")

    for i, t1_result in enumerate(tier2_candidates):
        ticker = t1_result.get("ticker", "UNKNOWN")
        filing = filing_lookup.get(ticker, {})

        try:
            t2_result = analyze_filing_tier2(
                ticker=ticker,
                company_name=filing.get("company_name", ""),
                filing_text=filing.get("text", ""),
                form_type=filing.get("form_type", "10-K"),
                model=tier2_model,
            )
            if t2_result:
                # Merge Tier 1 data
                t2_result["tier1_composite_score"] = t1_result.get("composite_score")
                t2_result["tier1_gem_potential"] = t1_result.get("gem_potential")
                t2_result["filing_date"] = t1_result.get("filing_date", "")
                t2_result["accession_number"] = t1_result.get("accession_number", "")
                results["tier2_results"].append(t2_result)
                results["stats"]["tier2_completed"] += 1
            else:
                results["errors"].append(f"{ticker}: Tier 2 analysis returned None")
        except Exception as e:
            results["errors"].append(f"{ticker}: Tier 2 error - {str(e)}")
            logger.error(f"Tier 2 error for {ticker}: {e}")

        if progress_callback:
            progress_callback(i + 1, tier2_count, ticker, 2, "Tier 2 deep analysis")

    # ---- Final ranking ----
    results["tier2_results"] = sorted(
        results["tier2_results"],
        key=lambda x: x.get("final_gem_score", 0),
        reverse=True,
    )

    return results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_api_status() -> Dict[str, Any]:
    """Check which APIs are available."""
    status = {
        "openai": {"available": False, "has_key": False, "has_sdk": False},
        "anthropic": {"available": False, "has_key": False, "has_sdk": False},
    }

    # OpenAI
    status["openai"]["has_key"] = bool(os.environ.get("OPENAI_API_KEY"))
    try:
        import openai
        status["openai"]["has_sdk"] = True
    except ImportError:
        pass
    status["openai"]["available"] = (
        status["openai"]["has_key"] and status["openai"]["has_sdk"]
    )

    # Anthropic
    status["anthropic"]["has_key"] = bool(os.environ.get("ANTHROPIC_API_KEY"))
    try:
        import anthropic
        status["anthropic"]["has_sdk"] = True
    except ImportError:
        pass
    status["anthropic"]["available"] = (
        status["anthropic"]["has_key"] and status["anthropic"]["has_sdk"]
    )

    return status


def estimate_cost(num_filings: int, tier2_percentile: float = 0.25) -> Dict[str, float]:
    """Estimate cost for analyzing filings."""
    # Assumptions
    avg_input_tokens = 30000
    avg_output_tokens = 1500
    tier2_count = int(num_filings * tier2_percentile)

    tier1_input = num_filings * avg_input_tokens / 1_000_000
    tier1_output = num_filings * avg_output_tokens / 1_000_000
    tier2_input = tier2_count * avg_input_tokens / 1_000_000
    tier2_output = tier2_count * avg_output_tokens / 1_000_000

    t1_config = MODELS[DEFAULT_TIER1_MODEL]
    t2_config = MODELS[DEFAULT_TIER2_MODEL]

    tier1_cost = (tier1_input * t1_config["input_cost_per_1m"] +
                  tier1_output * t1_config["output_cost_per_1m"])
    tier2_cost = (tier2_input * t2_config["input_cost_per_1m"] +
                  tier2_output * t2_config["output_cost_per_1m"])

    return {
        "tier1_filings": num_filings,
        "tier2_filings": tier2_count,
        "tier1_cost": round(tier1_cost, 2),
        "tier2_cost": round(tier2_cost, 2),
        "total_cost": round(tier1_cost + tier2_cost, 2),
        "tier1_model": DEFAULT_TIER1_MODEL,
        "tier2_model": DEFAULT_TIER2_MODEL,
    }


# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

def is_llm_available() -> bool:
    """Check if any LLM API is available."""
    status = check_api_status()
    return status["openai"]["available"] or status["anthropic"]["available"]


def llm_analyze_sentiment(current_text: str, prior_text: Optional[str] = None) -> Optional[Dict]:
    """Legacy function - now uses Tier 1 analysis."""
    try:
        result = analyze_filing_tier1(
            ticker="UNKNOWN",
            company_name="Unknown",
            filing_text=current_text,
            form_type="10-K",
        )
    except Exception as e:
        logger.error(f"llm_analyze_sentiment failed: {e}")
        return None
    if result:
        # Convert to legacy format
        return {
            "sentiment_results": [
                {
                    "category": k,
                    "current_level": "CONFIDENT" if v.get("score", 5) >= 7 else
                                    "NEUTRAL" if v.get("score", 5) >= 4 else "CAUTIOUS",
                    "score_impact": (v.get("score", 5) - 5) * 2,
                    "key_phrases": [v.get("evidence", "")],
                }
                for k, v in result.get("dimensions", {}).items()
            ],
            "sentiment_meta": {
                "overall_trajectory": result.get("gem_potential", "medium"),
            },
        }
    return None


def llm_analyze_flags(current_text: str, prior_text: Optional[str] = None) -> Optional[Dict]:
    """Legacy function - now uses Tier 1 analysis."""
    try:
        result = analyze_filing_tier1(
            ticker="UNKNOWN",
            company_name="Unknown",
            filing_text=current_text,
            form_type="10-K",
        )
    except Exception as e:
        logger.error(f"llm_analyze_flags failed: {e}")
        return None
    if result:
        flags = []
        for concern in result.get("top_3_concerns", []):
            flags.append({
                "rule_id": "llm_concern",
                "title": concern,
                "signal_type": "yellow_flag",
                "risk_level": "MODERATE",
                "score_impact": -5,
                "source": "llm",
            })
        for positive in result.get("top_3_positives", []):
            flags.append({
                "rule_id": "llm_positive",
                "title": positive,
                "signal_type": "green_flag",
                "risk_level": "LOW",
                "score_impact": 5,
                "source": "llm",
            })
        return {
            "flags": flags,
            "overall_assessment": result.get("gem_reasoning", ""),
        }
    return None
