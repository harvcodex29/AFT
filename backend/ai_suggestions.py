import os
import json
import logging
import urllib.request
import urllib.error
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_env():
    """Load .env file if present (no external dependency)."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())

_load_env()

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    "AIzaSyAHDeMWMvVgUcNiwVW9rWYCzLjbToUrDnQ"
)
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)
MAX_OUTPUT_TOKENS = 1024
TEMPERATURE       = 0.4


def _build_prompt(analysis: dict, forecast: dict, profile: dict | None = None) -> str:
    summary      = analysis.get("summary", {})
    financials   = analysis.get("financial_summary", {})
    alerts       = analysis.get("overspending_alerts", [])
    overall_pred = forecast.get("overall_prediction", {})
    top_cats     = forecast.get("top_predicted_categories", [])

    income       = financials.get("total_income", 0)
    net_savings  = financials.get("net_savings", 0)
    savings_rate = financials.get("savings_rate", 0)
    num_months   = summary.get("num_months_analysed", 1)

    monthly_income  = round(income / num_months, 0) if num_months else 0
    monthly_savings = round(net_savings / num_months, 0) if num_months else 0

    top_cat_text = "\n".join(
        f"  • {c['category']}: ₹{c.get('total', 0):,.0f} total, "
        f"₹{c.get('monthly_avg', 0):,.0f}/month avg"
        for c in summary.get("top_categories", [])
    )

    alert_text = "\n".join(
        f"  • {a['month']} | {a['category']}: ₹{a['amount']:,.0f} "
        f"({a['spike_ratio']:.1f}× avg ₹{a['average']:,.0f})"
        for a in alerts[:5]
    ) or "  None detected."

    pred_cat_text = "\n".join(
        f"  • {c['category']}: ₹{c['predicted_amount']:,.0f}"
        for c in top_cats[:5]
    )

    # Behaviour context
    behaviour_block = ""
    if profile:
        behaviour_block = f"""
--- BEHAVIOUR PROFILE ---
Spender type   : {profile.get('spender_type', 'unknown')} — {profile.get('spender_rationale', '')}
Trend          : {profile.get('trend_direction', 'stable')} (slope ₹{profile.get('trend_slope', 0):,.0f}/month)
Risk score     : {profile.get('risk_score', {}).get('score', 'N/A')}/100  Grade {profile.get('risk_score', {}).get('grade', '?')}
Patterns found : {', '.join(p['name'] for p in profile.get('patterns', [])) or 'None'}
"""

    prompt = f"""You are a friendly and pragmatic personal finance advisor.
A user has shared {num_months} months of bank data. Here is the full picture:

--- INCOME & SAVINGS ---
Monthly income  : ₹{monthly_income:,.0f}
Monthly savings : ₹{monthly_savings:,.0f}
Savings rate    : {savings_rate:.1f}%

--- SPENDING SUMMARY ---
Total spent     : ₹{summary.get('total_spent', 0):,.0f}
Monthly average : ₹{summary.get('overall_monthly_avg', 0):,.0f}

Top expense categories:
{top_cat_text}

--- OVERSPENDING ALERTS ---
{alert_text}
{behaviour_block}
--- NEXT-MONTH FORECAST ---
Predicted spend : ₹{overall_pred.get('predicted_amount', 0):,.0f}
Confidence band : ₹{overall_pred.get('confidence_band', {}).get('low', 0):,.0f} – ₹{overall_pred.get('confidence_band', {}).get('high', 0):,.0f}

Top predicted categories:
{pred_cat_text}

--- YOUR TASK ---
Write a concise, actionable financial advice report with:
1. **Overall Assessment** – 2-3 sentences on spending health and savings rate.
2. **Top 3 Saving Opportunities** – specific steps with estimated monthly savings in ₹.
3. **Category-Specific Tips** – 1-2 tips for the top 2 overspending categories.
4. **Next Month Action Plan** – 3 bullet-point goals tied to the forecast.
5. **Positive Reinforcement** – one thing the user is doing well.

Use ₹ (Indian Rupee). Encouraging tone, not preachy. Under 400 words.
"""
    return prompt


def _call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. Add it to your .env file or environment."
        )

    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": MAX_OUTPUT_TOKENS, "temperature": TEMPERATURE},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err = exc.read().decode("utf-8", errors="replace")
        logger.error("Gemini HTTP %s: %s", exc.code, err)
        raise RuntimeError(f"Gemini API error {exc.code}: {err}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error: {exc.reason}") from exc

    try:
        return body["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected Gemini response: {body}") from exc


def generate_suggestions(
    analysis: dict,
    forecast: dict,
    profile: dict | None = None
) -> dict:
    """
    Generate AI suggestions using full context:
    spending, income, savings, behaviour profile.
    """

    try:
        prompt = _build_prompt(analysis, forecast, profile)
        suggestions = _call_gemini(prompt)

        return {
            "suggestions": suggestions,
            "model": "gemini-2.0-flash",
            "status": "success",
            "error": None,
        }

    except Exception as exc:
        logger.exception("Suggestions failed: %s", exc)

        fallback_text = """
1. Track weekly expenses to improve visibility.
2. Set category-wise monthly budgets.
3. Reduce avoidable discretionary spending.
4. Maintain an emergency savings fund.
5. Review recurring subscriptions regularly.
"""

        return {
            "suggestions": fallback_text,
            "model": "gemini-2.0-flash",
            "status": "fallback",
            "error": str(exc),
        }


def quick_tip(category: str, monthly_avg: float) -> dict:
    prompt = (
        f"Give one concise (≤ 2 sentences) money-saving tip for someone spending "
        f"₹{monthly_avg:,.0f}/month on '{category}'. Specific, actionable, encouraging."
    )
    try:
        return {"category": category, "tip": _call_gemini(prompt).strip(), "status": "success"}
    except Exception as exc:
        return {"category": category, "tip": None, "status": "error", "error": str(exc)}
