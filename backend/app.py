"""
Flask REST API – Financial Analysis & Intervention System  v2.1

Endpoint flow
-------------
POST /api/upload         – Upload transactions (triggers full pipeline)
GET  /api/analyze        – Expense analysis + behaviour profile
GET  /api/predict        – Next-month spending forecast
GET  /api/suggest        – AI suggestions via Gemini (full context)
GET  /api/dashboard      – Consolidated dashboard with financial_summary

Supporting
----------
POST /api/demo-load
POST  /api/users         – Create user
GET   /api/users/<id>    – Fetch user
PATCH /api/users/<id>    – Update budget
GET   /api/alerts        – Fetch alerts
PATCH /api/alerts/<id>   – Mark alert read
GET   /api/health
"""

import uuid
import logging
import datetime as _dt
from flask import Flask, request, jsonify, abort
from flask_cors import CORS

import os
import database as db
from data_processing import parse_transactions, full_analysis
from prediction       import spending_forecast
from ai_suggestions   import generate_suggestions
from behavior_engine  import build_behavior_profile, generate_alerts

app = Flask(__name__)
CORS(app)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)

db.init_db()


def _ok(payload: dict, status: int = 200):
    return jsonify({"status": "success", **payload}), status


def _err(message: str, status: int = 400):
    return jsonify({"status": "error", "message": message}), status


def _require_user(user_id: str) -> dict:
    user = db.get_user(user_id)
    if not user:
        abort(404, description=f"User '{user_id}' not found. POST /api/users to create one.")
    return user


def _require_transactions(user_id: str) -> list[dict]:
    txns = db.get_transactions(user_id)
    if not txns:
        abort(400, description="No transactions found. POST /api/upload first.")
    return txns


def _run_full_pipeline(user_id: str, user: dict,
                       transactions: list[dict]) -> tuple[dict, dict, dict, list[dict]]:
    analysis = full_analysis(transactions)
    db.save_analysis(user_id, analysis)

    forecast = spending_forecast(monthly_totals=analysis["monthly_totals"],
                                 transactions=transactions)
    db.save_prediction(user_id, forecast)

    profile = build_behavior_profile(user_id, transactions, analysis, forecast, user)
    db.save_behavior_profile(user_id, profile)

    stale = ["budget_threshold", "pace_overspend", "category_spike",
             "category_rising", "risk_score", "forecast_over_budget",
             "spender_nudge", "low_savings_rate"] + \
            [f"pattern_{p['id']}" for p in profile.get("patterns", [])]
    for atype in stale:
        db.clear_old_alerts(user_id, atype)

    alert_dicts = generate_alerts(user_id, transactions, analysis, forecast, profile, user)
    for a in alert_dicts:
        db.save_alert(user_id, a["type"], a["severity"],
                      a["title"], a["message"], a.get("metadata"))

    return analysis, forecast, profile, alert_dicts


# Health


@app.get("/api/health")
def health():
    return _ok({"message": "Financial Analysis API is running.", "version": "2.1"})


# Users

@app.post("/api/users")
def create_user():
    body    = request.get_json(silent=True) or {}
    user_id = str(uuid.uuid4())
    user    = db.upsert_user(
        user_id        = user_id,
        name           = body.get("name", "User"),
        email          = body.get("email"),
        monthly_budget = float(body.get("monthly_budget", 50000)),
    )
    return _ok({"user": user}, 201)


@app.get("/api/users/<user_id>")
def get_user(user_id: str):
    return _ok({"user": _require_user(user_id)})


@app.patch("/api/users/<user_id>")
def update_user(user_id: str):
    _require_user(user_id)

    body = request.get_json(silent=True) or {}

    try:
        budget = float(body.get("monthly_budget", 0))
    except (TypeError, ValueError):
        return _err("'monthly_budget' must be numeric.")

    if budget <= 0:
        return _err("'monthly_budget' must be positive.")

    db.update_budget(user_id, budget)

    return _ok({
        "message": "Budget updated.",
        "monthly_budget": budget
    })


# 1. POST /api/upload


@app.post("/api/upload")
def upload():
    """
    Upload bank transactions and auto-trigger the full pipeline.

    Body: { "user_id": str, "transactions": [ {date, description, amount, type?}, ... ] }
    """
    body = request.get_json(silent=True)
    if not body:
        return _err("Request body must be valid JSON.")

    user_id = body.get("user_id", "").strip()
    raw     = body.get("transactions", [])

    if not user_id:
        return _err("'user_id' is required. Create a user at POST /api/users first.")
    if not isinstance(raw, list) or not raw:
        return _err("'transactions' must be a non-empty list.")

    user = _require_user(user_id)

    try:
        transactions = parse_transactions(raw)
    except ValueError as exc:
        return _err(str(exc))

    total_stored = db.save_transactions(user_id, transactions)
    all_txns     = db.get_transactions(user_id)
    months_found = sorted({t["month"] for t in all_txns})

    analysis, forecast, profile, alert_dicts = _run_full_pipeline(user_id, user, all_txns)

    return _ok({
        "user_id":           user_id,
        "transaction_count": total_stored,
        "months_found":      months_found,
        "alerts_generated":  len(alert_dicts),
        "risk_score":        profile.get("risk_score"),
        "spender_type":      profile.get("spender_type"),
        "financial_summary": analysis.get("financial_summary"),
    }, 201)

@app.post("/api/demo-load")
def demo_load():
    import json

    body = request.get_json(silent=True) or {}

    user_id = body.get("user_id")

    if not user_id:
        return _err("'user_id' required")

    user = db.get_user(user_id)

    if not user:
        user = db.upsert_user(
            user_id=user_id,
            name="Demo User",
            email=None,
            monthly_budget=50000
        )

    try:
        file_path = os.path.join(os.path.dirname(__file__), "sample_data.json")

        with open(file_path, "r") as f:
            data = json.load(f)

    except FileNotFoundError:
        return _err("sample_data.json not found")

    transactions = parse_transactions(data["transactions"])

    # NEW LINE → clear old demo data first
    db.clear_transactions(user_id)

    db.save_transactions(user_id, transactions)

    all_txns = db.get_transactions(user_id)

    analysis, forecast, profile, alerts = _run_full_pipeline(
        user_id,
        user,
        all_txns
    )

    return _ok({
        "message": "Sample data loaded and processed",
        "months_loaded": sorted({t["month"] for t in all_txns}),
        "alerts_generated": len(alerts)
    })


# 2. GET /api/analyze


@app.get("/api/analyze")
def analyze():
    """Query params: ?user_id=<uuid>&refresh=false"""
    user_id = request.args.get("user_id", "").strip()
    refresh  = request.args.get("refresh", "false").lower() == "true"

    if not user_id:
        return _err("Query param 'user_id' is required.")

    user         = _require_user(user_id)
    transactions = _require_transactions(user_id)

    analysis = None if refresh else db.get_latest_analysis(user_id)
    profile  = None if refresh else db.get_latest_behavior_profile(user_id)

    if analysis is None or profile is None:
        analysis, _, profile, _ = _run_full_pipeline(user_id, user, transactions)

    return _ok({
        "user_id":          user_id,
        "analysis":         analysis,
        "behavior_profile": profile,
    })


# 3. GET /api/predict

@app.get("/api/predict")
def predict():
    """Query params: ?user_id=<uuid>&blend_alpha=0.6&refresh=false"""
    user_id     = request.args.get("user_id", "").strip()
    blend_alpha = float(request.args.get("blend_alpha", 0.6))
    refresh      = request.args.get("refresh", "false").lower() == "true"

    if not user_id:
        return _err("Query param 'user_id' is required.")

    user         = _require_user(user_id)
    transactions = _require_transactions(user_id)

    forecast = None if refresh else db.get_latest_prediction(user_id)
    if forecast is None:
        analysis = db.get_latest_analysis(user_id) or full_analysis(transactions)
        forecast = spending_forecast(monthly_totals=analysis["monthly_totals"],
                                     transactions=transactions, blend_alpha=blend_alpha)
        db.save_prediction(user_id, forecast)

    monthly_budget = user.get("monthly_budget", 50000)
    predicted      = forecast.get("overall_prediction", {}).get("predicted_amount", 0)
    budget_gap     = monthly_budget - predicted

    return _ok({
        "user_id":  user_id,
        "forecast": forecast,
        "budget_context": {
            "monthly_budget":  monthly_budget,
            "predicted_spend": predicted,
            "budget_gap":      round(budget_gap, 2),
            "status":          "over_budget" if budget_gap < 0 else "within_budget",
        },
    })


# 4. GET /api/suggest

@app.get("/api/suggest")
def suggest():
    user_id = request.args.get("user_id", "").strip()

    if not user_id:
        return _err("Query param 'user_id' is required.")

    user         = _require_user(user_id)
    transactions = _require_transactions(user_id)

    analysis = db.get_latest_analysis(user_id)
    forecast = db.get_latest_prediction(user_id)
    profile  = db.get_latest_behavior_profile(user_id)

    if not analysis or not forecast or not profile:
        analysis, forecast, profile, _ = _run_full_pipeline(
            user_id, user, transactions
        )

    try:
        result = generate_suggestions(
            analysis,
            forecast,
            profile
        )

    except Exception as exc:
        logger.exception("Gemini API failed: %s", exc)

        result = {
            "suggestions": [
                "Track weekly spending to improve awareness.",
                "Reduce unnecessary category expenses this month.",
                "Set a fixed monthly savings target.",
                "Review recurring subscriptions regularly."
            ],
            "source": "fallback"
        }

    result["behavior_context"] = {
        "spender_type": profile.get("spender_type"),
        "risk_score": profile.get("risk_score"),
        "patterns_found": len(profile.get("patterns", [])),
        "trend": profile.get("trend_direction"),
        "savings_rate": profile.get("savings_rate"),
    }

    return _ok({"user_id": user_id, **result})


# 5. GET /api/dashboard

@app.get("/api/dashboard")
def dashboard():
    """
    Full dashboard payload — one call, everything the UI needs.
    Query params: ?user_id=<uuid>
    """
    user_id = request.args.get("user_id", "").strip()
    if not user_id:
        return _err("Query param 'user_id' is required.")

    user         = _require_user(user_id)
    transactions = _require_transactions(user_id)

    analysis = db.get_latest_analysis(user_id)
    forecast = db.get_latest_prediction(user_id)
    profile  = db.get_latest_behavior_profile(user_id)

    if not analysis or not forecast or not profile:
        analysis, forecast, profile, _ = _run_full_pipeline(user_id, user, transactions)

    # Current-month tracker
    now           = _dt.datetime.now()
    current_month = now.strftime("%Y-%m")
    next_month_1  = (_dt.date(now.year + 1, 1, 1) if now.month == 12
                     else _dt.date(now.year, now.month + 1, 1))
    days_total    = (next_month_1 - _dt.date(now.year, now.month, 1)).days
    days_left     = days_total - now.day

    curr_spend     = sum(t["amount"] for t in transactions
                         if t["type"] == "debit" and t["month"] == current_month)
    monthly_budget = user.get("monthly_budget", 50000)
    budget_pct     = round(curr_spend / monthly_budget * 100, 1) if monthly_budget else 0

    top_categories = sorted(
        [{"category": k, **v} for k, v in analysis.get("category_breakdown", {}).items()],
        key=lambda x: x["total"], reverse=True,
    )[:6]

    active_alerts = db.get_alerts(user_id, unread_only=True, limit=25)

    # Pull financial_summary straight from analysis
    financials     = analysis.get("financial_summary", {})
    num_months     = analysis.get("summary", {}).get("num_months_analysed", 1) or 1
    monthly_income = round(financials.get("total_income", 0) / num_months, 2)
    monthly_savings = round(financials.get("net_savings", 0) / num_months, 2)

    return _ok({
        "user_id": user_id,

        "user": {
            "name":           user.get("name"),
            "monthly_budget": monthly_budget,
            "currency":       user.get("currency", "INR"),
        },

        "current_month_tracker": {
            "month":            current_month,
            "spent_so_far":     round(curr_spend, 2),
            "monthly_budget":   monthly_budget,
            "budget_used_pct":  budget_pct,
            "remaining_budget": round(monthly_budget - curr_spend, 2),
            "days_left":        days_left,
            "days_total":       days_total,
            "status": ("critical" if budget_pct >= 95 else
                       "warning"  if budget_pct >= 80 else "on_track"),
        },

        # Income, expenses, savings, savings_rate — used by UI cards + Gemini
        "financial_summary": {
            "total_income":    financials.get("total_income"),
            "total_expense":   financials.get("total_expense"),
            "net_savings":     financials.get("net_savings"),
            "savings_rate":    financials.get("savings_rate"),
            "monthly_income":  monthly_income,
            "monthly_savings": monthly_savings,
        },

        "summary": {
            "total_spent":         analysis.get("summary", {}).get("total_spent"),
            "overall_monthly_avg": analysis.get("summary", {}).get("overall_monthly_avg"),
            "num_months_analysed": num_months,
            "monthly_totals":      analysis.get("monthly_totals"),
        },

        "top_categories": top_categories,

        "behavior_profile": {
            "spender_type":              profile.get("spender_type"),
            "spender_label":             profile.get("spender_label"),
            "spender_rationale":         profile.get("spender_rationale"),
            "trend_direction":           profile.get("trend_direction"),
            "trend_slope":               profile.get("trend_slope"),
            "risk_score":                profile.get("risk_score"),
            "patterns":                  profile.get("patterns", []),
            "budget_utilisation_pct":    profile.get("budget_utilisation_pct"),
            "estimated_monthly_savings": profile.get("estimated_monthly_savings"),
            "income":                    profile.get("income"),
            "savings_rate":              profile.get("savings_rate"),
        },

        "forecast": {
            "predicted_amount":         forecast.get("overall_prediction", {}).get("predicted_amount"),
            "confidence_band":          forecast.get("overall_prediction", {}).get("confidence_band"),
            "top_predicted_categories": forecast.get("top_predicted_categories", [])[:5],
        },

        "category_velocity":        profile.get("category_velocity", {}),
        "top_growing_categories":   profile.get("top_growing_categories", []),
        "top_shrinking_categories": profile.get("top_shrinking_categories", []),

        "alerts": {
            "total_unread": len(active_alerts),
            "critical":     [a for a in active_alerts if a["severity"] == "critical"],
            "warning":      [a for a in active_alerts if a["severity"] == "warning"],
            "info":         [a for a in active_alerts if a["severity"] == "info"],
        },
    })


# Alerts

@app.get("/api/alerts")
def get_alerts():
    """Query params: ?user_id=<uuid>&unread_only=false&limit=50"""

    user_id = request.args.get("user_id", "").strip()

    unread_only = request.args.get(
        "unread_only",
        "false"
    ).lower() == "true"

    try:
        limit = int(request.args.get("limit", 50))
    except ValueError:
        return _err("'limit' must be an integer.")

    if limit <= 0:
        return _err("'limit' must be positive.")

    if not user_id:
        return _err("Query param 'user_id' is required.")

    _require_user(user_id)

    alerts = db.get_alerts(
        user_id,
        unread_only=unread_only,
        limit=limit
    )

    counts = {
        "critical": 0,
        "warning": 0,
        "info": 0
    }

    for a in alerts:
        counts[a["severity"]] = counts.get(a["severity"], 0) + 1

    return _ok({
        "user_id": user_id,
        "total": len(alerts),
        "severity_counts": counts,
        "alerts": alerts
    })


@app.patch("/api/alerts/<alert_id>")
def mark_alert_read(alert_id: str):
    body    = request.get_json(silent=True) or {}
    user_id = body.get("user_id", "").strip()
    if not user_id:
        return _err("'user_id' is required in the body.")
    _require_user(user_id)
    if not db.mark_alert_read(alert_id, user_id):
        return _err("Alert not found or already read.", 404)
    return _ok({"message": "Alert marked as read.", "alert_id": alert_id})


# Error handlers

@app.errorhandler(400)
def bad_request(exc):
    return _err(getattr(exc, "description", str(exc)), 400)

@app.errorhandler(404)
def not_found(exc):
    return _err(getattr(exc, "description", "Not found."), 404)

@app.errorhandler(405)
def method_not_allowed(exc):
    return _err("Method not allowed.", 405)

@app.errorhandler(500)
def internal_error(exc):
    logger.exception("Server error: %s", exc)
    return _err("Internal server error.", 500)


if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=5000,
            debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
