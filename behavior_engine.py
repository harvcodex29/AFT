import statistics
from collections import defaultdict
from datetime import datetime
import calendar

SPENDER_TYPES = {
    "impulsive":    "Frequent small & large unplanned purchases, high variance",
    "conservative": "Consistently below-average spending, few spikes",
    "routine":      "Predictable fixed monthly bills dominate",
    "lifestyle":    "High food, entertainment, travel spend relative to total",
    "saver":        "Spending trending down month-over-month",
    "escalating":   "Spending trending up consistently",
}


def _today() -> datetime:
    return datetime.now()


def _pct(part: float, total: float) -> float:
    return round((part / total * 100) if total else 0, 1)


def _trend_slope(monthly_totals: dict) -> float:
    months = sorted(monthly_totals.keys())
    values = [monthly_totals[m] for m in months]
    n      = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    num    = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den    = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den else 0.0


def _classify_spender(monthly_totals: dict, category_breakdown: dict,
                      slope: float) -> tuple[str, str]:
    values = list(monthly_totals.values())
    if not values:
        return "unknown", "Not enough data"

    mean_spend = statistics.mean(values)
    std_dev    = statistics.stdev(values) if len(values) > 1 else 0
    cv         = std_dev / mean_spend if mean_spend else 0

    lifestyle_cats = {"Food & Dining", "Entertainment", "Travel", "Shopping"}
    fixed_cats     = {"Rent & Housing", "Utilities", "Finance"}

    lifestyle_pct = sum(
        v["monthly_avg"] for k, v in category_breakdown.items() if k in lifestyle_cats
    ) / mean_spend * 100 if mean_spend else 0

    fixed_pct = sum(
        v["monthly_avg"] for k, v in category_breakdown.items() if k in fixed_cats
    ) / mean_spend * 100 if mean_spend else 0

    if slope < -500:      return "saver",        f"Spend trending down ₹{abs(slope):.0f}/month"
    if slope > 1500:      return "escalating",   f"Spend growing ₹{slope:.0f}/month"
    if lifestyle_pct > 45: return "lifestyle",   f"{lifestyle_pct:.0f}% on lifestyle categories"
    if fixed_pct > 60:    return "routine",      f"{fixed_pct:.0f}% on fixed bills"
    if cv > 0.25:         return "impulsive",    f"High variance (CV={cv:.2f})"
    return "conservative", f"Consistent spend, low variance (CV={cv:.2f})"


def _detect_patterns(transactions: list[dict], monthly_totals: dict) -> list[dict]:
    patterns = []
    values   = list(monthly_totals.values())
    if len(values) < 2:
        return patterns

    # 1. Weekend splurge
    weekend_total = weekday_total = 0.0
    for t in transactions:
        if t["type"] != "debit":
            continue
        dow = datetime.strptime(t["date"], "%Y-%m-%d").weekday()
        if dow >= 5:
            weekend_total += t["amount"]
        else:
            weekday_total += t["amount"]

    debit_total = weekend_total + weekday_total
    if debit_total > 0:
        wpct = weekend_total / debit_total * 100
        if wpct > 38:
            patterns.append({
                "id": "weekend_splurge", "name": "Weekend splurger",
                "description": f"You spend {wpct:.0f}% of your money on weekends (expected ~29%).",
                "severity": "warning",
                "evidence": {"weekend_total": round(weekend_total, 2),
                             "weekday_total": round(weekday_total, 2),
                             "weekend_pct":   round(wpct, 1)},
            })

    # 2. Month-end rush
    early = late = 0.0
    for t in transactions:
        if t["type"] != "debit":
            continue
        if int(t["date"].split("-")[2]) <= 15:
            early += t["amount"]
        else:
            late  += t["amount"]

    if (early + late) > 0:
        late_pct = late / (early + late) * 100
        if late_pct > 58:
            patterns.append({
                "id": "month_end_rush", "name": "Month-end spender",
                "description": f"You spend {late_pct:.0f}% of your money in the second half of the month.",
                "severity": "info",
                "evidence": {"first_half": round(early, 2), "second_half": round(late, 2),
                             "second_half_pct": round(late_pct, 1)},
            })

    # 3. Continuously rising spend
    months     = sorted(monthly_totals.keys())
    spend_list = [monthly_totals[m] for m in months]
    if len(spend_list) >= 3:
        if all(spend_list[i] < spend_list[i + 1] for i in range(len(spend_list) - 1)):
            patterns.append({
                "id": "continuously_rising", "name": "Continuously rising spend",
                "description": f"Spending increased every month for {len(spend_list)} consecutive months.",
                "severity": "critical",
                "evidence": {"months": months, "amounts": [round(v, 2) for v in spend_list]},
            })

    # 4. Subscription creep
    sub_kw    = ["netflix", "prime", "hotstar", "spotify", "youtube premium",
                 "zee5", "sony liv", "jio cinema", "udemy", "coursera",
                 "linkedin premium", "notion", "zoom", "apple music"]
    sub_txns  = [t for t in transactions if t["type"] == "debit"
                 and any(kw in t["description"].lower() for kw in sub_kw)]
    unique_subs = {t["description"].lower()[:20] for t in sub_txns}
    if len(unique_subs) >= 3:
        monthly_cost = sum(t["amount"] for t in sub_txns) / max(len(months), 1)
        patterns.append({
            "id": "subscription_creep", "name": "Subscription creep",
            "description": f"{len(unique_subs)} active subscriptions costing ~₹{monthly_cost:.0f}/month.",
            "severity": "warning",
            "evidence": {"subscription_count": len(unique_subs),
                         "monthly_cost": round(monthly_cost, 2)},
        })

    # 5. Erratic food spend
    food_by_month: dict[str, float] = defaultdict(float)
    for t in transactions:
        if t["type"] == "debit" and t["category"] == "Food & Dining":
            food_by_month[t["month"]] += t["amount"]
    if len(food_by_month) >= 2:
        fv   = list(food_by_month.values())
        fmean = statistics.mean(fv)
        fstd  = statistics.stdev(fv)
        if fmean > 0 and fstd / fmean > 0.3:
            patterns.append({
                "id": "erratic_food_spend", "name": "Erratic food spending",
                "description": f"Food spend varies ±{fstd/fmean*100:.0f}% month-to-month. Try a weekly food budget.",
                "severity": "info",
                "evidence": {"monthly_food": {m: round(v, 2) for m, v in food_by_month.items()},
                             "avg": round(fmean, 2), "std": round(fstd, 2)},
            })

    return patterns


def _category_velocity(transactions: list[dict]) -> dict[str, dict]:
    months = sorted({t["month"] for t in transactions if t["type"] == "debit"})
    if len(months) < 2:
        return {}
    prev_m, curr_m = months[-2], months[-1]
    prev_cat: dict[str, float] = defaultdict(float)
    curr_cat: dict[str, float] = defaultdict(float)
    for t in transactions:
        if t["type"] != "debit":
            continue
        if t["month"] == prev_m:
            prev_cat[t["category"]] += t["amount"]
        elif t["month"] == curr_m:
            curr_cat[t["category"]] += t["amount"]

    result = {}
    for cat in set(prev_cat) | set(curr_cat):
        p   = prev_cat.get(cat, 0)
        c   = curr_cat.get(cat, 0)
        pct = ((c - p) / p * 100) if p > 0 else (100 if c > 0 else 0)
        result[cat] = {"prev": round(p, 2), "current": round(c, 2), "pct_change": round(pct, 1)}
    return result


def _risk_score(budget_util_pct: float, slope: float, patterns: list[dict]) -> dict:
    score  = min(budget_util_pct, 40)
    score += min(max(slope / 500, 0), 20)
    score += sum(15 if p["severity"] == "critical" else 7 for p in patterns)
    score  = min(round(score), 100)
    grade  = "A" if score < 25 else "B" if score < 45 else "C" if score < 60 else "D" if score < 75 else "F"
    return {"score": score, "grade": grade,
            "label": {"A": "Excellent", "B": "Good", "C": "Fair", "D": "At Risk", "F": "Critical"}[grade]}


def build_behavior_profile(user_id: str, transactions: list[dict],
                           analysis: dict, forecast: dict, user: dict) -> dict:
    monthly_totals_map = analysis.get("monthly_totals", {})
    cat_breakdown      = analysis.get("category_breakdown", {})
    financials         = analysis.get("financial_summary", {})
    income             = financials.get("total_income", 0)
    savings_rate       = financials.get("savings_rate", 0)

    slope        = _trend_slope(monthly_totals_map)
    spender_type, rationale = _classify_spender(monthly_totals_map, cat_breakdown, slope)
    patterns     = _detect_patterns(transactions, monthly_totals_map)
    cat_velocity = _category_velocity(transactions)

    monthly_budget  = user.get("monthly_budget", 50000)
    overall_avg     = analysis.get("summary", {}).get("overall_monthly_avg", 0)
    budget_util_pct = _pct(overall_avg, monthly_budget)

    predicted_spend = forecast.get("overall_prediction", {}).get("predicted_amount", overall_avg)
    est_savings     = max(monthly_budget - predicted_spend, 0)

    growing   = sorted(
        [{"category": k, **v} for k, v in cat_velocity.items() if v["pct_change"] > 0],
        key=lambda x: x["pct_change"], reverse=True,
    )[:3]
    shrinking = sorted(
        [{"category": k, **v} for k, v in cat_velocity.items() if v["pct_change"] < 0],
        key=lambda x: x["pct_change"],
    )[:3]

    return {
        "user_id":                   user_id,
        "computed_at":               _today().isoformat(),
        "spender_type":              spender_type,
        "spender_label":             SPENDER_TYPES.get(spender_type, ""),
        "spender_rationale":         rationale,
        "trend_slope":               round(slope, 2),
        "trend_direction":           "up" if slope > 100 else ("down" if slope < -100 else "stable"),
        "monthly_volatility":        round(statistics.stdev(list(monthly_totals_map.values()))
                                           if len(monthly_totals_map) > 1 else 0, 2),
        "budget_utilisation_pct":    budget_util_pct,
        "estimated_monthly_savings": round(est_savings, 2),
        "income":                    round(income, 2),
        "savings_rate":              round(savings_rate, 2),
        "patterns":                  patterns,
        "category_velocity":         cat_velocity,
        "top_growing_categories":    growing,
        "top_shrinking_categories":  shrinking,
        "risk_score":                _risk_score(budget_util_pct, slope, patterns),
    }


def generate_alerts(user_id: str, transactions: list[dict], analysis: dict,
                    forecast: dict, profile: dict, user: dict) -> list[dict]:
    alerts         = []
    monthly_budget = user.get("monthly_budget", 50000)
    now            = _today()
    days_in_month  = calendar.monthrange(now.year, now.month)[1]
    current_month  = now.strftime("%Y-%m")

    curr_spend   = sum(t["amount"] for t in transactions
                       if t["type"] == "debit" and t["month"] == current_month)
    pct_spent    = _pct(curr_spend, monthly_budget)
    pct_of_month = round(now.day / days_in_month * 100, 1)

    # Budget threshold (fires only the highest triggered tier)
    for threshold, sev in [(95, "critical"), (80, "warning"), (60, "info")]:
        if pct_spent >= threshold:
            days_left = days_in_month - now.day
            alerts.append({
                "type": "budget_threshold", "severity": sev,
                "title":   f"{'⛔' if sev == 'critical' else '⚠️'} Budget {threshold}% reached",
                "message": (f"You've spent ₹{curr_spend:,.0f} — {pct_spent:.0f}% of your "
                            f"₹{monthly_budget:,.0f} budget with {days_left} days left."),
                "metadata": {"current_spend": round(curr_spend, 2),
                             "monthly_budget": monthly_budget,
                             "pct_spent": pct_spent, "days_remaining": days_left},
            })
            break

    # Pace overspend projection
    if now.day > 3 and pct_of_month > 0:
        projected = curr_spend / (now.day / days_in_month)
        if projected > monthly_budget * 1.1:
            overshoot = projected - monthly_budget
            alerts.append({
                "type": "pace_overspend", "severity": "warning",
                "title":   "📈 On track to overspend",
                "message": (f"At current pace you'll spend ₹{projected:,.0f} this month — "
                            f"₹{overshoot:,.0f} over budget."),
                "metadata": {"projected_spend": round(projected, 2),
                             "overshoot": round(overshoot, 2),
                             "monthly_budget": monthly_budget, "day": now.day},
            })

    # Category spikes ≥ 2×
    for spike in (analysis.get("overspending_alerts") or [])[:3]:
        if spike["spike_ratio"] >= 2.0:
            sev = "critical" if spike["spike_ratio"] >= 3.0 else "warning"
            alerts.append({
                "type": "category_spike", "severity": sev,
                "title":   f"🚨 Unusual {spike['category']} spend in {spike['month']}",
                "message": (f"₹{spike['amount']:,.0f} on {spike['category']} — "
                            f"{spike['spike_ratio']:.1f}× your avg of ₹{spike['average']:,.0f}."),
                "metadata": spike,
            })

    # Behaviour pattern warnings
    for pat in profile.get("patterns", []):
        sev = pat.get("severity", "info")
        alerts.append({
            "type": f"pattern_{pat['id']}", "severity": sev,
            "title":   f"{'🔴' if sev == 'critical' else '🟡'} Pattern: {pat['name']}",
            "message": pat["description"],
            "metadata": pat.get("evidence"),
        })

    # MoM category velocity warnings
    for ci in profile.get("top_growing_categories", []):
        if ci["pct_change"] >= 40:
            sev = "critical" if ci["pct_change"] >= 80 else "warning"
            alerts.append({
                "type": "category_rising", "severity": sev,
                "title":   f"📊 {ci['category']} up {ci['pct_change']:.0f}%",
                "message": (f"{ci['category']} jumped from ₹{ci['prev']:,.0f} to "
                            f"₹{ci['current']:,.0f} vs last month."),
                "metadata": ci,
            })

    # Risk score alert
    risk = profile.get("risk_score", {})
    if risk.get("grade") in ("D", "F"):
        alerts.append({
            "type": "risk_score",
            "severity": "critical" if risk["grade"] == "F" else "warning",
            "title":   f"🔴 Financial risk: Grade {risk['grade']} ({risk['score']}/100)",
            "message": (f"Risk score {risk['score']}/100 ({risk['label']}). "
                        f"You're a '{profile.get('spender_type', '')}' spender. "
                        f"Check the suggestions tab for an action plan."),
            "metadata": risk,
        })

    # Forecast over budget
    pred_amount = forecast.get("overall_prediction", {}).get("predicted_amount", 0)
    if pred_amount > monthly_budget:
        excess = pred_amount - monthly_budget
        alerts.append({
            "type": "forecast_over_budget", "severity": "warning",
            "title":   "🔮 Forecast: next month over budget",
            "message": (f"Predicted spend ₹{pred_amount:,.0f} — ₹{excess:,.0f} over your "
                        f"₹{monthly_budget:,.0f} budget."),
            "metadata": {"predicted_amount": pred_amount,
                         "budget": monthly_budget, "excess": round(excess, 2)},
        })

    # Low savings rate
    savings_rate = profile.get("savings_rate", 0)
    if 0 < savings_rate < 10:
        alerts.append({
            "type": "low_savings_rate", "severity": "warning",
            "title":   f"💰 Savings rate only {savings_rate:.1f}%",
            "message": (f"You're saving just {savings_rate:.1f}% of your income. "
                        f"Aim for at least 20% to build a financial buffer."),
            "metadata": {"savings_rate": savings_rate},
        })

    # Spender-type nudge
    nudges = {
        "impulsive":  ("💡 Impulse spending detected",
                       "Try a 24-hour rule before purchases over ₹1,000."),
        "escalating": ("⬆️ Spending trend is climbing",
                       f"Spend rising ~₹{abs(profile.get('trend_slope', 0)):,.0f}/month. "
                       f"Forecast: ₹{pred_amount:,.0f} next month."),
        "lifestyle":  ("🍽️ Lifestyle costs dominating",
                       "Food, entertainment & shopping form a large share. Set a weekly discretionary cap."),
    }
    spender_type = profile.get("spender_type", "")
    if spender_type in nudges:
        title, msg = nudges[spender_type]
        alerts.append({
            "type": "spender_nudge", "severity": "info",
            "title": title, "message": msg,
            "metadata": {"spender_type": spender_type},
        })

    return alerts
