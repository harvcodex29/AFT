"""
prediction.py

Lightweight spending prediction using weighted moving average + linear trend.
No heavy ML dependencies – pure stdlib 
"""

from typing import Optional
import math


# Core helpers


def _weighted_moving_average(values: list[float], weights: Optional[list[float]] = None) -> float:
    """
    Compute a weighted average.  If no weights are given, recent months get
    exponentially higher weight (most-recent = highest weight).
    """
    n = len(values)
    if n == 0:
        return 0.0

    if weights is None:
        # Exponential weights: w_i = 2^i  (i=0 oldest … i=n-1 newest)
        weights = [2 ** i for i in range(n)]

    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0

    return sum(v * w for v, w in zip(values, weights)) / total_weight


def _linear_trend(values: list[float]) -> float:
    """
    Fit a simple OLS line y = a + b*x to the series and return the predicted
    next value (x = n).  Returns the plain mean if there is only one point.
    """
    n = len(values)
    if n == 0:
        return 0.0
    if n == 1:
        return values[0]

    x_vals = list(range(n))
    x_mean = sum(x_vals) / n
    y_mean = sum(values) / n

    numerator   = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, values))
    denominator = sum((x - x_mean) ** 2 for x in x_vals)

    if denominator == 0:
        return y_mean

    slope     = numerator / denominator
    intercept = y_mean - slope * x_mean

    return intercept + slope * n   # predict for x = n (next period)


def _clamp(value: float, min_val: float = 0.0) -> float:
    return max(value, min_val)


# Public API


def predict_next_month(
    monthly_totals: dict[str, float],
    blend_alpha: float = 0.6,
) -> dict:
    """
    Predict total spending for the month following the last one in
    *monthly_totals*.

    Strategy
    
    Blends two signals:
      • Weighted Moving Average  (gives more weight to recent months)
      • Linear Trend             (captures direction of change)

    ``blend_alpha`` controls the WMA weight (1-blend_alpha goes to the trend).

    Parameters
    
    monthly_totals : { "YYYY-MM": amount, … }   ordered or unordered
    blend_alpha    : float in [0, 1]

    Returns
    
    {
        "predicted_amount"  : float,
        "wma_estimate"      : float,
        "trend_estimate"    : float,
        "confidence_band"   : { "low": float, "high": float },
        "months_used"       : int,
        "method"            : str,
    }
    """
    if not monthly_totals:
        return {"error": "No monthly data provided for prediction."}

    sorted_months = sorted(monthly_totals.keys())
    values        = [monthly_totals[m] for m in sorted_months]
    n             = len(values)

    wma_est   = _weighted_moving_average(values)
    trend_est = _linear_trend(values)

    predicted = blend_alpha * wma_est + (1 - blend_alpha) * trend_est
    predicted = _clamp(predicted)

    # Confidence band: ±1 standard deviation of residuals (simple proxy)
    mean_val = sum(values) / n
    variance = sum((v - mean_val) ** 2 for v in values) / n
    std_dev  = math.sqrt(variance)

    return {
        "predicted_amount": round(predicted, 2),
        "wma_estimate":     round(wma_est,   2),
        "trend_estimate":   round(max(trend_est, 0), 2),
        "confidence_band": {
            "low":  round(_clamp(predicted - std_dev), 2),
            "high": round(predicted + std_dev,         2),
        },
        "months_used": n,
        "method":      f"WMA({blend_alpha:.0%}) + LinearTrend({1-blend_alpha:.0%})",
    }


def predict_by_category(
    transactions: list[dict],
    blend_alpha: float = 0.6,
) -> dict[str, dict]:
    """
    Run predict_next_month independently for each expense category.

    Returns
    -------
    { category: prediction_dict, … }
    """
    from collections import defaultdict

    cat_month: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for t in transactions:
        if t.get("type") == "debit":
            cat_month[t["category"]][t["month"]] += t["amount"]

    return {
        cat: predict_next_month(month_data, blend_alpha)
        for cat, month_data in cat_month.items()
    }


def spending_forecast(
    monthly_totals:      dict[str, float],
    transactions:        list[dict],
    blend_alpha:         float = 0.6,
) -> dict:
    """
    Combine overall and per-category predictions into a single forecast payload.
    """
    overall  = predict_next_month(monthly_totals, blend_alpha)
    by_cat   = predict_by_category(transactions,  blend_alpha)

    # Sort categories by predicted amount descending
    ranked = sorted(
        [{"category": c, **p} for c, p in by_cat.items() if "predicted_amount" in p],
        key=lambda x: x["predicted_amount"],
        reverse=True,
    )

    return {
        "overall_prediction":      overall,
        "category_predictions":    ranked,
        "top_predicted_categories": ranked[:5],
    }
