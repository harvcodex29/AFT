from datetime import datetime
from collections import defaultdict
from typing import Any

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Food & Dining":  ["restaurant", "cafe", "coffee", "swiggy", "zomato",
                       "pizza", "burger", "hotel", "dhaba", "mess", "tiffin"],
    "Groceries":      ["grocery", "supermarket", "bigbasket", "blinkit",
                       "grofers", "d-mart", "nature basket", "vegetables", "fruits", "milk"],
    "Transportation": ["uber", "ola", "rapido", "metro", "bus", "train",
                       "irctc", "petrol", "fuel", "toll", "parking", "auto", "cab"],
    "Utilities":      ["electricity", "water", "gas", "internet", "broadband",
                       "airtel", "jio", "bsnl", "vi ", "vodafone", "recharge",
                       "bill", "bescom", "mseb"],
    "Shopping":       ["amazon", "flipkart", "myntra", "ajio", "meesho",
                       "nykaa", "clothes", "fashion", "shoes", "decathlon"],
    "Entertainment":  ["netflix", "prime", "hotstar", "spotify", "youtube",
                       "movie", "cinema", "pvr", "inox", "game", "steam"],
    "Health":         ["hospital", "clinic", "pharmacy", "medicine",
                       "doctor", "diagnostic", "lab", "apollo", "medplus", "gym", "fitness"],
    "Education":      ["course", "udemy", "coursera", "book", "tuition",
                       "school", "college", "fee", "subscription"],
    "Finance":        ["emi", "loan", "insurance", "mutual fund", "sip",
                       "investment", "credit card", "tax", "fd", "rd"],
    "Rent & Housing": ["rent", "maintenance", "society", "flat", "pg", "hostel", "deposit"],
    "Personal Care":  ["salon", "spa", "haircut", "cosmetics", "beauty"],
    "Travel":         ["flight", "hotel booking", "oyo", "makemytrip",
                       "goibibo", "cleartrip", "airbnb", "trip", "tour"],
    "Transfers":      ["neft", "imps", "upi", "transfer", "sent to",
                       "paid to", "phonepe", "gpay", "paytm"],
    "ATM / Cash":     ["atm", "cash withdrawal", "withdrawal"],
    "Income":         ["salary", "credited", "refund", "cashback", "interest", "dividend"],
    "Other":          [],
}


def categorize_transaction(description: str) -> str:
    desc_lower = description.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in desc_lower for kw in keywords):
            return category
    return "Other"


def parse_transactions(raw_data: list[dict]) -> list[dict]:
    """
    Parse and enrich raw transaction records.
    Credits stored as negative amounts (income), debits as positive (expense).
    """
    transactions = []
    for raw in raw_data:
        try:
            date_obj = datetime.strptime(raw["date"].strip(), "%Y-%m-%d")
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid or missing 'date' in record {raw}: {exc}")

        try:
            amount = float(raw["amount"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid or missing 'amount' in record {raw}: {exc}")

        description = str(raw.get("description", "")).strip() or "Unknown"
        txn_type    = str(raw.get("type", "debit")).lower()


        if txn_type == "credit":
            amount = -abs(amount)
        else:
            amount = abs(amount)

        transactions.append({
            "date":        date_obj.strftime("%Y-%m-%d"),
            "month":       date_obj.strftime("%Y-%m"),
            "description": description,
            "amount":      amount,
            "type":        txn_type,
            "category":    categorize_transaction(description),
        })

    return transactions


def monthly_totals(transactions: list[dict]) -> dict[str, float]:
    """Expense totals per month (debit only, positive amounts)."""
    totals: dict[str, float] = defaultdict(float)
    for t in transactions:
        if t["type"] == "debit":
            totals[t["month"]] += t["amount"]
    return dict(sorted(totals.items()))


def monthly_income(transactions: list[dict]) -> dict[str, float]:
    """Income totals per month (credit only, returned as positive)."""
    totals: dict[str, float] = defaultdict(float)
    for t in transactions:
        if t["type"] == "credit":
            totals[t["month"]] += abs(t["amount"])
    return dict(sorted(totals.items()))


def category_breakdown(transactions: list[dict]) -> dict[str, dict]:
    """
    Per-category stats — monthly_avg uses only months where the category
    actually appeared, avoiding undercount dilution.
    """
    cat_amounts:  dict[str, list[float]] = defaultdict(list)
    cat_months:   dict[str, set[str]]    = defaultdict(set)

    for t in transactions:
        if t["type"] == "debit":
            cat_amounts[t["category"]].append(t["amount"])
            cat_months[t["category"]].add(t["month"])

    breakdown = {}
    for cat, amounts in cat_amounts.items():
        total       = sum(amounts)
        active_months = len(cat_months[cat]) or 1
        breakdown[cat] = {
            "total":       round(total, 2),
            "count":       len(amounts),
            "monthly_avg": round(total / active_months, 2),
        }
    return breakdown


def detect_overspending(transactions: list[dict], spike_multiplier: float = 1.5) -> list[dict]:
    cat_month: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for t in transactions:
        if t["type"] == "debit":
            cat_month[t["category"]][t["month"]] += t["amount"]

    alerts = []
    for cat, month_data in cat_month.items():
        if not month_data:
            continue
        avg = sum(month_data.values()) / len(month_data)
        if avg == 0:
            continue
        for month, total in month_data.items():
            ratio = total / avg
            if ratio >= spike_multiplier:
                alerts.append({
                    "month":       month,
                    "category":    cat,
                    "amount":      round(total, 2),
                    "average":     round(avg, 2),
                    "spike_ratio": round(ratio, 2),
                })

    alerts.sort(key=lambda x: x["spike_ratio"], reverse=True)
    return alerts


def full_analysis(transactions: list[dict]) -> dict[str, Any]:
    totals       = monthly_totals(transactions)
    income_map   = monthly_income(transactions)
    breakdown    = category_breakdown(transactions)
    overspending = detect_overspending(transactions)

    total_spent     = sum(totals.values())
    num_months      = max(len(totals), 1)
    overall_monthly_avg = round(total_spent / num_months, 2)

    # Financial summary — income, savings, savings rate
    income_total  = sum(income_map.values())
    expense_total = total_spent
    net_savings   = income_total - expense_total
    savings_rate  = (net_savings / income_total * 100) if income_total else 0

    top_categories = sorted(
        breakdown.items(), key=lambda x: x[1]["total"], reverse=True
    )[:5]

    return {
        "monthly_totals":      totals,
        "monthly_income":      income_map,
        "category_breakdown":  breakdown,
        "category_averages":   {c: v["monthly_avg"] for c, v in breakdown.items()},
        "overspending_alerts": overspending,
        "financial_summary": {
            "total_income":   round(income_total, 2),
            "total_expense":  round(expense_total, 2),
            "net_savings":    round(net_savings, 2),
            "savings_rate":   round(savings_rate, 2),
        },
        "summary": {
            "total_spent":         round(total_spent, 2),
            "num_months_analysed": num_months,
            "overall_monthly_avg": overall_monthly_avg,
            "top_categories":      [{"category": c, **s} for c, s in top_categories],
        },
    }
