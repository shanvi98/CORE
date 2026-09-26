"""Turn per-order SHAP contributions into the 3 plain-English reasons a
Kestrel warehouse employee reads.

`explain()` is the only function app.py needs. Everything else here is
template plumbing. No feature names or SHAP numbers ever reach the output
text — only a sentence plus a fact pulled from `reference_rates`, which
train.py computes once on the training data and stores in model_meta.json.
"""
from __future__ import annotations

from typing import Any

from .features import FEATURE_GROUPS

GROUP_ORDER_FALLBACK = [
    "Payment", "Channel", "Discount & price", "Delivery promise", "Gift",
    "Order timing", "Address", "Customer history", "Shield membership",
    "Customer tenure", "Product",
]


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _fmt_payment(row: dict, ref: dict) -> str:
    mode = row.get("payment_mode")
    rates = ref.get("payment_mode", {})
    if mode == "cod" and "cod" in rates and "prepaid_upi" in rates:
        return (
            f"Cash on delivery: COD orders come back {_pct(rates['cod'])} of the "
            f"time vs {_pct(rates['prepaid_upi'])} for UPI."
        )
    if mode in rates:
        best = min(rates, key=rates.get)
        return (
            f"Payment method {mode.replace('_', ' ')}: these orders return "
            f"{_pct(rates[mode])} of the time, vs {_pct(rates[best])} for "
            f"{best.replace('_', ' ')}."
        )
    return "Payment method on this order."


def _fmt_channel(row: dict, ref: dict) -> str:
    ch = row.get("sales_channel")
    rates = ref.get("sales_channel", {})
    if ch in rates:
        return f"Sold via {ch.replace('_', ' ')}: this channel returns {_pct(rates[ch])} of orders."
    return "Sales channel on this order."


def _fmt_price(row: dict, ref: dict) -> str:
    disc = row.get("discount_pct")
    if disc is not None:
        return f"{disc:.0f}% discount was applied at checkout."
    return "Order value and discount."


def _fmt_delivery(row: dict, ref: dict) -> str:
    days = row.get("promised_delivery_days")
    rates = ref.get("promised_bucket", {})
    if row.get("slow_promise"):
        fast = rates.get("<=3")
        slow = rates.get(">=8")
        if fast is not None and slow is not None:
            return (
                f"Promised in {int(days)} days: slow promises (8+ days) return "
                f"{_pct(slow)} of the time vs {_pct(fast)} for fast ones (3 days or less)."
            )
        return f"Promised in {int(days)} days: a slower delivery promise than most orders."
    return f"Promised in {int(days)} days: a quick delivery promise." if days is not None else "Delivery promise."


def _fmt_gift(row: dict, ref: dict) -> str:
    rates = ref.get("is_gift", {})
    if row.get("is_gift"):
        y, n = rates.get("Y"), rates.get("N")
        if y is not None and n is not None:
            return f"Marked as a gift: gift orders return {_pct(y)} of the time vs {_pct(n)} for non-gifts."
        return "Marked as a gift."
    return "Not a gift order."


def _fmt_timing(row: dict, ref: dict) -> str:
    if row.get("is_festive_window"):
        rates = ref.get("festive", {})
        y, n = rates.get("yes"), rates.get("no")
        if y is not None and n is not None:
            return f"Placed in the Oct-Nov festive window: these orders return {_pct(y)} of the time vs {_pct(n)} otherwise."
        return "Placed during the Oct-Nov festive window."
    return "Order timing looks typical."


def _fmt_address(row: dict, ref: dict) -> str:
    if row.get("pin_missing"):
        rates = ref.get("pin_missing", {})
        y, n = rates.get("yes"), rates.get("no")
        if y is not None and n is not None:
            return f"No delivery address captured (walk-in default pincode): these orders return {_pct(y)} of the time vs {_pct(n)} when an address is on file."
        return "No delivery address was captured for this order."
    note_type = row.get("note_type")
    if note_type and note_type not in ("none",):
        return f"Delivery note on file ({note_type.replace('_', ' ')})."
    return "Delivery address on file, no unusual notes."


def _fmt_customer_history(row: dict, ref: dict) -> str:
    if row.get("is_first_order"):
        rates = ref.get("first_order", {})
        f, r = rates.get("first"), rates.get("repeat")
        if f is not None and r is not None:
            return f"First order from this customer: first orders return {_pct(f)} of the time vs {_pct(r)} for repeat customers."
        return "This is the customer's first order with Kestrel."
    if row.get("ever_returned"):
        rates = ref.get("prior_returns_flag", {})
        y, n = rates.get("has_returns"), rates.get("none")
        if y is not None and n is not None:
            return f"Customer has returned before: {_pct(y)} of their orders come back vs {_pct(n)} for customers with a clean history."
        return "This customer has returned an order before."
    return "No previous returns from this customer."


def _fmt_shield(row: dict, ref: dict) -> str:
    rates = ref.get("shield_member", {})
    if row.get("shield_member"):
        y, n = rates.get("Y"), rates.get("N")
        if y is not None and n is not None:
            return f"Shield member: Shield customers return {_pct(y)} of orders vs {_pct(n)} for non-members (returns are free for them)."
        return "This customer is a Shield member."
    return "Not a Shield member."


def _fmt_tenure(row: dict, ref: dict) -> str:
    if row.get("signup_after_order"):
        return "Customer profile recorded after this order; using default assumptions for tenure."
    days = row.get("tenure_days")
    if days is not None and days < 30:
        return f"New customer: signed up {int(days)} days before this order."
    return "Established customer relationship."


def _fmt_product(row: dict, ref: dict) -> str:
    family = row.get("family")
    rates = ref.get("family", {})
    if family in rates:
        return f"{family}: this product family returns {_pct(rates[family])} of orders."
    return f"Product: {family or 'n/a'}."


_TEMPLATES = {
    "Payment": _fmt_payment,
    "Channel": _fmt_channel,
    "Discount & price": _fmt_price,
    "Delivery promise": _fmt_delivery,
    "Gift": _fmt_gift,
    "Order timing": _fmt_timing,
    "Address": _fmt_address,
    "Customer history": _fmt_customer_history,
    "Shield membership": _fmt_shield,
    "Customer tenure": _fmt_tenure,
    "Product": _fmt_product,
}


def explain(
    contrib_row: list[float],
    feature_row: dict[str, Any],
    meta: dict[str, Any],
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """Aggregate a single row's SHAP contributions into `top_n` grouped,
    plain-English reasons.

    contrib_row: log-odds contributions from
        booster.predict(X, pred_contrib=True)[i] — length = n_features + 1,
        last entry is the bias term (dropped here).
    feature_row: dict of feature_name -> raw value for this same row, used
        to fill in the template sentences.
    meta: model_meta.json contents; needs "feature_names" and
        "reference_rates".
    """
    feature_names: list[str] = meta["feature_names"]
    reference_rates: dict = meta.get("reference_rates", {})
    contribs = list(contrib_row)[: len(feature_names)]

    group_impact: dict[str, float] = {}
    for name, val in zip(feature_names, contribs):
        group = FEATURE_GROUPS.get(name, "Other")
        group_impact[group] = group_impact.get(group, 0.0) + float(val)

    ranked = sorted(group_impact.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top = ranked[:top_n]
    # Risk-increasing reasons first, ties broken by |impact| (already sorted).
    top.sort(key=lambda kv: (kv[1] <= 0, -abs(kv[1])))

    reasons = []
    for rank, (group, impact) in enumerate(top, start=1):
        template = _TEMPLATES.get(group)
        text = template(feature_row, reference_rates) if template else f"{group} factors."
        reasons.append(
            {
                "rank": rank,
                "factor": group,
                "direction": "increases_risk" if impact > 0 else "decreases_risk",
                "impact": round(impact, 4),
                "text": text,
            }
        )
    return reasons
