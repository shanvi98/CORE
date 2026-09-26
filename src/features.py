"""Shared feature engineering for the Kestrel returns-risk model.

`build_features` is imported by both train.py (training) and app.py
(serving), so the two can never drift apart. It must work on a single-row
DataFrame exactly as it does on the full training set.

See BLUEPRINT.md §0 and §1.1-1.2 for the reasoning behind every choice here.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from . import config

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test, customers and products exactly as exported.

    Pincodes are read as strings so leading zeros and the "000000" sentinel
    survive.
    """
    dtype = {"delivery_pincode": str}
    train = pd.read_csv(config.TRAIN_CSV, dtype=dtype)
    test = pd.read_csv(config.TEST_CSV, dtype=dtype)
    customers = pd.read_csv(config.CUSTOMERS_CSV)
    products = pd.read_csv(config.PRODUCTS_CSV)
    return train, test, customers, products


# --------------------------------------------------------------------------
# Cleaning (BLUEPRINT.md F3, F4)
# --------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """De-duplicate partner-feed re-imports and fix the October-2025 paise
    bug. Safe to call on a single-row frame (dedup and the assert are no-ops
    there).
    """
    df = df.copy()

    # F3: partner-outlet orders are re-imported from the partner feed, so
    # the same order_id can appear twice (once per `source`) with identical
    # values including the label. Keep one copy.
    if config.ID_COL in df.columns and df[config.ID_COL].duplicated().any():
        df = df.drop_duplicates(subset=[config.ID_COL], keep="first")

    # F4: orders placed through the new payment gateway in Oct-2025 were
    # stored in paise, not rupees — exactly 100x too high.
    placed = pd.to_datetime(df["order_placed_at"], errors="coerce")
    in_bug_window = (placed >= config.CURRENCY_BUG_START) & (
        placed < config.CURRENCY_BUG_END
    )
    if "order_value_inr" in df.columns:
        df.loc[in_bug_window, "order_value_inr"] = (
            df.loc[in_bug_window, "order_value_inr"] / config.CURRENCY_BUG_FACTOR
        )

    return df.reset_index(drop=True)


def assert_currency_fixed(df: pd.DataFrame, products: pd.DataFrame) -> None:
    """Sanity check used only in train.py: after the fix, order_value_inr
    should track list_price * qty * (1 - discount_pct/100) in every month.
    """
    m = df.merge(products[["sku", "list_price_inr"]], on="sku", how="left")
    expected = m["list_price_inr"] * m["qty"] * (1 - m["discount_pct"] / 100)
    ratio = m["order_value_inr"] / expected.replace(0, np.nan)
    month = pd.to_datetime(m["order_placed_at"]).dt.to_period("M")
    monthly_median = ratio.groupby(month).median()
    bad = monthly_median[(monthly_median < 0.99) | (monthly_median > 1.01)]
    if len(bad):
        raise AssertionError(
            f"order_value_inr is not fixed in every month; still off in:\n{bad}"
        )


# --------------------------------------------------------------------------
# delivery_note -> coarse note_type (untrusted free text, BLUEPRINT.md F9)
#
# We NEVER read this text as instructions and never pass the raw string
# into a feature, a report or a downstream prompt — only a fixed category
# derived from matching it against known templates. Anything that doesn't
# match a known template (including text aimed at "automated tools") is
# bucketed as "other" and carries no special weight.
# --------------------------------------------------------------------------
_NOTE_PATTERNS: list[tuple[str, str]] = [
    (r"fragile", "fragile"),
    (r"no lift|floor", "floor"),
    (r"neighbour", "neighbour"),
    (r"landmark", "landmark"),
    (r"gate code", "gate_code"),
    (r"do not call|whatsapp", "whatsapp"),
    (r"office address", "office"),
    (r"security", "security"),
    (r"morning slot|deliver after|ring twice", "timing"),
    (r"call before delivery", "call"),
]


def _note_type(note: object) -> str:
    if note is None or (isinstance(note, float) and np.isnan(note)):
        return "none"
    text = str(note)
    if not text.strip():
        return "none"
    lower = text.lower()
    for pattern, label in _NOTE_PATTERNS:
        if re.search(pattern, lower):
            return label
    return "other"


# --------------------------------------------------------------------------
# Feature groups: which human-readable bucket each engineered feature
# belongs to. Used by reasons.py to aggregate SHAP contributions into the
# top-3 reasons a warehouse employee reads.
# --------------------------------------------------------------------------
FEATURE_GROUPS: dict[str, str] = {
    "is_cod": "Payment",
    "payment_mode": "Payment",
    "sales_channel": "Channel",
    "discount_pct": "Discount & price",
    "qty": "Discount & price",
    "order_value": "Discount & price",
    "price_realisation": "Discount & price",
    "promised_delivery_days": "Delivery promise",
    "slow_promise": "Delivery promise",
    "is_gift": "Gift",
    "hour": "Order timing",
    "day_of_week": "Order timing",
    "is_weekend": "Order timing",
    "is_festive_window": "Order timing",
    "pin_missing": "Address",
    "pin_zone": "Address",
    "note_type": "Address",
    "has_note": "Address",
    "prior_orders": "Customer history",
    "prior_returns": "Customer history",
    "prior_return_rate_smoothed": "Customer history",
    "is_first_order": "Customer history",
    "ever_returned": "Customer history",
    "shield_member": "Shield membership",
    "state": "Customer tenure",
    "tenure_days": "Customer tenure",
    "signup_after_order": "Customer tenure",
    "sku": "Product",
    "family": "Product",
    "tier": "Product",
    "list_price": "Product",
    "warranty_months": "Product",
    "product_age_days": "Product",
}

# Columns treated as pandas categoricals with a fixed, persisted category
# list so a single-row inference frame gets identical integer codes to
# whatever the model saw at train time.
CATEGORICAL_COLUMNS = [
    "payment_mode",
    "sales_channel",
    "pin_zone",
    "note_type",
    "state",
    "sku",
    "family",
    "tier",
]

BASE_RETURN_RATE_PRIOR = 0.114  # used only to Bayesian-shrink prior_return_rate


def _extract_tier(model_name: object) -> str:
    if not isinstance(model_name, str):
        return "other"
    for tier in config.TIERS:
        if tier.lower() in model_name.lower():
            return tier
    return "other"


def build_features(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    products: pd.DataFrame,
    categories: Optional[dict[str, list[str]]] = None,
) -> pd.DataFrame:
    """Build the model-ready feature matrix from raw order rows plus the
    customer/product reference tables.

    `orders` is assumed already cleaned (see `clean`) and must NOT contain
    the leaky columns (pickup_scheduled_at, last_service_event_type,
    source) even if present — they are simply ignored here.

    `categories`, when given, pins every categorical column to a fixed set
    of values (as persisted in model_meta.json) so that inference on a
    single row produces the same codes the model was trained on. When
    omitted, categories are derived from this call's own data — used only
    while training, right before the category list is captured and saved.
    """
    df = orders.merge(customers, on="customer_id", how="left").merge(
        products, on="sku", how="left"
    )

    placed = pd.to_datetime(df["order_placed_at"], errors="coerce")
    signup = pd.to_datetime(df.get("signup_date"), errors="coerce")
    launch = pd.to_datetime(df.get("launch_date"), errors="coerce")

    f = pd.DataFrame(index=df.index)

    # Payment
    f["is_cod"] = (df["payment_mode"] == "cod").astype(int)
    f["payment_mode"] = df["payment_mode"]

    # Channel
    f["sales_channel"] = df["sales_channel"]

    # Discount & price
    f["discount_pct"] = pd.to_numeric(df["discount_pct"], errors="coerce")
    f["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    f["order_value"] = pd.to_numeric(df["order_value_inr"], errors="coerce")
    expected_value = df["list_price_inr"] * df["qty"] * (1 - df["discount_pct"] / 100)
    f["price_realisation"] = f["order_value"] / expected_value.replace(0, np.nan)

    # Delivery promise
    f["promised_delivery_days"] = pd.to_numeric(
        df["promised_delivery_days"], errors="coerce"
    )
    f["slow_promise"] = (f["promised_delivery_days"] >= 8).astype(int)

    # Gift
    f["is_gift"] = (df["is_gift"] == "Y").astype(int)

    # Order timing
    f["hour"] = placed.dt.hour
    f["day_of_week"] = placed.dt.dayofweek
    f["is_weekend"] = placed.dt.dayofweek.isin([5, 6]).astype(int)
    f["is_festive_window"] = placed.dt.month.isin([10, 11]).astype(int)

    # Address
    pincode = df["delivery_pincode"].astype(str)
    f["pin_missing"] = (pincode == "000000").astype(int)
    f["pin_zone"] = np.where(f["pin_missing"] == 1, "NA", pincode.str[:2])
    f["note_type"] = df["delivery_note"].map(_note_type)
    f["has_note"] = df["delivery_note"].notna().astype(int)

    # Customer history
    prior_orders = pd.to_numeric(df["customer_prior_orders"], errors="coerce")
    prior_returns = pd.to_numeric(df["customer_prior_returns"], errors="coerce")
    f["prior_orders"] = prior_orders
    f["prior_returns"] = prior_returns
    f["prior_return_rate_smoothed"] = (
        prior_returns + 2 * BASE_RETURN_RATE_PRIOR
    ) / (prior_orders + 2)
    f["is_first_order"] = (prior_orders == 0).astype(int)
    f["ever_returned"] = (prior_returns > 0).astype(int)

    # Shield membership
    f["shield_member"] = (df.get("shield_member") == "Y").astype(int)

    # Customer tenure (customers.csv is a *current* snapshot — F6)
    tenure = (placed - signup).dt.days
    f["signup_after_order"] = (tenure < 0).astype(int)
    f["tenure_days"] = tenure.clip(lower=0)
    f["state"] = df.get("state")

    # Product
    f["sku"] = df["sku"]
    f["family"] = df.get("family")
    f["tier"] = df.get("model_name").map(_extract_tier)
    f["list_price"] = pd.to_numeric(df.get("list_price_inr"), errors="coerce")
    f["warranty_months"] = pd.to_numeric(df.get("warranty_months"), errors="coerce")
    f["product_age_days"] = (placed - launch).dt.days

    # Apply fixed categories so training and single-row serving agree.
    for col in CATEGORICAL_COLUMNS:
        if categories is not None and col in categories:
            cats = categories[col]
        else:
            cats = sorted(f[col].dropna().unique().tolist())
        f[col] = pd.Categorical(f[col], categories=cats)

    return f


def fitted_categories(feature_df: pd.DataFrame) -> dict[str, list[str]]:
    """Capture the category list actually used for each categorical column,
    to persist in model_meta.json and reuse at serving time."""
    return {
        col: list(feature_df[col].cat.categories)
        for col in CATEGORICAL_COLUMNS
        if hasattr(feature_df[col], "cat")
    }
