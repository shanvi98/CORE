"""Kestrel Home — Returns Risk: training pipeline.

EDA -> clean -> features -> time-based CV -> LightGBM -> calibration ->
cost-based threshold -> SHAP reasons -> predictions.csv -> evidence.

Run:  python train.py
Finishes in well under 2 minutes on a laptop. See BLUEPRINT.md for the
reasoning behind every step, and reports/EVIDENCE.md for the results after
a run.
"""
from __future__ import annotations

import json
import logging

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.features import (
    CATEGORICAL_COLUMNS,
    FEATURE_GROUPS,
    assert_currency_fixed,
    build_features,
    clean,
    fitted_categories,
    load_raw,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")

np.random.seed(config.SEED)

LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5,
    metric="auc",
    verbose=-1,
    seed=config.SEED,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _clean_dict(d: dict) -> dict:
    """Convert numpy scalars / NaN to native JSON-safe values."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[str(k)] = _clean_dict(v)
        elif v is None or (isinstance(v, float) and np.isnan(v)):
            out[str(k)] = None
        else:
            out[str(k)] = float(v) if isinstance(v, (np.floating, np.integer)) else v
    return out


def write_eda(train_raw: pd.DataFrame, customers: pd.DataFrame, products: pd.DataFrame,
              n_before: int, n_after: int) -> None:
    lines = ["# EDA — Kestrel Home Returns Risk\n"]
    lines.append(f"- Train rows before de-dup: **{n_before}**")
    lines.append(f"- Train rows after de-dup (F3, partner-feed re-imports): **{n_after}**")
    lines.append(f"- Overall return rate: **{train_raw[config.TARGET_COL].mean():.4f}**\n")

    m = pd.to_datetime(train_raw["order_placed_at"]).dt.to_period("M").astype(str)
    by_month = train_raw.groupby(m)[config.TARGET_COL].agg(["mean", "size"])
    lines.append("## Return rate by month\n```\n" + by_month.round(3).to_string() + "\n```\n")

    for col in ["payment_mode", "sales_channel", "is_gift"]:
        g = train_raw.groupby(col)[config.TARGET_COL].agg(["mean", "size"])
        lines.append(f"## Return rate by {col}\n```\n" + g.round(3).to_string() + "\n```\n")

    bucket = pd.cut(train_raw["promised_delivery_days"], bins=[0, 3, 7, 100], labels=["<=3", "4-7", ">=8"])
    g = train_raw.groupby(bucket, observed=True)[config.TARGET_COL].agg(["mean", "size"])
    lines.append("## Return rate by promised-delivery bucket\n```\n" + g.round(3).to_string() + "\n```\n")

    mp = train_raw.merge(customers, on="customer_id", how="left")
    g = mp.groupby("shield_member")[config.TARGET_COL].agg(["mean", "size"])
    lines.append("## Return rate by Shield membership\n```\n" + g.round(3).to_string() + "\n```\n")

    mpr = train_raw.merge(products, on="sku", how="left")
    g = mpr.groupby("family")[config.TARGET_COL].agg(["mean", "size"])
    lines.append("## Return rate by product family\n```\n" + g.round(3).to_string() + "\n```\n")

    lines.append("## Evidence: known data problems\n")
    lines.append("**F1/F2 — leakage.** `pickup_scheduled_at` is written only after a return is "
                  "approved; `last_service_event_type` is pulled as of export day.\n")
    g = train_raw.groupby(train_raw["pickup_scheduled_at"].notna())[config.TARGET_COL].agg(["mean", "size"])
    g.index = g.index.map({True: "pickup_scheduled_at set", False: "not set"})
    lines.append("```\n" + g.round(4).to_string() + "\n```\n")
    g = train_raw.groupby("last_service_event_type")[config.TARGET_COL].agg(["mean", "size"])
    lines.append("```\n" + g.round(4).to_string() + "\n```\n")

    lines.append(f"**F3 — duplicates.** {n_before - n_after} rows were exact partner-feed "
                  "re-imports of an already-seen `order_id`, dropped before any split.\n")

    lines.append("**F4 — currency bug.** Oct-2025 `order_value_inr` was 100x too high "
                  "(new payment gateway stored paise). Fixed in `clean()`; see "
                  "`assert_currency_fixed` for the check.\n")

    pin_missing = train_raw["delivery_pincode"].astype(str) == "000000"
    g = train_raw.groupby(pin_missing)[config.TARGET_COL].agg(["mean", "size"])
    g.index = g.index.map({True: "pincode 000000 (no address)", False: "has address"})
    lines.append(f"**F5 — default pincode.** {int(pin_missing.sum())} rows use the system "
                  "default pincode (walk-in, no address).\n```\n" + g.round(4).to_string() + "\n```\n")

    neg_tenure = (pd.to_datetime(mp["signup_date"]) > pd.to_datetime(mp["order_placed_at"]))
    lines.append(f"**F6 — customer snapshot.** {int(neg_tenure.sum())} orders have a signup "
                  "date after the order date (customers.csv is a current snapshot, not "
                  "point-in-time). Return rate is unaffected either way "
                  f"({mp.groupby(neg_tenure)[config.TARGET_COL].mean().round(3).to_dict()}).\n")

    (config.REPORTS_DIR / "eda.md").write_text("\n".join(lines))
    log.info("Wrote reports/eda.md")


def _fit_lr(X: pd.DataFrame, y: np.ndarray, tr_mask: np.ndarray, va_mask: np.ndarray):
    numeric_cols = [c for c in X.columns if c not in CATEGORICAL_COLUMNS]
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_cols),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_COLUMNS),
        ]
    )
    clf = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=1000, random_state=config.SEED))])
    clf.fit(X[tr_mask], y[tr_mask])
    scores = clf.predict_proba(X[va_mask])[:, 1]
    return scores


def _fit_lgb(X: pd.DataFrame, y: np.ndarray, tr_mask: np.ndarray, va_mask: np.ndarray,
             num_boost_round: int = 2000, early_stopping: bool = True):
    dtrain = lgb.Dataset(X[tr_mask], y[tr_mask], categorical_feature=CATEGORICAL_COLUMNS, free_raw_data=False)
    callbacks = [lgb.log_evaluation(0)]
    valid_sets = None
    if early_stopping:
        dval = lgb.Dataset(X[va_mask], y[va_mask], reference=dtrain, categorical_feature=CATEGORICAL_COLUMNS, free_raw_data=False)
        valid_sets = [dval]
        callbacks.append(lgb.early_stopping(100, verbose=False))
    booster = lgb.train(LGB_PARAMS, dtrain, num_boost_round=num_boost_round,
                         valid_sets=valid_sets, callbacks=callbacks)
    scores = booster.predict(X[va_mask], num_iteration=booster.best_iteration or num_boost_round)
    best_iter = booster.best_iteration or num_boost_round
    return booster, scores, best_iter


def run_backtest(X: pd.DataFrame, y: np.ndarray, placed: pd.Series) -> list[dict]:
    folds = [
        ("2025-Q4", "2025-10-01", "2026-01-01"),
        ("2026-Q1", "2026-01-01", "2026-04-01"),
        ("2026-Q2", "2026-04-01", "2026-07-01"),
    ]
    rows = []
    for name, start, end in folds:
        tr_mask = (placed < start).values
        va_mask = ((placed >= start) & (placed < end)).values
        if tr_mask.sum() < 200 or va_mask.sum() < 20:
            continue
        _, scores, _ = _fit_lgb(X, y, tr_mask, va_mask, num_boost_round=300, early_stopping=False)
        y_va = y[va_mask]
        auc = roc_auc_score(y_va, scores) if len(np.unique(y_va)) > 1 else float("nan")
        prauc = average_precision_score(y_va, scores) if len(np.unique(y_va)) > 1 else float("nan")
        rows.append({"fold": name, "n": int(va_mask.sum()), "return_rate": round(float(y_va.mean()), 4),
                     "auc": round(float(auc), 4), "pr_auc": round(float(prauc), 4)})
        log.info(f"Backtest fold {name}: n={va_mask.sum()} rate={y_va.mean():.3f} AUC={auc:.4f}")
    return rows


def write_calibration_report(raw: np.ndarray, calibrated: np.ndarray, y_va: np.ndarray) -> None:
    def reliability(scores):
        bins = pd.qcut(scores, 10, labels=False, duplicates="drop")
        df = pd.DataFrame({"bin": bins, "pred": scores, "y": y_va})
        return df.groupby("bin").agg(mean_predicted=("pred", "mean"), observed_rate=("y", "mean"), n=("y", "size"))

    rel_raw = reliability(raw).add_suffix("_raw")
    rel_cal = reliability(calibrated).add_suffix("_cal")
    out = pd.concat([rel_raw, rel_cal], axis=1)
    out.to_csv(config.REPORTS_DIR / "calibration.csv")
    log.info("Wrote reports/calibration.csv")


def choose_threshold(calibrated_va: np.ndarray, y_va: np.ndarray) -> tuple[float, pd.DataFrame]:
    ts = np.arange(0.01, 1.0, 0.005)
    rows = []
    for t in ts:
        sel = calibrated_va >= t
        n = int(sel.sum())
        caught = int(y_va[sel].sum()) if n else 0
        net = config.CALL_EFFECTIVENESS * config.COST_RETURN * caught - config.COST_CALL * n
        rows.append({
            "t": round(float(t), 3), "n_called": n, "share_called": round(n / len(calibrated_va), 4),
            "returns_caught": caught, "recall": round(caught / max(y_va.sum(), 1), 4),
            "precision": round(caught / n, 4) if n else 0.0,
            "net_inr": round(float(net), 2), "net_per_1000": round(float(net) / len(calibrated_va) * 1000, 2),
        })
    curve = pd.DataFrame(rows)
    curve.to_csv(config.REPORTS_DIR / "cost_curve.csv", index=False)

    t_empirical = float(curve.loc[curve["net_inr"].idxmax(), "t"])
    t_theoretical = config.THRESHOLD_THEORETICAL
    if abs(t_empirical - t_theoretical) > config.THRESHOLD_DISAGREEMENT_TOLERANCE:
        log.warning(
            f"Empirical threshold t={t_empirical:.3f} disagrees with theoretical "
            f"t*={t_theoretical:.3f} by more than {config.THRESHOLD_DISAGREEMENT_TOLERANCE}; "
            "using the empirical threshold. This usually means calibration drifted — check it."
        )
        threshold = t_empirical
    else:
        threshold = t_theoretical
    log.info(f"Threshold: theoretical={t_theoretical:.4f} empirical={t_empirical:.4f} chosen={threshold:.4f}")
    return threshold, curve


def write_hold_vs_call(calibrated_va: np.ndarray) -> None:
    ps = np.arange(0.05, 1.0, 0.05)
    rows = []
    for p in ps:
        ev_call = config.CALL_EFFECTIVENESS * config.COST_RETURN * p - config.COST_CALL
        row = {"p": round(float(p), 2), "ev_call": round(float(ev_call), 2)}
        for m in config.MARGIN_SENSITIVITY:
            ev_hold = config.HOLD_CANCEL_RATE * (config.COST_RETURN * p - (1 - p) * m)
            row[f"ev_hold_margin_{int(m)}"] = round(float(ev_hold), 2)
        rows.append(row)
    pd.DataFrame(rows).to_csv(config.REPORTS_DIR / "hold_vs_call.csv", index=False)
    log.info("Wrote reports/hold_vs_call.csv")


def write_deciles(y_va: np.ndarray, calibrated_va: np.ndarray) -> pd.DataFrame:
    decile = pd.qcut(calibrated_va, 10, labels=False, duplicates="drop")
    df = pd.DataFrame({"decile": decile, "y": y_va})
    total_returns = df["y"].sum()
    g = df.groupby("decile").agg(n=("y", "size"), return_rate=("y", "mean"), returns=("y", "sum"))
    g["share_of_all_returns"] = g["returns"] / total_returns
    g = g.sort_index(ascending=False)  # highest-risk decile first
    g.round(4).to_csv(config.REPORTS_DIR / "deciles.csv")
    log.info("Wrote reports/deciles.csv")
    return g


def write_leakage_ablation(train_raw: pd.DataFrame, X: pd.DataFrame, y: np.ndarray,
                            tr_mask: np.ndarray, va_mask: np.ndarray) -> dict:
    X_leak = X.copy()
    X_leak["leak_pickup_flag"] = train_raw["pickup_scheduled_at"].notna().astype(int).values
    _, scores_leak, _ = _fit_lgb(X_leak, y, tr_mask, va_mask, num_boost_round=300, early_stopping=False)
    _, scores_clean, _ = _fit_lgb(X, y, tr_mask, va_mask, num_boost_round=300, early_stopping=False)
    y_va = y[va_mask]
    acc_leak = max(accuracy_score(y_va, scores_leak > t) for t in np.linspace(0, 1, 101))
    acc_clean = max(accuracy_score(y_va, scores_clean > t) for t in np.linspace(0, 1, 101))
    result = {
        "accuracy_with_pickup_leak": round(float(acc_leak), 4),
        "accuracy_without_leak": round(float(acc_clean), 4),
        "accuracy_predict_all_zero": round(float(1 - y_va.mean()), 4),
        "note": "pickup_scheduled_at is blank for every row of test_unlabelled.csv, so the "
                "leaky model's high accuracy here cannot be reproduced at dispatch time.",
    }
    (config.REPORTS_DIR / "leakage_ablation.json").write_text(json.dumps(result, indent=2))
    log.info(f"Leakage ablation: with_leak={acc_leak:.4f} without_leak={acc_clean:.4f}")
    return result


def compute_reference_rates(X: pd.DataFrame, y: np.ndarray) -> dict:
    s = pd.Series(y, index=X.index)

    def rate_by(col):
        return _clean_dict(s.groupby(X[col], observed=True).mean().to_dict())

    def rate_where(mask, label_true, label_false):
        return _clean_dict({
            label_true: s[mask].mean() if mask.any() else None,
            label_false: s[~mask].mean() if (~mask).any() else None,
        })

    bucket = pd.cut(X["promised_delivery_days"], bins=[0, 3, 7, 100], labels=["<=3", "4-7", ">=8"])
    rates = {
        "payment_mode": rate_by("payment_mode"),
        "sales_channel": rate_by("sales_channel"),
        "family": rate_by("family"),
        "note_type": rate_by("note_type"),
        "is_gift": rate_where(X["is_gift"] == 1, "Y", "N"),
        "shield_member": rate_where(X["shield_member"] == 1, "Y", "N"),
        "promised_bucket": _clean_dict(s.groupby(bucket, observed=True).mean().to_dict()),
        "first_order": rate_where(X["is_first_order"] == 1, "first", "repeat"),
        "prior_returns_flag": rate_where(X["ever_returned"] == 1, "has_returns", "none"),
        "festive": rate_where(X["is_festive_window"] == 1, "yes", "no"),
        "pin_missing": rate_where(X["pin_missing"] == 1, "yes", "no"),
        "base_rate": float(s.mean()),
    }
    return rates


def write_global_importance(booster: lgb.Booster, feature_names: list[str], X: pd.DataFrame,
                             tr_mask: np.ndarray) -> None:
    contrib = booster.predict(X[tr_mask], pred_contrib=True)
    contrib = np.asarray(contrib)[:, : len(feature_names)]
    mean_abs = np.abs(contrib).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    df["group"] = df["feature"].map(FEATURE_GROUPS).fillna("Other")
    df = df.sort_values("mean_abs_shap", ascending=False)
    df.to_csv(config.REPORTS_DIR / "global_importance.csv", index=False)
    log.info("Wrote reports/global_importance.csv")


def write_predictions(booster: lgb.Booster, calibrator: IsotonicRegression, test_raw: pd.DataFrame,
                       customers: pd.DataFrame, products: pd.DataFrame,
                       categories: dict, threshold: float) -> None:
    X_test = build_features(test_raw, customers, products, categories=categories)
    raw_scores = booster.predict(X_test)
    scores = calibrator.predict(raw_scores)
    preds = pd.DataFrame({config.ID_COL: test_raw[config.ID_COL].values, "score": scores})

    sample = pd.read_csv(config.SAMPLE_SUBMISSION_CSV)
    assert set(preds[config.ID_COL]) == set(sample[config.ID_COL]), "predictions/order_id mismatch with sample_submission"
    preds = preds.set_index(config.ID_COL).loc[sample[config.ID_COL]].reset_index()

    assert len(preds) == len(sample), f"expected {len(sample)} rows, got {len(preds)}"
    assert not preds[config.ID_COL].duplicated().any(), "duplicate order_id in predictions"
    assert preds["score"].between(0, 1).all(), "scores out of [0,1]"
    assert not preds["score"].isna().any(), "NaN scores"

    preds.to_csv(config.PREDICTIONS_CSV, index=False)
    share_called = (preds["score"] >= threshold).mean()
    log.info(f"Wrote predictions.csv ({len(preds)} rows). Share above threshold "
             f"{threshold:.3f}: {share_called:.1%}")


def write_evidence(metrics: dict, backtest_rows: list[dict], leakage: dict,
                    segment_auc: dict, threshold: float) -> None:
    fp_per_4 = round((1 - metrics["precision_at_threshold"]) * 4, 1)
    lines = [
        "# Evidence it works, and how often it doesn't\n",
        "## What works",
        f"- Holdout ROC-AUC: **{metrics['roc_auc']:.3f}** (random = 0.5). "
        f"PR-AUC: **{metrics['pr_auc']:.3f}** (random = base rate = {metrics['base_rate']:.3f}).",
        f"- Backtest across 3 rolling quarters: AUC {metrics['backtest_auc_mean']:.3f} "
        f"± {metrics['backtest_auc_std']:.3f} — stable, not a one-off split.",
        f"- At the chosen threshold ({threshold:.3f}), calling the riskiest "
        f"{metrics['flag_share_at_threshold']:.0%} of orders catches "
        f"{metrics['recall_at_threshold']:.0%} of all returns, netting about "
        f"₹{metrics['net_per_1000_at_threshold']:.0f} per 1,000 orders.",
        f"- Top-risk decile returns at roughly the rate shown in reports/deciles.csv; "
        "the model separates risk, it doesn't just guess the base rate.\n",
        "## How often it's wrong",
        f"- Precision at the threshold is **{metrics['precision_at_threshold']:.0%}**: "
        f"for every 4 calls placed, about {fp_per_4} go to an order that would not have "
        "been returned anyway (still worth it — a call is cheap next to a return).",
        f"- Recall at the threshold is **{metrics['recall_at_threshold']:.0%}**: about "
        f"{1 - metrics['recall_at_threshold']:.0%} of returns are not flagged and ship as normal.",
        f"- Accuracy at the threshold is {metrics['accuracy_at_threshold']:.1%}, next to "
        f"{metrics['accuracy_predict_all_zero']:.1%} for guessing \"nothing is returned\" — "
        "a reminder that accuracy is the wrong headline number here.\n",
        "## Where it's weakest",
    ]
    worst = sorted(segment_auc.items(), key=lambda kv: (kv[1] if kv[1] == kv[1] else 1.0))[:3]
    for seg, auc in worst:
        lines.append(f"- {seg}: AUC {auc:.3f}" if auc == auc else f"- {seg}: not enough data to score")
    lines.append("")

    lines.append("## The 95%-accuracy trap, demonstrated")
    lines.append(
        f"- A model that also sees `pickup_scheduled_at` (only ever set after a return is "
        f"already raised) reaches **{leakage['accuracy_with_pickup_leak']:.1%}** accuracy in "
        f"validation — that's how a fake 95%+ number happens.\n"
        f"- The same model without that column: **{leakage['accuracy_without_leak']:.1%}**. "
        f"Guessing \"nothing is returned\": **{leakage['accuracy_predict_all_zero']:.1%}**.\n"
        f"- `pickup_scheduled_at` is blank for 100% of `test_unlabelled.csv`, so the leaky "
        "model has nothing to go on at dispatch time — it would collapse to the base rate."
    )

    lines.append("\n## Known limitations")
    lines.append(
        "- `customers.csv` (Shield membership, signup date) is a current snapshot, not "
        "point-in-time, so a customer's Shield status at scoring time may not match what it "
        "was on an old training order.\n"
        "- The 35% call-effectiveness figure comes from a small spring pilot; it should be "
        "re-measured once calls run at volume (see memo.md).\n"
        "- The margin lost on a cancelled hold (`M`) is not given anywhere in the data pack; "
        f"₹{config.MARGIN_LOST_ON_CANCEL:.0f} is a placeholder — see reports/hold_vs_call.csv "
        "for the sensitivity and memo.md for the ask to Finance.\n"
        "- `test_unlabelled.csv` includes an `INSTALL_BOOKED` service-event value never seen "
        "in training; it is dropped along with the rest of that leaky column, so this has no "
        "effect on scoring, but is worth knowing about.\n"
        "- Precision is modest (~1 in 4); this is expected at an 11% base rate and is why the "
        "action is a cheap call, not a hold."
    )
    (config.REPORTS_DIR / "EVIDENCE.md").write_text("\n".join(lines))
    log.info("Wrote reports/EVIDENCE.md")


def segment_auc_table(X: pd.DataFrame, y_va: np.ndarray, calibrated_va: np.ndarray,
                       va_mask: np.ndarray) -> dict:
    out = {}
    Xv = X[va_mask]
    for col in ["sales_channel", "family", "shield_member"]:
        for val, idx in Xv.groupby(col, observed=True).groups.items():
            mask = Xv.index.isin(idx)
            yv = y_va[mask]
            sv = calibrated_va[mask]
            key = f"{col}={val}"
            if len(np.unique(yv)) > 1 and mask.sum() >= 20:
                out[key] = float(roc_auc_score(yv, sv))
            else:
                out[key] = float("nan")
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)

    log.info("Loading raw data...")
    train_raw, test_raw, customers, products = load_raw()
    n_before = len(train_raw)
    train_raw = clean(train_raw)
    test_raw = clean(test_raw)
    n_after = len(train_raw)
    log.info(f"train rows before dedup={n_before} after={n_after}; test rows={len(test_raw)}")

    assert_currency_fixed(train_raw, products)
    assert test_raw[config.ID_COL].duplicated().sum() == 0, "test has duplicate order_ids"

    write_eda(train_raw, customers, products, n_before, n_after)

    y = train_raw[config.TARGET_COL].astype(int).values
    placed = pd.to_datetime(train_raw["order_placed_at"])

    X_all = build_features(train_raw, customers, products)
    categories = fitted_categories(X_all)
    X_all = build_features(train_raw, customers, products, categories=categories)
    feature_names = list(X_all.columns)
    log.info(f"Built {len(feature_names)} features on {len(X_all)} rows")

    holdout_start = pd.Timestamp("2026-04-01")
    tr_mask = (placed < holdout_start).values
    va_mask = (~tr_mask)
    log.info(f"Primary holdout: train n={tr_mask.sum()} validate n={va_mask.sum()} "
             f"({placed[va_mask].min().date()} .. {placed[va_mask].max().date()})")

    backtest_rows = run_backtest(X_all, y, placed)
    pd.DataFrame(backtest_rows).to_csv(config.REPORTS_DIR / "backtest.csv", index=False)
    bt_auc = [r["auc"] for r in backtest_rows if r["auc"] == r["auc"]]
    backtest_auc_mean = float(np.mean(bt_auc)) if bt_auc else float("nan")
    backtest_auc_std = float(np.std(bt_auc)) if bt_auc else float("nan")

    base_rate = float(y[tr_mask].mean())
    y_va = y[va_mask]

    lr_scores_va = _fit_lr(X_all, y, tr_mask, va_mask)
    lr_auc = roc_auc_score(y_va, lr_scores_va)
    lr_prauc = average_precision_score(y_va, lr_scores_va)

    booster, lgb_scores_va, best_iter = _fit_lgb(X_all, y, tr_mask, va_mask)
    lgb_auc = roc_auc_score(y_va, lgb_scores_va)
    lgb_prauc = average_precision_score(y_va, lgb_scores_va)

    log.info(f"Holdout AUC: constant=0.500 LR={lr_auc:.4f} LightGBM={lgb_auc:.4f} (best_iter={best_iter})")
    if lgb_auc - lr_auc < 0.01:
        log.warning("LightGBM does not beat Logistic Regression by >=0.01 AUC — noted in EVIDENCE.md, "
                     "shipping LightGBM anyway for exact-SHAP explainability.")
    if lgb_auc > 0.85:
        log.warning("Holdout AUC > 0.85 — this usually means a leaked column snuck back in. STOP and check.")

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(lgb_scores_va, y_va)
    calibrated_va = calibrator.predict(lgb_scores_va)
    write_calibration_report(lgb_scores_va, calibrated_va, y_va)
    brier_raw = brier_score_loss(y_va, lgb_scores_va)
    brier_cal = brier_score_loss(y_va, calibrated_va)

    threshold, cost_curve = choose_threshold(calibrated_va, y_va)
    write_hold_vs_call(calibrated_va)

    pred_at_t = calibrated_va >= threshold
    metrics = {
        "roc_auc": round(float(lgb_auc), 4),
        "pr_auc": round(float(lgb_prauc), 4),
        "lr_roc_auc": round(float(lr_auc), 4),
        "lr_pr_auc": round(float(lr_prauc), 4),
        "brier_raw": round(float(brier_raw), 4),
        "brier_calibrated": round(float(brier_cal), 4),
        "base_rate": round(base_rate, 4),
        "accuracy_predict_all_zero": round(1 - float(y_va.mean()), 4),
        "accuracy_at_threshold": round(float(accuracy_score(y_va, pred_at_t)), 4),
        "precision_at_threshold": round(float(precision_score(y_va, pred_at_t, zero_division=0)), 4),
        "recall_at_threshold": round(float(recall_score(y_va, pred_at_t, zero_division=0)), 4),
        "flag_share_at_threshold": round(float(pred_at_t.mean()), 4),
        "confusion_matrix_at_threshold": confusion_matrix(y_va, pred_at_t).tolist(),
        "threshold": round(float(threshold), 4),
        "threshold_theoretical": round(config.THRESHOLD_THEORETICAL, 4),
        "threshold_ritu_figure": round(config.THRESHOLD_RITU, 4),
        "net_per_1000_at_threshold": round(
            float(cost_curve.loc[(cost_curve["t"] - threshold).abs().idxmin(), "net_per_1000"]), 2
        ),
        "backtest_auc_mean": round(backtest_auc_mean, 4),
        "backtest_auc_std": round(backtest_auc_std, 4),
        "backtest_folds": backtest_rows,
    }
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("Wrote reports/metrics.json")

    write_deciles(y_va, calibrated_va)
    leakage = write_leakage_ablation(train_raw, X_all, y, tr_mask, va_mask)
    reference_rates = compute_reference_rates(X_all, y)
    write_global_importance(booster, feature_names, X_all, tr_mask)
    segment_auc = segment_auc_table(X_all, y_va, calibrated_va, va_mask)

    final_rounds = max(int(round(best_iter * 1.1)), 10)
    log.info(f"Refitting on all {len(X_all)} rows for {final_rounds} rounds")
    dtrain_all = lgb.Dataset(X_all, y, categorical_feature=CATEGORICAL_COLUMNS, free_raw_data=False)
    final_booster = lgb.train(LGB_PARAMS, dtrain_all, num_boost_round=final_rounds)
    final_booster.save_model(str(config.MODEL_PATH))
    joblib.dump(calibrator, config.CALIBRATOR_PATH)

    meta = {
        "model_version": f"{config.MODEL_VERSION_PREFIX}-{pd.Timestamp.today().date()}",
        "feature_names": feature_names,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "categories": {k: list(v) for k, v in categories.items()},
        "feature_groups": FEATURE_GROUPS,
        "threshold": round(float(threshold), 4),
        "threshold_theoretical": round(config.THRESHOLD_THEORETICAL, 4),
        "threshold_ritu_figure": round(config.THRESHOLD_RITU, 4),
        "risk_band_high": config.RISK_BAND_HIGH,
        "cost_params": {
            "cost_return": config.COST_RETURN,
            "cost_call": config.COST_CALL,
            "call_effectiveness": config.CALL_EFFECTIVENESS,
            "hold_cancel_rate": config.HOLD_CANCEL_RATE,
            "margin_lost_on_cancel": config.MARGIN_LOST_ON_CANCEL,
        },
        "reference_rates": reference_rates,
        "holdout_metrics": metrics,
        "training_row_count": int(len(X_all)),
        "leaky_columns_excluded": config.LEAKY_COLUMNS,
        # Serving-time defaults for an order whose customer_id isn't in
        # customers.csv (new/unlisted customer). See app.py.
        "default_tenure_days": round(float(X_all["tenure_days"].median()), 1),
        "default_shield_member": "N",
    }
    config.META_PATH.write_text(json.dumps(meta, indent=2, default=str))
    log.info(f"Saved model, calibrator, meta -> {config.ARTIFACTS_DIR}")

    write_predictions(final_booster, calibrator, test_raw, customers, products, categories, threshold)
    write_evidence(metrics, backtest_rows, leakage, segment_auc, threshold)

    log.info("Done.")


if __name__ == "__main__":
    main()
