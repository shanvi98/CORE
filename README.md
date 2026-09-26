# Kestrel Home — Returns Risk

A pre-dispatch returns-risk model plus a small service: one endpoint that
scores an order and explains why, and one screen that calls it. No paid API
key required. See [BLUEPRINT.md](BLUEPRINT.md) for the full design
rationale and [reports/EVIDENCE.md](reports/EVIDENCE.md) for how well it
actually works.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put the assignment's data files in `data/`: `train.csv`, `test_unlabelled.csv`,
`customers.csv`, `products.csv`, `sample_submission.csv`. `data/` is
git-ignored — this data must not be published or committed (ops-policy §10).

## Train

```bash
python train.py
```

Takes well under a minute. This cleans the data (drops leaked and
duplicated columns, fixes the October-2025 currency bug — see
BLUEPRINT.md §0), engineers features, validates on a time-based holdout
plus a 3-fold backtest, calibrates the score, picks a cost-based decision
threshold, and writes:

- `artifacts/` — `model.txt`, `calibrator.pkl`, `model_meta.json` (used by the API)
- `reports/` — `EVIDENCE.md`, `metrics.json`, `eda.md`, `cost_curve.csv`, `deciles.csv`,
  `calibration.csv`, `backtest.csv`, `global_importance.csv`, `hold_vs_call.csv`,
  `leakage_ablation.json`
- `predictions.csv` — one score per row of `test_unlabelled.csv`, in the
  shape of `sample_submission.csv`

## Run the service

```bash
uvicorn app:app --port 8000
```

Open <http://localhost:8000>. No API key is needed — the app scores orders
and explains them with plain-English templates. If you skip the `train.py`
step, the app still starts; it just answers every prediction with a clear
"model not loaded" message instead of crashing (try it: `mv artifacts
artifacts.bak`, restart, then `mv artifacts.bak artifacts` to put it back).

### Example request

```bash
curl -s localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "order_id": "KO2610504",
    "order_placed_at": "2026-07-01 00:51",
    "customer_id": "KC104433",
    "sku": "KH-AF-01",
    "sales_channel": "app",
    "payment_mode": "cod",
    "discount_pct": 10,
    "qty": 1,
    "order_value_inr": 4679.1,
    "promised_delivery_days": 9,
    "delivery_pincode": "440365",
    "is_gift": "N",
    "customer_prior_orders": 0,
    "customer_prior_returns": 0,
    "delivery_note": "Call before delivery"
  }' | python3 -m json.tool
```

Expected shape:

```json
{
  "order_id": "KO2610504",
  "score": 0.1656,
  "risk_band": "medium",
  "recommended_action": "CALL_BEFORE_DISPATCH",
  "threshold": 0.1118,
  "expected_saving_inr": 21.6,
  "reasons": [
    {"rank": 1, "factor": "Payment", "direction": "increases_risk", "impact": 0.64,
     "text": "Cash on delivery: COD orders come back 19% of the time vs 8% for UPI."},
    "... 2 more"
  ],
  "note": null,
  "warnings": [],
  "explanation_source": "template",
  "model_version": "lgbm-2026-09-26"
}
```

## Tests

```bash
pytest -q
```

Model-dependent tests are skipped (not failed) if `artifacts/` doesn't
exist yet — run `python train.py` first to get full coverage.

## Optional: LLM-rephrased reasons

The three reason sentences are plain templates by default. To have an LLM
rephrase them for tone (the underlying score and which three factors are
picked never change), set both:

```bash
export ANTHROPIC_API_KEY=sk-...
export USE_LLM=1
pip install anthropic   # not installed by default — the app works without it
```

If the key is missing, `USE_LLM` isn't set, the `anthropic` package isn't
installed, or the call fails or times out (3s), the service falls back to
the template text automatically — it never errors because of this.

## Troubleshooting

- **"Risk model not loaded"** — run `python train.py`, then restart `uvicorn`.
- **Port already in use** — `uvicorn app:app --port 8001` (or free port 8000).
- **`lightgbm` fails to import on macOS** (`Library not loaded: libomp.dylib`) —
  `brew install libomp`.
- **`ModuleNotFoundError`** — make sure the virtualenv is activated and
  `pip install -r requirements.txt` finished without errors.

## Deliverables map

| Deliverable | Where |
|---|---|
| `predictions.csv` | repo root, produced by `train.py` |
| The service | `app.py` (API) + `static/index.html` (UI) |
| Evidence it works, and how often it doesn't | `reports/EVIDENCE.md` |
| One-page memo to Ritu | `memo.md` |
| Submission form | `submission-form.md` |
| Technical spec / blueprint | `BLUEPRINT.md` |

## Data handling

Per ops-policy §10, Kestrel data must not be published, uploaded to a
public repository, or shared beyond the engagement team. `data/` and
`artifacts/` are git-ignored for that reason — don't remove them from
`.gitignore`.
