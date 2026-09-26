# Submission form — Kestrel Home Returns Risk (Variant A)

> Note: the literal `submission-form.md` template wasn't among the files recovered for this
> session (data samples, README, ops-policy.pdf, email thread, sample_submission.csv were
> provided; a form template wasn't). This form covers everything the assignment brief itself
> asked the submission form to contain. If you have the actual template with different field
> labels, send it and I'll transpose these answers into it.

## 1. What I expect `predictions.csv` to score, and why

I expect the scores to rank orders correctly with an **ROC-AUC of roughly 0.75, likely
between 0.73 and 0.78**, on the hidden outcomes for `test_unlabelled.csv`.

That estimate comes from a 3-fold, time-based backtest (validating on Oct–Dec 2025,
Jan–Mar 2026, and Apr–Jun 2026 in turn, always training on everything earlier): AUC
0.741 / 0.759 / 0.758, mean **0.753 ± 0.008** — stable across quarters, not a lucky split.
The July–September 2026 test window is simply the next quarter in that same sequence, so I
expect similar performance, with two caveats:

- `test_unlabelled.csv` has a somewhat different channel mix than training (fewer
  partner-outlet orders) and includes one `last_service_event_type` value
  (`INSTALL_BOOKED`) never seen in training — moot for scoring, since that whole column is
  excluded as leakage (see below), but worth flagging as a sign the export snapshot differs
  slightly from training.
- The scores are **not** at their best on Robot Vacuum and Room Heater orders (segment AUC
  ≈0.71–0.75 versus ≈0.77 overall) — see `reports/EVIDENCE.md`.

**Accuracy is deliberately not the metric I optimised for or would offer as "proof it
works."** At an 11.4% base rate, predicting "nothing is returned" is already 88.5%
accurate and useless. I optimised ranking quality (AUC / PR-AUC) and a rupee-denominated
decision threshold instead — see `BLUEPRINT.md` §1.4 and `reports/EVIDENCE.md`.

Two columns in the raw export — `pickup_scheduled_at` and `last_service_event_type` — are
only ever populated *after* a return has already been raised, and are both empty (or, for
the second, out-of-distribution) in the test snapshot. A model that uses them scores 98.5%
in validation; I excluded both and treat that 98.5% figure as a demonstration of the trap,
not a result (see `reports/leakage_ablation.json`).

## 2. Stack

Python 3.11+, LightGBM (gradient-boosted trees, native categoricals, exact SHAP via
`pred_contrib`), scikit-learn (logistic-regression baseline, isotonic calibration),
pandas/numpy, FastAPI + vanilla HTML/JS for the service. No paid API required; an optional
LLM rephrasing step (`ANTHROPIC_API_KEY` + `USE_LLM=1`) is off by default.

## 3. How to run it

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# put the data files in data/
python train.py                    # writes artifacts/, reports/, predictions.csv
uvicorn app:app --port 8000         # open http://localhost:8000
pytest -q                           # 7 tests
```

Full instructions, troubleshooting and a curl example are in `README.md`.

## 4. Where the evidence is

`reports/EVIDENCE.md` — what works, how often it's wrong, where it's weakest, and the
leakage ablation, in plain language. Supporting data in `reports/metrics.json`,
`cost_curve.csv`, `deciles.csv`, `calibration.csv`, `backtest.csv`, `global_importance.csv`,
`hold_vs_call.csv`, `leakage_ablation.json`, and `eda.md`.

## 5. Known limitations

- `customers.csv` (Shield status, signup date) is a current snapshot, not point-in-time, so
  a customer's Shield status at scoring time may not exactly match what it was on an old
  training order.
- The 35% call-effectiveness figure is from a small spring pilot and should be re-measured
  once calls run at volume.
- The margin lost when a held order is cancelled (`M` in the hold-vs-call math) isn't given
  anywhere in the data pack; the model uses a placeholder and flags it — see
  `reports/hold_vs_call.csv` and the memo's ask to Finance.
- Precision at the operating threshold is modest (~1 in 4 calls goes to an order that
  wouldn't have returned) — expected at an 11% base rate, and why the action is a cheap
  call rather than a hold.

## 6. AI tools used, and how

Claude (Anthropic) was used throughout: to read and cross-check the data pack (train/test
CSVs, `customers.csv`, `products.csv`, `ops-policy.pdf`, the email thread) for the leakage,
duplication and currency issues described above; to design the feature set, validation
scheme and cost-based threshold; and to write `train.py`, `app.py`, `static/index.html`,
this form and the memo. All reported metrics were computed by actually running the training
pipeline against the provided data, not estimated or invented. I reviewed and take
responsibility for the analysis, the code, and the numbers in this submission.
