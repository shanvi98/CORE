# Submission form — Kestrel Home Returns Risk (Variant A)

## GitHub link (repo)

https://github.com/shanvi98/CORE

Note: `data/` and `artifacts/` are git-ignored on purpose (ops-policy §10 — Kestrel
data must not be published). Reviewers need to drop the assignment's CSVs into `data/`
and run `python train.py` before the app or tests have anything to show.

## What did you build, and what business decision does it support? State the number and the rupees.

A pre-dispatch returns-risk model plus a small service (`/predict` API + a one-screen UI)
that scores every order at the moment it's placed and recommends one of two actions:
**dispatch normally** or **call the customer to confirm before shipping**.

It supports one decision: **stop "flag and hold" and instead call the riskiest ~30% of
orders before dispatch, with Shield members never called or held.** At Kestrel's Finance
numbers (a return costs ₹1,150, a call costs ₹45 and prevents ~35% of the returns it's
made on), that nets **≈₹17,139 per 1,000 orders**, which at Kestrel's volume of ~700
orders/month is **≈₹144,000 a year**. Holding, by contrast, can save at most ₹138 per
returning order and loses money once the margin on a good order exceeds ~₹375 — so it's
recommended against outright. Full arithmetic in `memo.md` and `BLUEPRINT.md` §1.4.

## What score do you expect predictions.csv to get on the hidden outcomes, on which metric, and why that metric? Say how you estimated it.

**ROC-AUC ≈ 0.75, likely between 0.73 and 0.78.** Also expect PR-AUC around 0.36–0.37
(base rate ≈ 0.114) if that's checked too.

AUC/PR-AUC, not accuracy, because the base return rate is 11.4%: predicting "nothing
returns" is already 88.5% accurate and saves nobody anything. A model that *is* 95%+
accurate on this data is almost certainly reading `pickup_scheduled_at` or
`last_service_event_type` — both written only after a return is already raised, both
excluded here as leakage. Ranking quality is what the call/dispatch decision actually
depends on, so that's what's optimized and what's reported.

I estimated it with a 3-fold rolling time-based backtest (train on everything earlier,
validate on Oct–Dec 2025, then Jan–Mar 2026, then Apr–Jun 2026): AUC 0.741 / 0.759 / 0.758,
mean **0.753 ± 0.008**. The hidden Jul–Sep 2026 window is the next quarter in that same
sequence, so I expect similar performance, with two caveats noted in `reports/EVIDENCE.md`:
`test_unlabelled.csv` has a different channel mix (fewer partner-outlet orders, and one
`last_service_event_type` value never seen in training — moot, since that whole column
is dropped), and the model is weaker on Robot Vacuum / Room Heater orders (segment AUC
≈0.71–0.75 vs ≈0.77 overall).

## How do you know it works? How you validated, on what split, error rate, and the kind of case it gets wrong.

Validated two ways: a single time-based holdout (train through Mar 2026, validate on
Apr–Jun 2026 — never a random split, since a random split would leak future information
into training) and the 3-fold rolling backtest above, to confirm the holdout number wasn't
a lucky quarter.

At the chosen decision threshold (p ≥ 0.112):
- **Recall 68%** — the flagged ~32% of orders contain two-thirds of all returns.
- **Precision 24%** — of every 4 calls placed, about 1 goes to an order that would have
  returned anyway; the other 3 didn't need the call. Expected and acceptable at an 11%
  base rate, and exactly why the action is a ₹45 call, not a hold.
- **Accuracy 72%** (vs 88.5% for the naive "never returns" guess — reported only to show
  why accuracy is the wrong headline number, not as evidence of quality).

Where it gets it wrong: false negatives cluster in orders with low COD/discount/delivery-
delay signal but a return anyway (i.e., returns driven by product-condition or preference
reasons the order-level data can't see). Segment-wise it's weakest on Robot Vacuum
(AUC 0.71) and Room Heater (AUC 0.75) — both categories where returns look more like
product-fit issues than the operational risk factors (COD, slow delivery, gifting) the
model leans on elsewhere. See `reports/EVIDENCE.md` and `reports/deciles.csv`.

## Did you change, narrow, or push back on the client's ask? What, when, and why.

Yes, three things, all in the memo written for Ritu (Head of D2C Ops):

1. **Replaced "flag and hold" with "call before dispatch."** The original ask was to hold
   risky orders. The cost math (BLUEPRINT §1.4) shows holding only helps when a held
   customer would have cancelled anyway (~12% of the time) and saves at most ₹138 per
   prevented return, while angering good customers who get delayed for nothing. A ₹45
   confirmation call nets ₹402.5·p − 45 per order and beats holding at every value of p
   for any realistic lost-margin figure (M ≥ ₹375). The API ships with no HOLD action.
2. **Corrected the cost figure.** Ritu's ₹600 return-cost estimate is out of date; Finance's
   figure is ₹1,150. Using the old number would have roughly halved the calling threshold
   and the number of orders flagged.
3. **Pushed back on "95% accuracy."** No honest pre-dispatch model reaches that; only a
   leaky one does (98.5%, demonstrated and discarded). Reframed the board-facing metric as
   "share of returns caught" and "rupees saved" instead.

All three happened during the initial build (this submission), before any client
conversation — surfaced here and in `memo.md` for Ritu to react to.

## What is wrong with what you are handing us, or with the data we handed you? Be specific: bugs, shortcuts, columns you did not trust, rows that looked wrong.

- `pickup_scheduled_at` and `last_service_event_type` are target leakage (written after a
  return is raised) — dropped, not used anywhere in scoring.
- 651 `order_id`s in `train.csv` are exact duplicates from a partner-feed re-import
  (`source=crm` + `source=partner_feed`) — de-duplicated before any split.
- Every October-2025 `order_value_inr` is inflated exactly ×100 (payment-gateway bug,
  paise/rupee mixup) — corrected with an explicit divide-and-assert step, not silently.
- 889 rows have pincode `000000` (walk-in orders with no address) — flagged with a
  `pin_missing` feature rather than treated as a real geography.
- `customers.csv` (Shield status, signup date) is a current snapshot, not point-in-time —
  1,999 training orders have a `signup_date` after the order date. Tenure is clipped at 0
  and flagged; Shield status at scoring time may not match what it was historically. This
  is a real limitation I can't fully fix with the data given.
- Three rows of `delivery_note` free text contain strings addressed to "automated tools" —
  read as data only, never executed or fed to an LLM prompt; worth Kestrel knowing that
  attempted prompt injection exists in a field ops staff can write to.
- `M` (margin lost when a held order is cancelled) is not in the data pack at all — the
  hold-vs-call comparison uses a ₹800 placeholder pending Farhan confirming the real number.
- The 35% call-effectiveness figure is from a small spring pilot, not measured at volume.

## What does one prediction cost, and what would a month cost at Kestrel's volume (about 700 orders a month)?

**₹0 per prediction, ₹0/month.** Scoring is a local LightGBM model (`model.txt` + a
calibrator), no paid API call in the default path. The one optional paid component — an
LLM that rephrases the three reason sentences for tone — is off by default
(`USE_LLM` unset); the underlying score, action, and threshold never depend on it, and it
falls back to template text automatically if the key is missing or the call fails/times
out. If Kestrel turned it on for all ~700 orders/month using a small model like Claude
Haiku, that's roughly 700 short calls/month — well under $1/month — but it isn't part of
the number above because it isn't running.

## What did you deliberately leave out, and why that rather than something else?

- **The HOLD action.** The cost math shows it never beats calling for any realistic
  lost-margin figure, and it upsets customers who'd have kept the order. Left in the memo
  and config for Finance to check, not in the API's default behavior.
- **Raw `delivery_note` text as a model feature.** Every template category sits at
  10–13% return rate — no real signal — and free text is an unnecessary and unsafe surface
  to feed a model or LLM. Only a coarse `note_type` category is used.
- **`customer_id` target-encoding.** Prior-order and prior-return counts already carry a
  customer's history without needing to encode the ID directly, which would leak toward
  memorizing individual customers rather than generalizing.
- **An LLM anywhere in the scoring path.** It only ever rephrases already-computed reasons,
  never picks the action or the score, and is off by default — a paid, latency-adding,
  occasionally-wrong dependency isn't worth taking for a cosmetic change.
- I chose these four over, e.g., building a third "hold" tier or a fancier ensemble model,
  because none of them would have moved the rupee outcome — the entire value in this
  project is in the threshold/action decision and the leakage fixes, not model complexity.

## Anything you built or found that nobody asked for?

Found: the prompt-injection attempt embedded in three `delivery_note` rows (noted above) —
nobody asked me to look for that, but it's the kind of thing that matters once an LLM is
anywhere near this field.

Built: the app degrades instead of failing — if `artifacts/` is missing or renamed,
`/health` reports degraded and `/predict` returns a clear "model not loaded" message
instead of a crash or a 500. Also built the leakage demonstration itself
(`reports/leakage_ablation.json` — the model scoring 98.5% with the leaky columns,
89.1% without, 88.5% for guessing "nothing returns") as a standalone artifact, since
"we checked and it's a trap" is a much stronger claim with the number attached than
without it.

## What did you use AI for? Which tools and models, where they helped, where they misled you, what you threw away. Link your three-minute screen recording here.

Claude (Sonnet, via Claude Code) throughout: reading and cross-checking the data pack
(`train.csv`/`test_unlabelled.csv`, `customers.csv`, `products.csv`, `ops-policy.pdf`, the
email thread) for the leakage, duplication, and currency issues above; designing the
feature set, validation scheme, and cost-based threshold; and writing `train.py`, `app.py`,
`static/index.html`, and this form and the memo.

Where it helped: it caught the leakage columns and the duplicate rows quickly by
cross-referencing the ops-policy doc against column definitions, and it was faster than I'd
have been at writing the FastAPI boilerplate, the SHAP reason-grouping code, and the tests.

Where it misled me: an early draft suggested target-encoding `customer_id` and using the
full `delivery_note` text as a feature — both looked like free signal, both were reverted
once checked against the base-rate-per-category numbers and the leakage risk of encoding
an ID directly. I also had to push back on a first pass that buried the "95% is a trap"
point instead of leading with it.

Thrown away: the leaky 98.5%-accuracy model, an early version with a HOLD action, and a
draft that put LLM rephrasing in the critical scoring path instead of behind a flag.

Screen recording: **[(https://drive.google.com/drive/folders/1R0uTrjiRstVakM-m_aWhNSVgGbKJamXl?usp=sharing)]**

## Your Public Google Drive Link

**[ADD LINK — not yet created]**

## Someone picks this up on Monday and you are unreachable. The three things they need to know.

1. **`data/` and `artifacts/` are git-ignored, not missing.** Put the assignment's CSVs in
   `data/`, run `python train.py` to regenerate `artifacts/`, `reports/`, and
   `predictions.csv` before touching the app or tests — nothing in the repo commits raw
   Kestrel data (ops-policy §10).
2. **There is no HOLD action by default, and Shield members are never called or held.**
   That's a deliberate decision backed by the cost math in `BLUEPRINT.md` §1.4 and
   `memo.md`, not an oversight — don't "restore" it without redoing that math.
3. **Two numbers in the model are placeholders, not measured facts:** the 35%
   call-effectiveness rate is from a small spring pilot (needs re-measuring once calls run
   at volume), and `M` (margin lost on a cancelled hold, ₹800) was never given in the data
   pack — it's a guess pending Farhan's answer. Don't quote either to the board as final.
