# Evidence it works, and how often it doesn't

## What works
- Holdout ROC-AUC: **0.768** (random = 0.5). PR-AUC: **0.366** (random = base rate = 0.114).
- Backtest across 3 rolling quarters: AUC 0.753 ± 0.008 — stable, not a one-off split.
- At the chosen threshold (0.112), calling the riskiest 32% of orders catches 68% of all returns, netting about ₹17139 per 1,000 orders.
- Top-risk decile returns at roughly the rate shown in reports/deciles.csv; the model separates risk, it doesn't just guess the base rate.

## How often it's wrong
- Precision at the threshold is **24%**: for every 4 calls placed, about 3.0 go to an order that would not have been returned anyway (still worth it — a call is cheap next to a return).
- Recall at the threshold is **68%**: about 32% of returns are not flagged and ship as normal.
- Accuracy at the threshold is 72.0%, next to 88.5% for guessing "nothing is returned" — a reminder that accuracy is the wrong headline number here.

## Where it's weakest
- family=Robot Vacuum: AUC 0.710
- family=Room Heater: AUC 0.754
- shield_member=0: AUC 0.755

## The 95%-accuracy trap, demonstrated
- A model that also sees `pickup_scheduled_at` (only ever set after a return is already raised) reaches **98.5%** accuracy in validation — that's how a fake 95%+ number happens.
- The same model without that column: **89.1%**. Guessing "nothing is returned": **88.5%**.
- `pickup_scheduled_at` is blank for 100% of `test_unlabelled.csv`, so the leaky model has nothing to go on at dispatch time — it would collapse to the base rate.

## Known limitations
- `customers.csv` (Shield membership, signup date) is a current snapshot, not point-in-time, so a customer's Shield status at scoring time may not match what it was on an old training order.
- The 35% call-effectiveness figure comes from a small spring pilot; it should be re-measured once calls run at volume (see memo.md).
- The margin lost on a cancelled hold (`M`) is not given anywhere in the data pack; ₹800 is a placeholder — see reports/hold_vs_call.csv for the sensitivity and memo.md for the ask to Finance.
- `test_unlabelled.csv` includes an `INSTALL_BOOKED` service-event value never seen in training; it is dropped along with the rest of that leaky column, so this has no effect on scoring, but is worth knowing about.
- Precision is modest (~1 in 4); this is expected at an 11% base rate and is why the action is a cheap call, not a hold.