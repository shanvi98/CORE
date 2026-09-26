# Returns risk — what to do next week

**To:** Ritu Deshpande, Head of D2C Operations
**From:** Kabir Nanda
**Re:** The returns model you asked for

## The decision

Don't hold orders at dispatch — it costs more than it saves. Instead, **call the riskiest orders before we ship them**, the way the spring pilot did. Shield members are never called or held; returns are free for them and Meenal's team will hear about it otherwise. Run this as a **4-week pilot** first.

## The number

"95% accuracy" isn't the right bar, and no honest model can hit it. Only 1 order in 9 comes back, so predicting *nothing* ever gets returned is already 88% accurate and catches nothing. The only way to reach 95%+ is to let the model see information we only have *after* a customer has already asked to return the order — useless at dispatch time. We checked: that fake version scores 98%.

The honest number: **calling the riskiest third of orders catches about two-thirds of all our returns.** The riskiest 10% of orders return about half the time; the safest 10%, about 1 in 50. For the board, we'd suggest "share of returns caught" instead of "accuracy" — that's the number that moves the P&L.

## The rupees

We used Finance's figure: a return costs **₹1,150** all-in, not the ₹600 first quoted. A confirmation call costs **₹45** and stops about **35%** of the returns it's made on — it pays for itself on any order with better than roughly 1-in-9 odds of coming back, comfortably true for our flagged third.

That call plan nets **≈₹17,000 per 1,000 orders**. At our volume (~700 orders/month), that's roughly **₹140,000 a year**, and the model itself costs nothing per order to run.

Holding loses money by comparison. It only helps when a held customer would have cancelled anyway (about 12% of the time), and saves at most ₹138 per order that would have returned. Weigh in the good orders we'd lose by holding them, and calling wins outright once the margin on an order is worth more than a few hundred rupees.

## What to do next week

1. **Ops:** start calling flagged orders — same vendor and script as the spring pilot.
2. **Farhan:** confirm the margin lost on a cancelled order, so we can formally close the "hold" option.
3. **Meenal:** brief the desk — Shield customers get a call at most, never a hold.
4. **Log every call outcome** — the 35%-prevented figure is from a small pilot; four weeks of real calls will give us the true number.
5. **Tanmay:** fix two issues at the source — the October payment-gateway bug that inflated order values 100x, and the duplicate partner-outlet orders in the export.
6. **Review in 4 weeks:** returns caught, rupees saved, cancellation rate on called orders.
