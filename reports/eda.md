# EDA — Kestrel Home Returns Risk

- Train rows before de-dup: **11155**
- Train rows after de-dup (F3, partner-feed re-imports): **10504**
- Overall return rate: **0.1142**

## Return rate by month
```
                  mean  size
order_placed_at             
2025-04          0.150   661
2025-05          0.120   674
2025-06          0.094   678
2025-07          0.118   736
2025-08          0.129   714
2025-09          0.085   707
2025-10          0.116   700
2025-11          0.106   698
2025-12          0.102   684
2026-01          0.113   719
2026-02          0.128   664
2026-03          0.109   743
2026-04          0.121   705
2026-05          0.115   722
2026-06          0.110   699
```

## Return rate by payment_mode
```
               mean  size
payment_mode             
cod           0.188  3152
emi           0.091  1278
prepaid_card  0.088  2153
prepaid_upi   0.077  3921
```

## Return rate by sales_channel
```
                 mean  size
sales_channel              
app             0.107  3582
marketplace     0.136  2869
partner_outlet  0.091  1287
web             0.112  2766
```

## Return rate by is_gift
```
          mean  size
is_gift             
N        0.110  9778
Y        0.165   726
```

## Return rate by promised-delivery bucket
```
                         mean  size
promised_delivery_days             
<=3                     0.076  2300
4-7                     0.115  7107
>=8                     0.187  1097
```

## Return rate by Shield membership
```
                mean  size
shield_member             
N              0.094  8177
Y              0.186  2327
```

## Return rate by product family
```
                    mean  size
family                        
Air Fryer          0.120  1516
Ceiling Fan        0.067  1531
Induction Cooktop  0.083  1476
Mixer Grinder      0.068  1451
Robot Vacuum       0.195  1554
Room Heater        0.117  1525
Water Purifier     0.148  1451
```

## Evidence: known data problems

**F1/F2 — leakage.** `pickup_scheduled_at` is written only after a return is approved; `last_service_event_type` is pulled as of export day.

```
                           mean  size
pickup_scheduled_at                  
not set                  0.0075  9273
pickup_scheduled_at set  0.9180  1231
```

```
                           mean  size
last_service_event_type              
DEMO_DONE                0.0000  1417
INSTALL_DONE             0.0000  1354
NONE                     0.0425  6536
REVERSE_PICKUP           1.0000   750
TECH_VISIT               0.3848   447
```

**F3 — duplicates.** 651 rows were exact partner-feed re-imports of an already-seen `order_id`, dropped before any split.

**F4 — currency bug.** Oct-2025 `order_value_inr` was 100x too high (new payment gateway stored paise). Fixed in `clean()`; see `assert_currency_fixed` for the check.

**F5 — default pincode.** 848 rows use the system default pincode (walk-in, no address).
```
                               mean  size
delivery_pincode                         
has address                  0.1133  9656
pincode 000000 (no address)  0.1250   848
```

**F6 — customer snapshot.** 1999 orders have a signup date after the order date (customers.csv is a current snapshot, not point-in-time). Return rate is unaffected either way ({False: 0.115, True: 0.113}).
