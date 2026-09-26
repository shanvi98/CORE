"""Shared configuration and constants for the Kestrel returns-risk model.

Every path, cost parameter and fixed category list lives here so that
train.py and app.py (built later) import the exact same values instead of
duplicating them.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test_unlabelled.csv"
CUSTOMERS_CSV = DATA_DIR / "customers.csv"
PRODUCTS_CSV = DATA_DIR / "products.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"

MODEL_PATH = ARTIFACTS_DIR / "model.txt"
CALIBRATOR_PATH = ARTIFACTS_DIR / "calibrator.pkl"
META_PATH = ARTIFACTS_DIR / "model_meta.json"

PREDICTIONS_CSV = ROOT / "predictions.csv"

SEED = 42

# --------------------------------------------------------------------------
# Columns that are only known AFTER a return has already been raised.
# These must never be used as model inputs — see BLUEPRINT.md F1/F2/F3.
# --------------------------------------------------------------------------
LEAKY_COLUMNS = ["pickup_scheduled_at", "last_service_event_type", "source"]

TARGET_COL = "returned"
ID_COL = "order_id"

# --------------------------------------------------------------------------
# Cost parameters used for the call-vs-hold economics (ops-policy.pdf §4/§7,
# Farhan's and Ritu's emails). See BLUEPRINT.md §1.4 for the derivation.
# --------------------------------------------------------------------------
COST_RETURN = 1150.0          # Rs, all-in cost of a processed return (Farhan's figure)
COST_RETURN_RITU = 600.0      # Rs, Ritu's out-of-date figure — kept for comparison only
COST_CALL = 45.0              # Rs, per completed pre-dispatch confirmation call
CALL_EFFECTIVENESS = 0.35     # share of returns a call prevents (spring pilot)
HOLD_CANCEL_RATE = 0.12       # share of held (>24h) orders the customer cancels
MARGIN_LOST_ON_CANCEL = 800.0  # Rs, PLACEHOLDER — unknown, ask Farhan (see memo)
MARGIN_SENSITIVITY = [0.0, 375.0, 800.0, 1500.0]

# Theoretical break-even threshold: COST_CALL / (CALL_EFFECTIVENESS * COST_RETURN)
THRESHOLD_THEORETICAL = COST_CALL / (CALL_EFFECTIVENESS * COST_RETURN)
THRESHOLD_RITU = COST_CALL / (CALL_EFFECTIVENESS * COST_RETURN_RITU)
# If the empirical (net-maximising) threshold found on the holdout differs
# from the theoretical one by more than this, prefer the empirical one and
# warn — it means the calibration is off.
THRESHOLD_DISAGREEMENT_TOLERANCE = 0.03

RISK_BAND_HIGH = 0.30  # score >= this => "high" risk band (else "medium" if >= threshold)

# --------------------------------------------------------------------------
# Fixed category lists — training and serving must use identical codes, so
# these are the single source of truth. model_meta.json persists them too.
# --------------------------------------------------------------------------
SALES_CHANNELS = ["app", "web", "marketplace", "partner_outlet"]
PAYMENT_MODES = ["prepaid_upi", "prepaid_card", "cod", "emi"]
IS_GIFT_VALUES = ["N", "Y"]

SKUS = [
    "KH-AF-01", "KH-AF-02", "KH-AF-03",
    "KH-MG-01", "KH-MG-02", "KH-MG-03",
    "KH-WP-01", "KH-WP-02", "KH-WP-03",
    "KH-RV-01", "KH-RV-02", "KH-RV-03",
    "KH-IC-01", "KH-IC-02", "KH-IC-03",
    "KH-CF-01", "KH-CF-02", "KH-CF-03",
    "KH-RH-01", "KH-RH-02", "KH-RH-03",
]

FAMILIES = [
    "Air Fryer", "Mixer Grinder", "Water Purifier", "Robot Vacuum",
    "Induction Cooktop", "Ceiling Fan", "Room Heater",
]

TIERS = ["Lite", "Pro", "Max"]

NOTE_TYPES = [
    "none", "landmark", "neighbour", "gate_code", "timing", "fragile",
    "security", "office", "floor", "call", "whatsapp", "other",
]

# Indian states appearing in customers.csv are not enumerated here (too many
# and not fixed by the assignment); features.py derives the category list
# from the data at train time and persists it in model_meta.json.

# --------------------------------------------------------------------------
# Data quality fix windows (BLUEPRINT.md F4): Oct-2025 order values were
# recorded through a new payment gateway that stored paise, not rupees.
# --------------------------------------------------------------------------
CURRENCY_BUG_START = "2025-10-01"
CURRENCY_BUG_END = "2025-11-01"  # exclusive
CURRENCY_BUG_FACTOR = 100.0

MODEL_VERSION_PREFIX = "lgbm"
