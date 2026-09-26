"""Kestrel Home — Returns Risk: FastAPI service.

One scoring endpoint. Starts and serves on a clean machine with no paid API
key. If the model artifacts are missing, the app still starts — /predict
answers 503 instead of crashing. See BLUEPRINT.md §2.2 and README.md.

Run:  uvicorn app:app --port 8000
"""
from __future__ import annotations

import json
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from src import config
from src.features import build_features, clean
from src.reasons import explain

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("app")

STATIC_INDEX = config.ROOT / "static" / "index.html"
LLM_TIMEOUT_SECONDS = 3.0


# --------------------------------------------------------------------------
# App state — set once at startup, never raises even when things are missing.
# --------------------------------------------------------------------------
class AppState:
    def __init__(self) -> None:
        self.booster: Optional[lgb.Booster] = None
        self.calibrator = None
        self.meta: dict[str, Any] = {}
        self.load_error: Optional[str] = None
        self.customers: dict[str, dict] = {}
        self.products: dict[str, dict] = {}
        self.customers_loaded = False
        self.products_loaded = False
        self.llm_enabled = False


state = AppState()


def _load_model() -> None:
    try:
        if not config.MODEL_PATH.exists() or not config.META_PATH.exists():
            raise FileNotFoundError(
                f"Model artifacts not found at {config.ARTIFACTS_DIR}. Run `python train.py` first."
            )
        state.booster = lgb.Booster(model_file=str(config.MODEL_PATH))
        state.meta = json.loads(config.META_PATH.read_text())
        state.calibrator = joblib.load(config.CALIBRATOR_PATH)
        state.load_error = None
        log.info(f"Loaded model {state.meta.get('model_version')} from {config.ARTIFACTS_DIR}")
    except Exception as exc:  # noqa: BLE001 — deliberately broad: never let a bad artifact crash the app
        state.booster = None
        state.calibrator = None
        state.meta = {}
        state.load_error = str(exc)
        log.warning(f"Model not loaded: {exc}")


def _load_reference_data() -> None:
    try:
        if config.CUSTOMERS_CSV.exists():
            df = pd.read_csv(config.CUSTOMERS_CSV)
            state.customers = df.set_index("customer_id").to_dict(orient="index")
            state.customers_loaded = True
            log.info(f"Loaded {len(state.customers)} customers from {config.CUSTOMERS_CSV}")
        else:
            log.info(f"{config.CUSTOMERS_CSV} not found; unknown-customer defaults will be used for every order.")
    except Exception as exc:  # noqa: BLE001
        log.warning(f"Could not load customers.csv: {exc}")

    try:
        if config.PRODUCTS_CSV.exists():
            df = pd.read_csv(config.PRODUCTS_CSV)
            state.products = df.set_index("sku").to_dict(orient="index")
            state.products_loaded = True
            log.info(f"Loaded {len(state.products)} products from {config.PRODUCTS_CSV}")
        else:
            log.info(f"{config.PRODUCTS_CSV} not found; product details will fall back to defaults.")
    except Exception as exc:  # noqa: BLE001
        log.warning(f"Could not load products.csv: {exc}")


def _check_llm() -> None:
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    use_llm = os.environ.get("USE_LLM", "0") == "1"
    state.llm_enabled = has_key and use_llm
    if not has_key:
        log.info("ANTHROPIC_API_KEY not set — reasons will use plain-English templates only.")
    elif not use_llm:
        log.info("ANTHROPIC_API_KEY set but USE_LLM!=1 — reasons will use templates only.")
    else:
        log.info("LLM rephrasing enabled (optional; falls back to templates on any error).")


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_model()
    _load_reference_data()
    _check_llm()
    yield


app = FastAPI(title="Kestrel Returns Risk", lifespan=lifespan)

# Local-dev convenience only: lets static/index.html be opened via file://
# and still call the API. Not a production security posture.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Request / response schemas
# --------------------------------------------------------------------------
class OrderIn(BaseModel):
    model_config = {"extra": "allow"}

    order_id: str
    order_placed_at: str
    customer_id: str
    sku: str
    sales_channel: Literal["app", "web", "marketplace", "partner_outlet"]
    payment_mode: Literal["prepaid_upi", "prepaid_card", "cod", "emi"]
    discount_pct: float = Field(ge=0, le=100)
    qty: int = Field(ge=1)
    order_value_inr: float = Field(gt=0)
    promised_delivery_days: int = Field(ge=1, le=30)
    delivery_pincode: str
    is_gift: Literal["Y", "N"]
    customer_prior_orders: int = Field(ge=0)
    customer_prior_returns: int = Field(ge=0)
    delivery_note: Optional[str] = Field(default=None, max_length=300)

    @field_validator("order_placed_at")
    @classmethod
    def _valid_datetime(cls, v: str) -> str:
        parsed = pd.to_datetime(v, errors="coerce")
        if pd.isna(parsed):
            raise ValueError('must look like "YYYY-MM-DD HH:MM" (or ISO 8601)')
        return v

    @field_validator("delivery_pincode")
    @classmethod
    def _valid_pincode(cls, v: str) -> str:
        if not re.fullmatch(r"\d{6}", v):
            raise ValueError("must be exactly 6 digits (use 000000 if no address was captured)")
        return v

    @field_validator("sku")
    @classmethod
    def _valid_sku(cls, v: str) -> str:
        known = set(state.meta.get("categories", {}).get("sku", config.SKUS)) or set(config.SKUS)
        if v not in known:
            raise ValueError(f"unknown sku; must be one of {sorted(known)}")
        return v

    @model_validator(mode="after")
    def _prior_returns_within_orders(self) -> "OrderIn":
        if self.customer_prior_returns > self.customer_prior_orders:
            raise ValueError("customer_prior_returns cannot exceed customer_prior_orders")
        return self


class Reason(BaseModel):
    rank: int
    factor: str
    direction: Literal["increases_risk", "decreases_risk"]
    impact: float
    text: str


class PredictOut(BaseModel):
    order_id: str
    score: float
    risk_band: Literal["low", "medium", "high"]
    recommended_action: Literal["CALL_BEFORE_DISPATCH", "DISPATCH"]
    threshold: float
    expected_saving_inr: float
    reasons: list[Reason]
    note: Optional[str]
    warnings: list[str]
    explanation_source: Literal["template", "llm"]
    model_version: Optional[str]


# --------------------------------------------------------------------------
# Error handlers — graceful, never a stack trace to the client
# --------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    fields = []
    for err in exc.errors():
        loc = [str(p) for p in err.get("loc", []) if p != "body"]
        fields.append({"field": ".".join(loc) or "body", "message": err.get("msg", "invalid value")})
    return JSONResponse(status_code=422, content={"error": "invalid_input", "fields": fields})


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled error while serving a request")
    return JSONResponse(
        status_code=500,
        content={"error": "internal", "message": "Something went wrong scoring this order."},
    )


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/")
async def root():
    if STATIC_INDEX.exists():
        return FileResponse(STATIC_INDEX)
    return HTMLResponse(
        "<h1>Kestrel Returns Risk</h1><p>The UI file (static/index.html) hasn't been "
        "built yet. The API itself is up — see <a href='/health'>/health</a> and "
        "POST /predict.</p>",
        status_code=200,
    )


@app.get("/health")
async def health():
    if state.booster is not None:
        return {
            "status": "ok",
            "model_loaded": True,
            "llm_enabled": state.llm_enabled,
            "model_version": state.meta.get("model_version"),
            "message": "Ready.",
        }
    return {
        "status": "degraded",
        "model_loaded": False,
        "llm_enabled": state.llm_enabled,
        "model_version": None,
        "message": state.load_error or "Risk model not loaded. Run `python train.py` then restart.",
    }


def _order_to_frame(order: OrderIn) -> pd.DataFrame:
    row = {
        "order_id": order.order_id,
        "order_placed_at": order.order_placed_at,
        "customer_id": order.customer_id,
        "sku": order.sku,
        "sales_channel": order.sales_channel,
        "payment_mode": order.payment_mode,
        "discount_pct": order.discount_pct,
        "qty": order.qty,
        "order_value_inr": order.order_value_inr,
        "promised_delivery_days": order.promised_delivery_days,
        "delivery_pincode": order.delivery_pincode,
        "is_gift": order.is_gift,
        "customer_prior_orders": order.customer_prior_orders,
        "customer_prior_returns": order.customer_prior_returns,
        "delivery_note": order.delivery_note,
    }
    return pd.DataFrame([row])


def _lookup_customer(customer_id: str, warnings: list[str]) -> pd.DataFrame:
    rec = state.customers.get(customer_id)
    if rec is not None:
        return pd.DataFrame([{"customer_id": customer_id, **rec}])
    if state.customers_loaded:
        warnings.append(f"Customer {customer_id} not found; using defaults.")
    default_shield = state.meta.get("default_shield_member", "N")
    default_tenure_days = state.meta.get("default_tenure_days", 365)
    # Synthesize a signup_date that yields the median tenure at build_features time.
    signup = (pd.Timestamp.today() - pd.Timedelta(days=default_tenure_days)).date().isoformat()
    return pd.DataFrame([{
        "customer_id": customer_id, "city": None, "state": None,
        "signup_date": signup, "shield_member": default_shield,
    }])


def _lookup_product(sku: str, warnings: list[str]) -> pd.DataFrame:
    rec = state.products.get(sku)
    if rec is not None:
        return pd.DataFrame([{"sku": sku, **rec}])
    if state.products_loaded:
        warnings.append(f"Product {sku} not found in product reference data; using defaults.")
    else:
        warnings.append("Product reference data (data/products.csv) not available; using defaults.")
    return pd.DataFrame([{
        "sku": sku, "family": None, "model_name": None,
        "list_price_inr": None, "warranty_months": None, "launch_date": None,
    }])


def _reject_leaky_fields(order: OrderIn, warnings: list[str]) -> None:
    extra = getattr(order, "model_extra", None) or {}
    for key in extra:
        if key in config.LEAKY_COLUMNS:
            warnings.append(f"Ignored post-dispatch field: {key}")


def _explanation(reasons: list[dict], warnings: list[str]) -> tuple[list[dict], str]:
    """Optionally rephrase the 3 template sentences with an LLM. Falls back
    to the templates verbatim on any error, timeout, or missing dependency.
    The score and the choice/order of reasons are never touched here.
    """
    if not state.llm_enabled:
        return reasons, "template"
    try:
        import anthropic  # optional dependency — imported lazily, not in requirements.txt

        client = anthropic.Anthropic(timeout=LLM_TIMEOUT_SECONDS)
        model_name = os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001")
        payload = [{"factor": r["factor"], "direction": r["direction"], "text": r["text"]} for r in reasons]
        prompt = (
            "Rephrase each of these 3 risk-factor sentences for a warehouse employee, "
            "one short plain sentence each, same meaning and same order, no jargon. "
            "Return ONLY a JSON array of 3 strings, nothing else.\n\n"
            f"{json.dumps(payload)}"
        )
        resp = client.messages.create(
            model=model_name, max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text
        rephrased = json.loads(text)
        if isinstance(rephrased, list) and len(rephrased) == len(reasons):
            for r, new_text in zip(reasons, rephrased):
                r["text"] = str(new_text)
            return reasons, "llm"
        raise ValueError("unexpected LLM response shape")
    except Exception as exc:  # noqa: BLE001 — any failure falls back silently
        log.warning(f"LLM rephrase failed, falling back to templates: {exc}")
        warnings.append("llm_rephrase_failed")
        return reasons, "template"


@app.post("/predict", response_model=PredictOut)
async def predict(order: OrderIn):
    if state.booster is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "model_unavailable",
                "message": state.load_error or "The risk model isn't loaded. Run: python train.py",
                "hint": "See README → Train",
            },
        )

    warnings: list[str] = []
    _reject_leaky_fields(order, warnings)

    orders_df = clean(_order_to_frame(order))
    customers_df = _lookup_customer(order.customer_id, warnings)
    products_df = _lookup_product(order.sku, warnings)

    categories = state.meta.get("categories", {})
    X = build_features(orders_df, customers_df, products_df, categories=categories)
    X = X[state.meta["feature_names"]]

    raw_score = float(state.booster.predict(X)[0])
    score = float(state.calibrator.predict([raw_score])[0])
    score = min(max(score, 0.0), 1.0)

    threshold = float(state.meta["threshold"])
    risk_band_high = float(state.meta.get("risk_band_high", 0.30))
    if score >= risk_band_high:
        risk_band = "high"
    elif score >= threshold:
        risk_band = "medium"
    else:
        risk_band = "low"

    cost = state.meta["cost_params"]
    expected_saving = max(0.0, cost["call_effectiveness"] * cost["cost_return"] * score - cost["cost_call"])
    action = "CALL_BEFORE_DISPATCH" if score >= threshold else "DISPATCH"

    contrib = state.booster.predict(X, pred_contrib=True)[0]
    feature_row = X.iloc[0].to_dict()
    reasons = explain(contrib, feature_row, state.meta)
    reasons, explanation_source = _explanation(reasons, warnings)

    note = None
    if bool(feature_row.get("shield_member")):
        note = "Shield member: call, never hold."

    return PredictOut(
        order_id=order.order_id,
        score=round(score, 4),
        risk_band=risk_band,
        recommended_action=action,
        threshold=round(threshold, 4),
        expected_saving_inr=round(expected_saving, 1),
        reasons=[Reason(**r) for r in reasons],
        note=note,
        warnings=warnings,
        explanation_source=explanation_source,
        model_version=state.meta.get("model_version"),
    )
