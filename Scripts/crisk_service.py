# crisk_service.py — minimal REST API for CRISK (Pydantic v2 / FastAPI)
from pathlib import Path
from typing import List, Optional
import json

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# --- paths (adjust if your repo layout is different)
BASE = Path("Dataset")
MODEL_PATH   = BASE / "crisk_models" / "crisk_model.pkl"
POLICY_PATH  = BASE / "crisk_policy" / "final_policy_threshold.json"   # created earlier
READOUT_PATH = BASE / "crisk_readout"

# --- load model + policy
pipe = joblib.load(MODEL_PATH)
try:
    policy = json.loads(POLICY_PATH.read_text())
    THR = float(policy.get("thr_pd_month", 0.05))          # monthly PD threshold chosen
    REVIEW_UP = float(policy.get("review_upper", THR*1.45))  # optional review window
except Exception:
    THR, REVIEW_UP = 0.05, 0.07

LGD, UTIL = 0.85, 0.40
BAND_MULT = {"A (<=2%)": 1.20, "B (2–5%)": 1.00, "C (5–10%)": 0.80, "D (>10%)": 0.50}

# --- request/response schemas
FEATURE_ORDER = [
    "limit_bal","sex","education","marriage","age",
    "pay_1","pay_2","pay_3","pay_4","pay_5","pay_6",
    "bill_amt1","bill_amt2","bill_amt3","bill_amt4","bill_amt5","bill_amt6",
    "pay_amt1","pay_amt2","pay_amt3","pay_amt4","pay_amt5","pay_amt6",
]

class Applicant(BaseModel):
    # keep names consistent with your training columns
    limit_bal: float
    sex: int
    education: int
    marriage: int
    age: int
    pay_1: int
    pay_2: int
    pay_3: int
    pay_4: int
    pay_5: int
    pay_6: int
    bill_amt1: float
    bill_amt2: float
    bill_amt3: float
    bill_amt4: float
    bill_amt5: float
    bill_amt6: float
    pay_amt1: float
    pay_amt2: float
    pay_amt3: float
    pay_amt4: float
    pay_amt5: float
    pay_amt6: float
    applicant_id: Optional[str] = None

class ScoreRequest(BaseModel):
    # Pydantic v2: use Field(min_length=1) instead of conlist(..., min_items=1)
    applicants: List[Applicant] = Field(min_length=1)

class ScoreResponse(BaseModel):
    applicant_id: Optional[str] = None
    pd_month: float
    pd_1y: float
    decision: str
    limit: float

app = FastAPI(title="CRISK API", version="1.0")

def _risk_band(pd_1y: float) -> str:
    if pd_1y <= 0.02: return "A (<=2%)"
    if pd_1y <= 0.05: return "B (2–5%)"
    if pd_1y <= 0.10: return "C (5–10%)"
    return "D (>10%)"

def _limit_with_policy(row: pd.Series) -> float:
    # baseline by band
    band = _risk_band(row["pd_1y"])
    baseline = row["limit_bal"] * BAND_MULT.get(band, 0.7)
    # EL cap
    cap_mult = (0.03 / (row["pd_1y"] * LGD * UTIL)) if row["pd_1y"] > 0 else 1.0
    cap_mult = min(max(cap_mult, 0.0), 1.0)
    limit = baseline * cap_mult
    return float(np.clip(limit, 1000, 200000))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists(), "threshold": THR}

@app.post("/score", response_model=List[ScoreResponse])
def score(req: ScoreRequest):
    # to DataFrame in the exact feature order
    df = pd.DataFrame([a.model_dump() for a in req.applicants])
    X = df[FEATURE_ORDER].copy()

    # monthly PD from model
    pd_month = pipe.predict_proba(X)[:, 1]
    pd_1y = 1 - (1 - pd_month) ** 12

    # decisions
    decisions = np.where(pd_month <= THR, "Approve",
                 np.where(pd_month <= REVIEW_UP, "Review", "Decline"))

    # limits (0 for Decline)
    out: List[ScoreResponse] = []
    for i, row in df.iterrows():
        pd1y = float(pd_1y[i])
        dec  = str(decisions[i])
        lim  = _limit_with_policy(pd.Series({**row.to_dict(), "pd_1y": pd1y})) if dec != "Decline" else 0.0
        out.append(ScoreResponse(
            applicant_id=row.get("applicant_id"),
            pd_month=float(pd_month[i]),
            pd_1y=pd1y,
            decision=dec,
            limit=lim
        ))
    return out
