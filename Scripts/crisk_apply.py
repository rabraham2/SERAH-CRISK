# crisk_apply.py — score new applications, make approve/review/decline + limits

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# ---------- paths ----------
BASE   = Path("Dataset")
CLEAN  = BASE / "crisk_clean"
MODELS = BASE / "crisk_models"
POLICY = BASE / "crisk_policy"
READOUT= BASE / "crisk_readout"
POLICY.mkdir(parents=True, exist_ok=True)

MODEL_PATH   = MODELS / "crisk_model.pkl"
THR_PATH     = POLICY / "final_policy_threshold.json"   # optional (from your frontier/sweep)
PORTF_PATH   = POLICY / "final_policy_portfolio.json"   # optional (contains approval_rate)
DEFAULT_DEMO = BASE / "new_apps_demo.csv"               # created on first run if needed

# ---------- helpers ----------
REQ_COLS = (
    [f"pay_{i}" for i in [1,2,3,4,5,6]] +
    [f"bill_amt{i}" for i in [1,2,3,4,5,6]] +
    [f"pay_amt{i}"  for i in [1,2,3,4,5,6]] +
    ["sex","education","marriage","age","limit_bal"]
)

def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model not found at: {path}")
    return joblib.load(path)

def read_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suf in (".xlsx", ".xls"):
        # header=1 for UCI original excel; for generic xlsx keep header=0
        try:
            return pd.read_excel(path, engine="openpyxl")
        except Exception:
            return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suf}")

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing training columns with safe defaults."""
    out = df.copy()
    defaults = {
        # categorical-ish / ordinal
        "sex": 2,          # 1/2 in UCI; using 2 as default
        "education": 2,    # 1..4 common; 2 ~ university
        "marriage": 2,     # 1..3 common; 2 ~ single
        # numeric
        "age": 35,
        "limit_bal": 50000,
        **{f"pay_{i}": 0 for i in [1,2,3,4,5,6]},
        **{f"bill_amt{i}": 0.0 for i in [1,2,3,4,5,6]},
        **{f"pay_amt{i}":  0.0 for i in [1,2,3,4,5,6]},
    }
    for c in REQ_COLS:
        if c not in out.columns:
            out[c] = defaults[c]
    # order columns roughly like training (not strictly required, but neat)
    return out[[*(c for c in REQ_COLS), *(c for c in out.columns if c not in REQ_COLS)]]

def load_policy_threshold() -> dict:
    """Try to load a saved threshold. Return dict with keys:
       - 'thr_pd_m' (monthly PD threshold for approval) OR None
       - 'approval_rate' (float 0..1) OR None
       - 'review_band' (float, width around threshold) default 0.10
    """
    out = {"thr_pd_m": None, "approval_rate": None, "review_band": 0.10}
    try:
        if THR_PATH.exists():
            with open(THR_PATH, "r") as f:
                d = json.load(f)
            # accept several key names
            out["thr_pd_m"] = (
                d.get("thr_pd_m") or d.get("thr_approve") or d.get("threshold") or d.get("threshold_approve")
            )
        if out["approval_rate"] is None and PORTF_PATH.exists():
            with open(PORTF_PATH, "r") as f:
                p = json.load(f)
            out["approval_rate"] = p.get("approval_rate")
    except Exception:
        pass
    return out

def make_demo_if_needed() -> Path:
    if DEFAULT_DEMO.exists():
        return DEFAULT_DEMO
    src = CLEAN / "test.parquet"
    if not src.exists():
        raise FileNotFoundError(f"Cannot create demo — missing {src}")
    df = pd.read_parquet(src)
    df = df.drop(columns=["target"], errors="ignore")
    df.head(1000).to_csv(DEFAULT_DEMO, index=False)
    print(f"(No --input provided) Created demo at: {DEFAULT_DEMO}")
    return DEFAULT_DEMO

# ---- limit/portfolio helpers (same logic as your limit strategy) ----
LGD, UTIL = 0.85, 0.40
EL_BUDGET = 0.03  # EL as share of limit tolerated for EL-cap
MULT_BY_BAND = {
    "A (<=2%)": 1.20,
    "B (2–5%)": 1.00,
    "C (5–10%)": 0.80,
    "D (>10%)": 0.50,
}
BINS   = [-np.inf, 0.02, 0.05, 0.10, np.inf]
LABELS = ["A (<=2%)","B (2–5%)","C (5–10%)","D (>10%)"]

def assign_limits(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # 1y PD from monthly PD
    pd_m = pd.to_numeric(out["pd_month"], errors="coerce").clip(0,1).fillna(0.0)
    out["pd_1y"] = 1.0 - (1.0 - pd_m)**12
    out["pd_1y"] = out["pd_1y"].clip(0,1)
    # bands
    out["risk_band"] = pd.cut(out["pd_1y"], bins=BINS, labels=LABELS)
    mult = pd.to_numeric(out["risk_band"].astype(str).map(MULT_BY_BAND), errors="coerce").fillna(0.7)
    # baseline*mult
    if "limit_bal" not in out.columns:
        out["limit_bal"] = 50_000
    out["baseline_limit"] = out["limit_bal"] * mult
    # EL cap
    cap_mult = (EL_BUDGET / (out["pd_1y"] * LGD * UTIL)).replace([np.inf,-np.inf], np.nan).clip(upper=1.0)
    out["el_cap_multiplier"] = cap_mult
    out["limit_rec"] = (out["baseline_limit"] * out["el_cap_multiplier"]).fillna(out["baseline_limit"])
    out["limit_rec"] = out["limit_rec"].clip(lower=1_000, upper=200_000)
    # final: 0 for Decline, keep for Approve, 0 (or keep) for Review — here we set 0 to force manual review
    out["limit_final"] = np.where(out["decision"].str.lower().eq("decline"), 0,
                           np.where(out["decision"].str.lower().eq("review"), 0, out["limit_rec"]))
    return out

# ---------- main ----------
def main(input_path: Path | None = None):
    # 0) resolve input
    if input_path is None:
        input_path = make_demo_if_needed()
    df_raw = read_any(input_path)
    df = ensure_columns(df_raw)

    # 1) load model & score
    pipe = load_model(MODEL_PATH)
    pd_month = pipe.predict_proba(df[REQ_COLS])[:, 1]  # probability of default (= monthly PD proxy)
    scored = df.copy()
    scored["pd_month"] = pd_month

    # 2) load policy threshold (if you saved one); else derive from approval rate; else default to 5%
    pol = load_policy_threshold()
    scores = scored["pd_month"]
    thr = pol["thr_pd_m"]
    approval_rate = pol["approval_rate"]

    if thr is None:
        if approval_rate is None:
            approval_rate = 0.05  # fallback to your Option A outcome
        # lowest PD are safest => approve the lowest 'approval_rate' portion
        thr = float(scores.quantile(approval_rate))  # approve scores <= thr
    # define a review band ~10% of population above the approval threshold (cap within [0,1])
    band = float(pol.get("review_band", 0.10))
    q_low  = thr
    q_high = float(scores.quantile(min(approval_rate + band, 0.99)))

    # 3) make decisions
    decision = np.where(scores <= q_low, "Approve",
                 np.where(scores <= q_high, "Review", "Decline"))
    decided = scored.assign(decision=decision, thr_approve=thr, review_upper=q_high)

    # 4) limits for approvals (and zero for declines & reviews)
    decided = assign_limits(decided)

    # 5) outputs
    out_csv = POLICY / "new_batch_decisions.csv"
    out_json= POLICY / "new_batch_summary.json"
    decided.to_csv(out_csv, index=False)

    summary = {
        "n_applications": int(len(decided)),
        "approval_rate": float((decided["decision"] == "Approve").mean()),
        "review_rate":   float((decided["decision"] == "Review").mean()),
        "decline_rate":  float((decided["decision"] == "Decline").mean()),
        "thr_pd_month":  float(thr),
        "review_upper":  float(q_high),
        "avg_limit_approved": float(decided.loc[decided["decision"]=="Approve","limit_final"].mean()),
        "sum_limit_approved": float(decided.loc[decided["decision"]=="Approve","limit_final"].sum()),
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("✓ Scored and decided.")
    print("  Decisions ->", out_csv)
    print("  Summary   ->", out_json)
    print(summary)

# ... keep all your existing imports, helpers, and main() exactly as you have ...

if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser(
        description="CRISK: apply model & policy to new applications",
        add_help=True
    )
    ap.add_argument(
        "--input",
        help="Path to CSV/Parquet/XLSX with new applications (if omitted, a demo is created)."
    )
    # <-- This is the key line: ignore any extra args that PyCharm/pydevconsole adds
    args, unknown = ap.parse_known_args()
    if unknown:
        # Optional: print once so you know they're being ignored
        print(f"(Note) Ignoring unknown args: {unknown}")
    in_path = Path(args.input) if args.input else None
    # If you launched from the Python Console, just call main() with no args or your path
    main(in_path)
