# crisk_limit_strategy.py — turn decisions into approve limits + portfolio view

import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("Dataset")
READOUT = BASE / "crisk_readout"
CLEAN   = BASE / "crisk_clean"
OUT     = BASE / "crisk_policy"
OUT.mkdir(parents=True, exist_ok=True)

# Load decisions + scored cohort (for features like limit_bal, age, sex)
dec = pd.read_csv(READOUT / "decisions_1y.csv")           # decision + thresholds + score/pd_month/target
scored = pd.read_csv(READOUT / "scored_cohort_1y.csv")

# Join if we have a key; else align by row order
key = next((c for c in ["applicant_id","customer_id","id"] if c in dec.columns and c in scored.columns), None)
if key:
    df = dec.merge(scored, on=key, how="left", suffixes=("","_sc"))
else:
    df = dec.copy()
    for c in scored.columns:
        if c not in df.columns:
            df[c] = scored[c].values[:len(df)]

# Choose score column
score_col = "pd_month" if "pd_month" in df.columns else ("score" if "score" in df.columns else None)
if score_col is None:
    raise RuntimeError("Need a score column (pd_month or score).")

# Ensure 1y PD
if "pd_1y" not in df.columns:
    pd_m = pd.to_numeric(df[score_col], errors="coerce").clip(0, 1).fillna(0.0)
    df["pd_1y"] = 1.0 - (1.0 - pd_m) ** 12
df["pd_1y"] = df["pd_1y"].clip(0, 1)

# Base limit if missing
if "limit_bal" not in df.columns:
    df["limit_bal"] = 50000

# Risk bands by PD
bins = [-np.inf, 0.02, 0.05, 0.10, np.inf]
labels = ["A (<=2%)","B (2–5%)","C (5–10%)","D (>10%)"]
df["risk_band"] = pd.cut(df["pd_1y"], bins=bins, labels=labels)

# EL-capped limit policy
LGD = 0.85
UTIL = 0.40
EL_BUDGET = 0.03

mult_by_band = {
    "A (<=2%)": 1.20,
    "B (2–5%)": 1.00,
    "C (5–10%)": 0.80,
    "D (>10%)": 0.50,
}

# FIX: make multiplier a float series (not categorical) before fillna
mult = df["risk_band"].astype(str).map(mult_by_band)      # object floats or NaN
mult = pd.to_numeric(mult, errors="coerce").fillna(0.7)   # <- safe fill

df["baseline_limit"] = df["limit_bal"] * mult

# EL cap
cap = (EL_BUDGET / (df["pd_1y"] * LGD * UTIL)).replace([np.inf, -np.inf], np.nan)
df["el_cap_multiplier"] = cap.clip(upper=1.0)
df["limit_rec"] = (df["baseline_limit"] * df["el_cap_multiplier"]).fillna(df["baseline_limit"])

# Practical guards
df["limit_rec"] = df["limit_rec"].clip(lower=1000, upper=200000)

# Apply decisions: declines -> 0 limit
df["limit_final"] = np.where(df["decision"].astype(str).str.lower().eq("decline"), 0, df["limit_rec"])

# Portfolio KPIs
approved = df[df["limit_final"] > 0].copy()
n_apps = len(df); n_approved = len(approved)
approval_rate = n_approved / max(1, n_apps)

approved["EL"] = approved["pd_1y"] * LGD * UTIL * approved["limit_final"]
portfolio = {
    "n_applications": int(n_apps),
    "n_approved": int(n_approved),
    "approval_rate": float(approval_rate),
    "avg_limit": float(approved["limit_final"].mean() if n_approved else 0),
    "sum_limit": float(approved["limit_final"].sum() if n_approved else 0),
    "expected_loss_total": float(approved["EL"].sum() if n_approved else 0),
    "expected_loss_rate": float(approved["EL"].sum() / max(1.0, approved["limit_final"].sum())) if n_approved else 0.0,
}

# Rollups
by_band = (approved.groupby("risk_band", observed=True)
           .agg(apps=("pd_1y","size"),
                avg_pd=("pd_1y","mean"),
                avg_limit=("limit_final","mean"),
                sum_limit=("limit_final","sum"),
                exp_loss=("EL","sum"))
           .reset_index())

# Save
OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / "decisions_with_limits.csv", index=False)
by_band.to_csv(OUT / "portfolio_by_risk_band.csv", index=False)
pd.Series(portfolio).to_json(OUT / "portfolio_summary.json", indent=2)

print("✓ Limits assigned and policy pack saved to:", OUT.resolve())
print("Portfolio:", portfolio)
print(by_band)
