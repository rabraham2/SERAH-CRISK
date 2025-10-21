# crisk_policy_sweep.py — search approval %, bad-rate cap & get best portfolio under EL cap

import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("Dataset")
READOUT = BASE / "crisk_readout"
OUT     = BASE / "crisk_policy"
OUT.mkdir(parents=True, exist_ok=True)

# Knobs / constraints
APPROVAL_GRID = np.arange(0.35, 0.71, 0.05)     # target approval rates to try
BAD_CAPS      = [0.06, 0.08, 0.10, 0.12]        # max bad-rate among approved
EL_CAP_PORT   = 0.10                             # portfolio expected-loss rate ceiling (10%)
LGD, UTIL     = 0.85, 0.40

# limit multipliers by risk band
MULT_BY_BAND = {
    "A (<=2%)": 1.20,
    "B (2–5%)": 1.00,
    "C (5–10%)": 0.80,
    "D (>10%)": 0.50,
}

# Load scored cohort
df = pd.read_csv(READOUT / "scored_cohort_1y.csv")   # has pd_month + target
if "pd_month" not in df.columns:
    raise RuntimeError("Need 'pd_month' in scored_cohort_1y.csv")
if "target" not in df.columns:
    raise RuntimeError("Need 'target' (0/1) in scored_cohort_1y.csv")

if "limit_bal" not in df.columns:
    df["limit_bal"] = 50_000

pd_m = pd.to_numeric(df["pd_month"], errors="coerce").clip(0, 1).fillna(0.0)
df["pd_1y"] = 1.0 - (1.0 - pd_m) ** 12
df["pd_1y"] = df["pd_1y"].clip(0, 1)

bins = [-np.inf, 0.02, 0.05, 0.10, np.inf]
labels = ["A (<=2%)", "B (2–5%)", "C (5–10%)", "D (>10%)"]
df["risk_band"] = pd.cut(df["pd_1y"], bins=bins, labels=labels)

# helper to compute portfolio metrics for an approval selection (boolean ndarray)
def portfolio_metrics(selection_mask: np.ndarray) -> dict:
    approved = df.loc[selection_mask].copy()
    if approved.empty:
        return dict(
            approval_rate=0.0, approved_bad_rate=0.0,
            avg_limit=0.0, sum_limit=0.0,
            expected_loss_total=0.0, expected_loss_rate=0.0
        )

    mult = pd.to_numeric(approved["risk_band"].astype(str).map(MULT_BY_BAND),
                         errors="coerce").fillna(0.7)
    baseline_limit = approved["limit_bal"] * mult
    cap_mult = (0.03 / (approved["pd_1y"] * LGD * UTIL)).replace([np.inf, -np.inf], np.nan).clip(upper=1.0)
    limit_rec = (baseline_limit * cap_mult).fillna(baseline_limit).clip(lower=1_000, upper=200_000)

    approved["limit_final"] = limit_rec
    approved["EL"] = approved["pd_1y"] * LGD * UTIL * approved["limit_final"]

    return dict(
        approval_rate=float(len(approved) / len(df)),
        approved_bad_rate=float(approved["target"].mean()),
        avg_limit=float(approved["limit_final"].mean()),
        sum_limit=float(approved["limit_final"].sum()),
        expected_loss_total=float(approved["EL"].sum()),
        expected_loss_rate=float(approved["EL"].sum() / max(1.0, approved["limit_final"].sum())),
    )

rows = []
pd_month_np = df["pd_month"].to_numpy()
target_np   = df["target"].to_numpy()

for bad_cap in BAD_CAPS:
    for a in APPROVAL_GRID:
        # approve lowest-risk (lowest pd_month) portion
        thr = float(np.quantile(pd_month_np, 1 - a))
        approve_mask = (pd_month_np <= thr)

        # tighten approval if bad-rate too high
        approved_bad = target_np[approve_mask].mean() if np.any(approve_mask) else 1.0
        target_a = float(a)
        while approved_bad > bad_cap and target_a > 0.20:
            target_a -= 0.01
            thr = float(np.quantile(pd_month_np, 1 - target_a))
            approve_mask = (pd_month_np <= thr)
            approved_bad = target_np[approve_mask].mean() if np.any(approve_mask) else 1.0

        metrics = portfolio_metrics(approve_mask)

        rows.append({
            "approval_target_req": float(a),
            "bad_cap_req": float(bad_cap),
            "approval_rate": metrics["approval_rate"],
            "approved_bad_rate": metrics["approved_bad_rate"],
            "expected_loss_rate": metrics["expected_loss_rate"],
            "sum_limit": metrics["sum_limit"],
            "avg_limit": metrics["avg_limit"],
        })

res = pd.DataFrame(rows)
res.to_csv(OUT / "policy_sweep.csv", index=False)

ok = res[res["expected_loss_rate"] <= EL_CAP_PORT].sort_values("sum_limit", ascending=False)
print(f"Top candidates meeting EL cap ≤ {EL_CAP_PORT:.0%}:")
if ok.empty:
    print("(none meet EL cap) — showing lowest expected_loss_rate instead:\n")
    print(res.sort_values(["expected_loss_rate", "sum_limit"]).head(10).to_string(index=False))
else:
    print(ok.head(10).to_string(index=False))

print(f"\nSaved sweep to: {(OUT / 'policy_sweep.csv').resolve()}")
