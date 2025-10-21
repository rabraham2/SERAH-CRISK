# crisk_new_batch_apply_limits_and_compare.py
# Apply limit policy to baseline & tuned decisions, then compare portfolios.

import json
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- paths ----------
BASE     = Path("Dataset")
POLICY   = BASE / "crisk_policy"
POLICY.mkdir(parents=True, exist_ok=True)

COHORT_CSV       = BASE / "new_apps_demo.csv"               # created earlier
DECISIONS_BASE   = POLICY / "new_batch_decisions.csv"       # 5/10/85
DECISIONS_TUNED  = POLICY / "new_batch_decisions_tuned.csv" # re-thresholded

# ---------- policy constants ----------
LGD   = 0.85
UTIL  = 0.40
EL_BUDGET = 0.03  # used for EL cap multiplier
MULT_BY_BAND = {
    "A (<=2%)": 1.20,
    "B (2–5%)": 1.00,
    "C (5–10%)": 0.80,
    "D (>10%)": 0.50,
}

# ---------- helpers ----------
def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def align_merge(cohort: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    key = next((c for c in ["row_id","applicant_id","customer_id","id"]
                if c in cohort.columns and c in decisions.columns), None)
    if key:
        return decisions.merge(cohort, on=key, how="left", suffixes=("","_coh"))

    n = min(len(cohort), len(decisions))
    df = decisions.iloc[:n].copy()
    for c in cohort.columns:
        if c not in df.columns:
            df[c] = cohort[c].values[:n]
    return df

def add_pd_1y(df: pd.DataFrame, score_col: str = "pd_month") -> pd.DataFrame:
    if score_col not in df.columns:
        raise RuntimeError(f"Need '{score_col}' in decisions/cohort.")
    pd_m = pd.to_numeric(df[score_col], errors="coerce").clip(0, 1).fillna(0.0)
    df["pd_1y"] = 1.0 - (1.0 - pd_m) ** 12
    return df

def add_risk_band(df: pd.DataFrame) -> pd.DataFrame:
    bins = [-np.inf, 0.02, 0.05, 0.10, np.inf]
    labels = ["A (<=2%)","B (2–5%)","C (5–10%)","D (>10%)"]
    df["risk_band"] = pd.cut(df["pd_1y"], bins=bins, labels=labels)
    return df

def apply_limit_policy(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    if "limit_bal" not in df.columns:
        df["limit_bal"] = 50_000

    mult = pd.to_numeric(df["risk_band"].astype(str).map(MULT_BY_BAND), errors="coerce").fillna(0.7)
    baseline = df["limit_bal"] * mult

    with np.errstate(divide="ignore", invalid="ignore"):
        cap_mult = (EL_BUDGET / (df["pd_1y"] * LGD * UTIL))
    cap_mult = pd.to_numeric(cap_mult, errors="coerce").replace([np.inf, -np.inf], np.nan).clip(upper=1.0)

    limit_rec = (baseline * cap_mult).fillna(baseline).clip(lower=1_000, upper=200_000)

    d = df["decision"].astype(str).str.lower()
    df["limit_final"] = np.where(d.eq("decline"), 0.0,
                          np.where(d.eq("approve"), limit_rec, 0.0))

    df["EL"] = df["pd_1y"] * LGD * UTIL * df["limit_final"]
    df["approved_flag"] = (df["limit_final"] > 0).astype(int)
    df["review_flag"]   = d.eq("review").astype(int)
    df["decline_flag"]  = d.eq("decline").astype(int)
    return df

def portfolio_summary(df: pd.DataFrame) -> dict:
    n = len(df)
    appr = int(df["approved_flag"].sum())
    rev  = int(df["review_flag"].sum())
    dec  = int(df["decline_flag"].sum())
    sum_lim = float(df.loc[df["approved_flag"] == 1, "limit_final"].sum())
    avg_lim = float(df.loc[df["approved_flag"] == 1, "limit_final"].mean() or 0.0)
    denom = max(1.0, sum_lim)
    el_rate = float(df["EL"].sum() / denom)

    return {
        "n_applications": int(n),
        "approval_rate": float(appr / n),
        "review_rate":   float(rev  / n),
        "decline_rate":  float(dec  / n),
        "avg_limit_approved": avg_lim,
        "sum_limit_approved": sum_lim,
        "expected_loss_rate": el_rate,
    }

def fairness_slices(df: pd.DataFrame) -> dict:
    out = {}
    if "sex" in df.columns:
        ser = df.groupby("sex", observed=True)["approved_flag"].mean()
        out["approval_rate_by_sex"] = {str(k): float(v) for k, v in ser.items()}

    if "age" in df.columns:
        bands = pd.cut(df["age"], bins=[18,25,35,45,55,65,200], right=False)
        ser = (
            pd.DataFrame({"band": bands, "approved": df["approved_flag"]})
            .groupby("band", observed=True)["approved"]
            .mean()
        )
        # Convert Interval keys -> strings for JSON
        out["approval_rate_by_age_band"] = {str(k): float(v) for k, v in ser.items()}
    return out

def run_one(label: str, cohort: pd.DataFrame, decisions_path: Path) -> dict:
    dec = safe_read_csv(decisions_path)
    df  = align_merge(cohort, dec)
    df  = add_pd_1y(df, "pd_month")
    df  = add_risk_band(df)
    df  = apply_limit_policy(df)

    out_csv = POLICY / f"limits_{label}.csv"
    df.to_csv(out_csv, index=False)

    summary = portfolio_summary(df)
    summary.update(fairness_slices(df))

    with open(POLICY / f"limits_{label}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary

# ---------- run ----------
cohort = safe_read_csv(COHORT_CSV)

baseline = run_one("baseline", cohort, DECISIONS_BASE)
tuned    = run_one("tuned",    cohort, DECISIONS_TUNED)

comparison = {
    "baseline": baseline,
    "tuned": tuned,
    "delta": {
        "approval_rate":       float(tuned["approval_rate"]      - baseline["approval_rate"]),
        "expected_loss_rate":  float(tuned["expected_loss_rate"] - baseline["expected_loss_rate"]),
        "sum_limit_approved":  float(tuned["sum_limit_approved"] - baseline["sum_limit_approved"]),
        "avg_limit_approved":  float(tuned["avg_limit_approved"] - baseline["avg_limit_approved"]),
    }
}

with open(POLICY / "new_batch_limits_compare.json", "w") as f:
    json.dump(comparison, f, indent=2)

print("✓ Wrote:")
print("  -", (POLICY / "limits_baseline.csv").resolve())
print("  -", (POLICY / "limits_tuned.csv").resolve())
print("  -", (POLICY / "limits_baseline_summary.json").resolve())
print("  -", (POLICY / "limits_tuned_summary.json").resolve())
print("  -", (POLICY / "new_batch_limits_compare.json").resolve())
print("\nComparison summary:\n", json.dumps(comparison, indent=2))
