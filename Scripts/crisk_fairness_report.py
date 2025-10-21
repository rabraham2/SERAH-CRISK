# crisk_fairness_report.py
# Fairness slices for the latest batch decisions (JSON-safe)

from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE = Path("Dataset")
POL  = BASE / "crisk_policy"
OUT  = POL / "fairness_report"
OUT.mkdir(parents=True, exist_ok=True)

# ---- load ----
# If your file name differs, change below:
decisions_path = POL / "new_batch_decisions.csv"
decisions = pd.read_csv(decisions_path)

# Ensure required columns exist
if "decision" not in decisions.columns or "pd_month" not in decisions.columns:
    raise RuntimeError(f"Expected columns ['decision','pd_month'] in {decisions_path}")

# Optional columns for slices/metrics
has_sex = "sex" in decisions.columns
has_age = "age" in decisions.columns
has_limit = "limit_final" in decisions.columns

# Build age bands for reporting if age exists
if has_age:
    decisions["age_band"] = pd.cut(
        decisions["age"],
        bins=[18,25,35,45,55,65,200],
        right=False,
        include_lowest=True
    )

# Normalize decision text
decisions["decision"] = decisions["decision"].astype(str)

# ---- helpers ----
def json_safe(v):
    """Convert numpy and Interval types to JSON-safe Python primitives/strings."""
    if isinstance(v, (np.generic,)):  # np.integer/np.floating/bool_
        return v.item()
    if isinstance(v, pd.Interval):
        return str(v)
    return v

def df_to_records_json_safe(df: pd.DataFrame):
    recs = df.to_dict(orient="records")
    return [{k: json_safe(v) for k, v in r.items()} for r in recs]

def by_group(df: pd.DataFrame, group_col: str):
    if group_col not in df.columns:
        return None
    g = df.groupby(group_col, observed=True)  # observed=True silences the FutureWarning

    # approval/share metrics
    approved_mask = df["decision"].str.lower().eq("approve")
    review_mask   = df["decision"].str.lower().eq("review")
    decline_mask  = df["decision"].str.lower().eq("decline")

    # group aggregations
    agg = g.agg(
        n=("decision", "size"),
        approval_rate=("decision", lambda s: s.str.lower().eq("approve").mean()),
        review_rate=("decision",   lambda s: s.str.lower().eq("review").mean()),
        decline_rate=("decision",  lambda s: s.str.lower().eq("decline").mean()),
        avg_pd=("pd_month", "mean"),
    )

    # avg limit only across approved rows if we have limits
    if has_limit:
        # compute approved-only means per group
        approved = df.loc[approved_mask]
        if not approved.empty:
            mean_lim = approved.groupby(group_col, observed=True)["limit_final"].mean()
            agg["avg_limit_approved"] = mean_lim
        else:
            agg["avg_limit_approved"] = np.nan

    agg = agg.reset_index()

    # Convert group key column to string so it's JSON-safe (Intervals -> str)
    agg[group_col] = agg[group_col].astype(str)
    return agg

# ---- build group reports ----
reports = {}
for col in (["sex"] if has_sex else []) + (["age_band"] if has_age else []):
    tbl = by_group(decisions, col)
    if tbl is None or tbl.empty:
        continue
    # reference group = most frequent
    ref = tbl.loc[tbl["n"].idxmax(), col]
    ref_row = tbl[tbl[col] == ref].iloc[0]
    # ratio vs reference (avoid 0-div)
    denom = ref_row["approval_rate"] if ref_row["approval_rate"] else 1e-9
    tbl["approval_rate_ratio_vs_ref"] = tbl["approval_rate"] / denom

    # Save CSV and bundle JSON-safe table
    tbl.to_csv(OUT / f"by_{col}.csv", index=False)
    reports[col] = {
        "reference_group": str(ref),
        "table": df_to_records_json_safe(tbl),
    }

# ---- overall summary ----
overall = {
    "approval_rate": float(decisions["decision"].str.lower().eq("approve").mean()),
    "review_rate":   float(decisions["decision"].str.lower().eq("review").mean()),
    "decline_rate":  float(decisions["decision"].str.lower().eq("decline").mean()),
    # PD * LGD(0.85) * Util(0.40) as an EL-rate proxy (no $ limits in denominator)
    "expected_loss_rate_proxy": float((pd.to_numeric(decisions["pd_month"], errors="coerce")
                                       .clip(0,1)
                                       .fillna(0.0) * 0.85 * 0.40).mean()),
}

summary = {
    "n_applications": int(len(decisions)),
    "overall": overall,
    "group_reports": reports,
}

with open(OUT / "fairness_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("✓ Fairness report saved to:", OUT.resolve())
print(json.dumps(overall, indent=2))
for k in reports:
    print(f"- Group slice written: by_{k}.csv (ref={reports[k]['reference_group']})")
