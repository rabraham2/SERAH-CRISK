# crisk_score_readout.py — Score cohort, yearly risk, slices by age/gender, and readout

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix
)

BASE = Path("Dataset")
IN_CLEAN  = BASE / "crisk_clean"     # where data_preparation.py wrote splits
IN_MODEL  = BASE / "crisk_models"    # where crisk_train.py saved the model
OUT       = BASE / "crisk_readout"
OUT.mkdir(parents=True, exist_ok=True)

# 1) Load model + data
pipe = joblib.load(IN_MODEL / "crisk_model.pkl")

# Use the full processed dataset to “score everyone”
df_all = pd.read_parquet(IN_CLEAN / "dataset_all.parquet").reset_index(drop=True)
y_true = df_all["target"].astype(int).to_numpy()
X_all  = df_all.drop(columns=["target"])

# 2) Monthly PD, Annual PD, risk bands
pd_month = pipe.predict_proba(X_all)[:, 1]
# Convert “next-month default” prob to 12-month prob (independence assumption)
pd_year = 1.0 - (1.0 - pd_month) ** 12

# risk band cutoffs (editable)
def band_from_pd(prob: float) -> str:
    if prob >= 0.20:   # >= 20% annual PD
        return "High"
    if prob >= 0.05:   # 5–20%
        return "Medium"
    return "Low"       # < 5%

risk_band = np.vectorize(band_from_pd)(pd_year)

# 3) Threshold predictions
thr_star = 0.5
metrics_path = IN_MODEL / "metrics.json"
if metrics_path.exists():
    try:
        meta = json.loads(metrics_path.read_text(encoding="utf-8"))
        thr_star = float(meta["valid"]["threshold_star"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        # leave thr_star at default 0.5 if metrics file malformed/missing fields
        pass

y_hat = (pd_month >= thr_star).astype(int)   # monthly PD thresholding for classification view

# 4) Readout metrics
auc   = roc_auc_score(y_true, pd_month)
ap    = average_precision_score(y_true, pd_month)
brier = brier_score_loss(y_true, pd_month)
rep   = classification_report(y_true, y_hat, output_dict=True, zero_division=0)
cm    = confusion_matrix(y_true, y_hat)

summary = {
    "AUC_month": float(auc),
    "AP_month": float(ap),
    "Brier_month": float(brier),
    "threshold_star_month": float(thr_star),
    "confusion_at_star": cm.tolist(),
    "report_at_star": rep,
    "base_prevalence": float(np.mean(y_true)),
}

(OUT / "readout_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

# 5) Demographic slices
scored = X_all.copy()
scored["pd_month"] = pd_month
scored["pd_year"]  = pd_year
scored["risk_band"] = risk_band
scored["target"]   = y_true

# Age bins for readable summaries
age_bins  = [0, 25, 35, 45, 55, 65, 200]
age_lbls  = ["<=25","26-35","36-45","46-55","56-65","65+"]
scored["age_band"] = pd.cut(scored["age"], bins=age_bins, labels=age_lbls, right=True, include_lowest=True)

# Sex mapping (dataset: 1=male, 2=female)
sex_map = {1: "Male", 2: "Female"}
scored["sex_label"] = scored["sex"].map(sex_map).fillna(scored["sex"].astype(str))

# Slices: by sex, by age band, by sex×age
agg_cols = {
    "n": ("pd_year", "size"),
    "mean_pd_month": ("pd_month", "mean"),
    "mean_pd_year":  ("pd_year",  "mean"),
    "share_high":    ("risk_band", lambda s: np.mean(s=="High")),
    "share_medium":  ("risk_band", lambda s: np.mean(s=="Medium")),
    "share_low":     ("risk_band", lambda s: np.mean(s=="Low")),
    "avg_limit_bal": ("limit_bal","mean"),
    "actual_default_rate": ("target","mean"),
}

by_sex     = scored.groupby("sex_label", observed=True).agg(**agg_cols).reset_index()
by_age     = scored.groupby("age_band",  observed=True).agg(**agg_cols).reset_index()
by_sex_age = scored.groupby(["sex_label","age_band"], observed=True).agg(**agg_cols).reset_index()

by_sex.to_csv(OUT / "slice_by_sex.csv", index=False)
by_age.to_csv(OUT / "slice_by_age.csv", index=False)
by_sex_age.to_csv(OUT / "slice_by_sex_age.csv", index=False)

# 6) Illustrative credit-limit policy
def suggest_limit(row: pd.Series) -> float:
    base = float(row["limit_bal"])
    band = row["risk_band"]
    if band == "Low":
        return min(base * 1.10, base * 1.50)      # +10%, cap +50%
    if band == "Medium":
        return base                                # hold
    return max(base * 0.60, 10000.0)               # -40%, floor 10k

scored["limit_suggested"] = scored.apply(suggest_limit, axis=1)

# Save full scored cohort (careful if this is sensitive)
keep_cols = [
    "sex", "sex_label", "education", "marriage", "age", "age_band", "limit_bal",
    "pd_month", "pd_year", "risk_band", "limit_suggested", "target"
]
scored[keep_cols].to_csv(OUT / "scored_cohort_1y.csv", index=False)

# Quick rollup of policy impact (illustrative)
policy_rollup = (scored
                 .groupby("risk_band", as_index=False)
                 .agg(n=("pd_year","size"),
                      avg_pd_year=("pd_year","mean"),
                      current_limit=("limit_bal","mean"),
                      suggested_limit=("limit_suggested","mean")))
policy_rollup["avg_limit_delta"] = policy_rollup["suggested_limit"] - policy_rollup["current_limit"]
policy_rollup.to_csv(OUT / "policy_rollup.csv", index=False)

print("✓ Readout saved to:", OUT.resolve())
print("AUC (monthly):", round(auc, 4), "| AP:", round(ap, 4), "| Brier:", round(brier, 4))
print("Risk-band counts:", dict(scored["risk_band"].value_counts()))
print("Files:")
for fname in ["readout_summary.json","slice_by_sex.csv","slice_by_age.csv",
              "slice_by_sex_age.csv","scored_cohort_1y.csv","policy_rollup.csv"]:
    print("  -", OUT / fname)
