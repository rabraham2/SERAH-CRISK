# crisk_pick_thresholds.py — robust threshold & band picker (IDE-warning safe)

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

OUT = Path("Dataset/crisk_readout")
INP = OUT / "scored_cohort_1y.csv"
df = pd.read_csv(INP)

# 1) Auto-detect columns
SCORE_CANDS = [
    "score", "pd", "pd_month", "prob_default", "probability", "proba",
    "risk_score", "p_default", "p_bad", "pd_1y", "pd_next_month"
]
LABEL_CANDS = ["y", "target", "default", "default_flag", "label", "is_bad", "bad"]

def pick_col(cands: list[str], frame: pd.DataFrame) -> Optional[str]:
    for c in cands:
        if c in frame.columns:
            return c
    for c in frame.columns:
        cl = c.lower()
        if any(k in cl for k in ("pd", "prob", "score")):
            return c
    return None

score_col = pick_col(SCORE_CANDS, df)
if score_col is None:
    raise RuntimeError(
        f"Couldn't find a score/probability column in {INP}. "
        f"Expected one of {SCORE_CANDS} or a column containing 'pd'/'prob'/'score'."
    )

label_col = next((c for c in LABEL_CANDS if c in df.columns), None)
df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
if label_col is not None:
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)

# 2) Decide score direction (higher = riskier?)
higher_is_riskier = True
if label_col is not None:
    q_hi, q_lo = df[score_col].quantile([0.90, 0.10])
    top = df[df[score_col] >= q_hi]
    bot = df[df[score_col] <= q_lo]
    if not top.empty and not bot.empty:
        higher_is_riskier = float(top[label_col].mean()) >= float(bot[label_col].mean())

# 3) Policy knobs
APPROVAL_TARGET = 0.55   # approve ~55%
MAX_BAD_RATE    = 0.10   # among approved, default rate ≤10%
REVIEW_WIDTH    = 0.10   # middle 10% as "Review"

# 4) Compute approval threshold
def approval_mask(thr: float, higher_risky: bool) -> pd.Series:
    # explicit return type helps PyCharm infer Series (not bool)
    if higher_risky:
        return df[score_col].le(thr)
    else:
        return df[score_col].ge(thr)

thr_approve = float(df[score_col].quantile((1 - APPROVAL_TARGET) if higher_is_riskier else APPROVAL_TARGET))
mask_approve: pd.Series = approval_mask(thr_approve, higher_is_riskier)

def approved_bad_rate(mask: pd.Series) -> float:
    if label_col is None:
        return float("nan")
    n_app = int(mask.sum())
    if n_app == 0:
        return float("nan")
    return float(df.loc[mask, label_col].mean())

n_approved = int(mask_approve.sum())
approval_rate = n_approved / float(len(df))
bad_rate = approved_bad_rate(mask_approve)

# tighten approval threshold until bad-rate is met (if labels exist)
if label_col is not None and np.isfinite(bad_rate):
    target = APPROVAL_TARGET
    for _ in range(40):  # safety bound
        if bad_rate <= MAX_BAD_RATE:
            break
        target = max(0.20, target - 0.01)
        thr_approve = float(df[score_col].quantile((1 - target) if higher_is_riskier else target))
        mask_approve = approval_mask(thr_approve, higher_is_riskier)
        n_approved = int(mask_approve.sum())
        approval_rate = n_approved / float(len(df))
        bad_rate = approved_bad_rate(mask_approve)

# 5) Bands: Approve / Review / Decline
low_q  = max(0.0, 0.5 - REVIEW_WIDTH/2)
high_q = min(1.0, 0.5 + REVIEW_WIDTH/2)
q_low, q_high = df[score_col].quantile([low_q, high_q])

if higher_is_riskier:
    bins   = [-np.inf, thr_approve, q_high, np.inf]
    labels = ["Approve", "Review", "Decline"]
else:
    bins   = [-np.inf, q_low, thr_approve, np.inf]
    labels = ["Decline", "Review", "Approve"]

decision = pd.cut(df[score_col], bins=bins, labels=labels, include_lowest=True, right=True)

# 6) Save & summary
out = df.copy()
out["decision"] = decision
out["thr_approve"] = thr_approve
out["score_direction"] = "higher_is_riskier" if higher_is_riskier else "higher_is_safer"
out.to_csv(OUT / "decisions_1y.csv", index=False)

approve_mask: pd.Series = out["decision"].eq("Approve")
approval_rate_final = float(approve_mask.mean())  # boolean Series mean -> rate

summary = {
    "score_col": score_col,
    "label_col": label_col,
    "higher_is_riskier": higher_is_riskier,
    "thr_approve": thr_approve,
    "approval_rate": approval_rate_final,
    "counts": {k: int(v) for k, v in out["decision"].value_counts(dropna=False).items()},
}
if label_col is not None and np.isfinite(bad_rate):
    summary["approved_bad_rate"] = float(bad_rate)

print(summary)
print("Saved:", OUT / "decisions_1y.csv")
