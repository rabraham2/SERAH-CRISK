# crisk_new_batch_rethreshold.py
# Retune approvals/review band for THIS batch (no retrain)
import numpy as np, pandas as pd
from pathlib import Path

BASE = Path("Dataset"); POL = BASE / "crisk_policy"
INP  = POL / "new_batch_decisions.csv"   # must contain pd_month
OUTP = POL / "new_batch_decisions_tuned.csv"

APPROVAL_TARGET = 0.10     # 10% approvals
REVIEW_WIDTH    = 0.05     # 5% around the cutoff goes to Review

df = pd.read_csv(INP)
if "pd_month" not in df: raise RuntimeError("new_batch_decisions.csv must have pd_month")

thr = df["pd_month"].quantile(APPROVAL_TARGET)  # lower PD = safer (approve)
low  = np.maximum(APPROVAL_TARGET - REVIEW_WIDTH/2, 0.0)
high = np.minimum(APPROVAL_TARGET + REVIEW_WIDTH/2, 1.0)
q_low, q_high = df["pd_month"].quantile([low, high])

def decide(pd_m):
    if pd_m <= q_low:  return "Approve"
    if pd_m <= q_high: return "Review"
    return "Decline"

df["decision"] = df["pd_month"].apply(decide)
df.to_csv(OUTP, index=False)

print({
    "approval_rate": float((df.decision=="Approve").mean()),
    "review_rate":   float((df.decision=="Review").mean()),
    "decline_rate":  float((df.decision=="Decline").mean()),
    "thr_pd_month":  float(thr),
    "q_low": float(q_low), "q_high": float(q_high),
})
print("Saved:", OUTP.resolve())
