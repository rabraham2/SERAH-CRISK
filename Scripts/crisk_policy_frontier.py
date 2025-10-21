# crisk_policy_frontier.py — approval-rate vs EL-rate and pick EL-cap point

import numpy as np, pandas as pd
from pathlib import Path

BASE = Path("Dataset")
READOUT = BASE / "crisk_readout"
OUT     = BASE / "crisk_policy"
OUT.mkdir(parents=True, exist_ok=True)

EL_CAP    = 0.10         # portfolio expected loss rate target (10%)
LGD, UTIL = 0.85, 0.40

df = pd.read_csv(READOUT / "scored_cohort_1y.csv")
if "pd_month" not in df or "target" not in df:
    raise RuntimeError("Need columns 'pd_month' and 'target' in scored_cohort_1y.csv")

pd_m = pd.to_numeric(df["pd_month"], errors="coerce").clip(0,1).fillna(0.0)
pd_1y = 1 - (1 - pd_m)**12
df["pd_1y"] = pd_1y

# sort from safest to riskiest
df = df.sort_values("pd_1y").reset_index(drop=True)

grid = np.linspace(0.05, 0.80, 76)  # try 5%..80% approvals
rows = []
n = len(df)
for a in grid:
    k = int(round(a * n))
    if k <= 0:
        continue
    approved = df.iloc[:k]
    el_rate = float((approved["pd_1y"] * LGD * UTIL).mean())  # portfolio EL rate
    bad_rate = float(approved["target"].mean())               # empirical label rate (for sanity)
    rows.append({"approval_rate": float(a),
                 "expected_loss_rate": el_rate,
                 "approved_bad_rate": bad_rate})

res = pd.DataFrame(rows)
res.to_csv(OUT / "policy_frontier.csv", index=False)

ok = res[res["expected_loss_rate"] <= EL_CAP].sort_values("approval_rate", ascending=False)
print(f"Saved frontier to: {(OUT/'policy_frontier.csv').resolve()}")
if ok.empty:
    print(f"No approval rate meets EL ≤ {EL_CAP:.0%}. Best (lowest EL) observed:")
    print(res.nsmallest(5, "expected_loss_rate").to_string(index=False))
else:
    choice = ok.iloc[0]
    print(f"Choose approval_rate ≈ {choice['approval_rate']:.2%} to meet EL ≤ {EL_CAP:.0%}")
    print(choice.to_string(index=False))
