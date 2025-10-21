# crisk_policy_finalize_from_cap.py
import numpy as np, pandas as pd
from pathlib import Path

BASE = Path("Dataset")
READOUT = BASE / "crisk_readout"
OUT     = BASE / "crisk_policy"
OUT.mkdir(parents=True, exist_ok=True)

EL_CAP    = 0.12      # ← relax to 12% (change if you want)
LGD, UTIL = 0.85, 0.40

df = pd.read_csv(READOUT / "scored_cohort_1y.csv")
pd_m = pd.to_numeric(df["pd_month"], errors="coerce").clip(0,1).fillna(0.0)
df["pd_1y"] = 1 - (1 - pd_m)**12
df = df.sort_values("pd_1y").reset_index(drop=True)

# find highest approval rate that satisfies the EL cap
grid = np.linspace(0.03, 0.80, 78)
ok = []
n = len(df)
for a in grid:
    k = int(round(a*n))
    if k <= 0:
        continue
    approved = df.iloc[:k]
    el_rate = float((approved["pd_1y"] * LGD * UTIL).mean())
    if el_rate <= EL_CAP:
        ok.append((a, el_rate))
if not ok:
    best = min(grid, key=lambda a: float((df.iloc[:max(1,int(round(a*n)))]["pd_1y"]*LGD*UTIL).mean()))
    print(f"No approval rate meets EL ≤ {EL_CAP:.0%}. Best approval~{best:.0%}.")
    a_pick = best
else:
    a_pick, el_pick = max(ok, key=lambda t: t[0])
    print(f"Picked approval ≈ {a_pick:.0%} meeting EL ≤ {EL_CAP:.0%} (EL≈{el_pick:.2%}).")

# threshold at that approval rate
thr = float(df["pd_1y"].quantile(a_pick))
df["decision"] = np.where(df["pd_1y"] <= thr, "Approve", "Decline")

# simple conservative limits (exposure control)
if "limit_bal" not in df: df["limit_bal"] = 50_000
mult = np.where(df["pd_1y"] <= thr/2, 1.2, 0.8)   # more for safest half of approved
df["limit_final"] = df["limit_bal"] * mult
df.loc[df["decision"] == "Decline", "limit_final"] = 0

approved = df[df["decision"] == "Approve"].copy()
portfolio = {
    "approval_rate": float((df["decision"] == "Approve").mean()),
    "expected_loss_rate": float((approved["pd_1y"]*LGD*UTIL).mean()) if len(approved) else 0.0,
    "sum_limit": float(approved["limit_final"].sum()),
    "avg_limit": float(approved["limit_final"].mean() if len(approved) else 0.0),
}

df.to_csv(OUT/"final_policy_decisions.csv", index=False)
pd.Series({"pd_threshold_1y": thr, "approval_rate": portfolio["approval_rate"]}).to_json(
    OUT/"final_policy_threshold.json", indent=2
)
pd.Series(portfolio).to_json(OUT/"final_policy_portfolio.json", indent=2)
print("Saved final policy pack to:", OUT.resolve())
print("Portfolio:", portfolio)
