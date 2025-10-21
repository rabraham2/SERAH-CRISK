# crisk_new_batch_report.py
# Creates per-group slice metrics + a couple of PNGs so you can present results
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("Dataset")
POL  = BASE / "crisk_policy"
OUT  = POL / "new_batch_report"
OUT.mkdir(parents=True, exist_ok=True)

# Load the decisions file created by crisk_apply.py
df = pd.read_csv(POL / "new_batch_decisions.csv")  # expects: decision, pd_month, (sex, age, limit_final) if present

# Basic overview
summary = {
    "n_applications": int(len(df)),
    "approval_rate": float((df["decision"]=="Approve").mean()),
    "review_rate":   float((df["decision"]=="Review").mean()),
    "decline_rate":  float((df["decision"]=="Decline").mean()),
}
if "limit_final" in df:
    summary.update({
        "avg_limit_approved": float(df.loc[df.decision=="Approve","limit_final"].mean()),
        "sum_limit_approved": float(df.loc[df.decision=="Approve","limit_final"].sum()),
    })

pd.Series(summary).to_json(OUT/"overview.json", indent=2)

# Slices (if columns exist)
slices = []
if "sex" in df:
    s = (df.groupby("sex", observed=True)
           .agg(n=("decision","size"),
                approval_rate=("decision", lambda x: np.mean(x=="Approve")),
                review_rate=("decision",   lambda x: np.mean(x=="Review")),
                decline_rate=("decision",  lambda x: np.mean(x=="Decline")),
                avg_pd=("pd_month","mean"))
           .reset_index())
    s.to_csv(OUT/"slice_sex.csv", index=False); slices.append(("sex", s))
if "age" in df:
    bins = [18,25,35,45,55,65,200]
    df["age_band"] = pd.cut(df["age"], bins=bins, right=False)
    s = (df.groupby("age_band", observed=True)
           .agg(n=("decision","size"),
                approval_rate=("decision", lambda x: np.mean(x=="Approve")),
                avg_pd=("pd_month","mean"))
           .reset_index())
    s.to_csv(OUT/"slice_age.csv", index=False); slices.append(("age_band", s))

# Simple charts
def barh_counts(col_name, data):
    plt.figure(figsize=(7,4))
    data = data.sort_values("n", ascending=True)
    plt.barh(data[col_name].astype(str), data["n"])
    plt.title(f"Count by {col_name}")
    plt.tight_layout(); plt.savefig(OUT/f"counts_{col_name}.png", dpi=140); plt.close()

for name, s in slices:
    barh_counts(name, s)

# Decision mix pie
plt.figure(figsize=(4.5,4.5))
df["decision"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.ylabel(""); plt.title("Decision mix")
plt.tight_layout(); plt.savefig(OUT/"decision_mix.png", dpi=140); plt.close()

print("✓ Report saved to:", OUT.resolve())
print(summary)
