# data_preparation.py — Data Cleaning and Formatting (robust target detection)

import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

# Paths
BASE = Path("Dataset")
OUT = BASE / "crisk_clean"
OUT.mkdir(parents=True, exist_ok=True)

# Helpers
def load_uci_dataset(base: Path) -> pd.DataFrame:
    """Load UCI Taiwan credit card dataset from CSV/XLSX/XLS."""
    candidates = [
        base / "UCI_Credit_Card.csv",
        base / "UCI_Credit_Card.xlsx",
        base / "UCI_Credit_Card.xls",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise FileNotFoundError("Could not find UCI_Credit_Card.{csv,xlsx,xls} under 'Dataset/'.")

    suffix = src.suffix.lower()
    if suffix == ".csv":
        data = pd.read_csv(src)
    elif suffix == ".xlsx":
        data = pd.read_excel(src, header=1, engine="openpyxl")  # UCI Excel header starts at row 1
    elif suffix == ".xls":
        # If you get an engine error: pip install "xlrd==1.2.0"
        data = pd.read_excel(src, header=1, engine="xlrd")
    else:
        raise ValueError(f"Unsupported file type: {src}")

    data.columns = [c.strip() for c in data.columns]
    return data

def norm_colname(name: str) -> str:
    """Lowercase, strip, and remove all non-alphanumeric characters."""
    s = re.sub(r"[^0-9a-zA-Z]+", "", str(name).strip().lower())
    return s

def pick_column(df: pd.DataFrame, candidates) -> str | None:
    """
    Return the actual column name in df that matches any candidate
    when normalized by norm_colname(). None if not found.
    """
    normalized_map = {norm_colname(c): c for c in df.columns}
    for cand in candidates:
        key = norm_colname(cand)
        if key in normalized_map:
            return normalized_map[key]
    return None


# Load

raw = load_uci_dataset(BASE)


# Robust rename (handles dotted/space variants)
# Map of desired_name -> list of candidate variants we might see
wanted = {
    "limit_bal": ["LIMIT_BAL", "limit_bal"],
    "sex": ["SEX", "sex"],
    "education": ["EDUCATION", "education"],
    "marriage": ["MARRIAGE", "marriage"],
    "age": ["AGE", "age"],
    "pay_1": ["PAY_0", "PAY_1", "pay_0", "pay_1"],
    "pay_2": ["PAY_2", "pay_2"],
    "pay_3": ["PAY_3", "pay_3"],
    "pay_4": ["PAY_4", "pay_4"],
    "pay_5": ["PAY_5", "pay_5"],
    "pay_6": ["PAY_6", "pay_6"],
    "bill_amt1": ["BILL_AMT1", "bill_amt1"],
    "bill_amt2": ["BILL_AMT2", "bill_amt2"],
    "bill_amt3": ["BILL_AMT3", "bill_amt3"],
    "bill_amt4": ["BILL_AMT4", "bill_amt4"],
    "bill_amt5": ["BILL_AMT5", "bill_amt5"],
    "bill_amt6": ["BILL_AMT6", "bill_amt6"],
    "pay_amt1": ["PAY_AMT1", "pay_amt1"],
    "pay_amt2": ["PAY_AMT2", "pay_amt2"],
    "pay_amt3": ["PAY_AMT3", "pay_amt3"],
    "pay_amt4": ["PAY_AMT4", "pay_amt4"],
    "pay_amt5": ["PAY_AMT5", "pay_amt5"],
    "pay_amt6": ["PAY_AMT6", "pay_amt6"],
    # target variations across CSV vs Excel
    "target": [
        "default payment next month",    # Excel
        "default.payment.next.month",    # CSV
        "target"
    ],
}

rename_map = {}
for desired, cands in wanted.items():
    actual = pick_column(raw, cands)
    if actual is not None:
        rename_map[actual] = desired

df = raw.rename(columns=rename_map)

# Drop ID if present (handles ID/id/Id)
id_col = pick_column(df, ["ID", "id"])
if id_col:
    df = df.drop(columns=[id_col])

# Ensure target exists
if "target" not in df.columns:
    raise RuntimeError(
        "Could not find the target column. Expected one of "
        "'default.payment.next.month' (CSV) or 'default payment next month' (Excel)."
    )

# Clean infinities and fully-empty rows
df = df.replace({np.inf: np.nan, -np.inf: np.nan})
df = df.dropna(how="all")

# Cast target to int
df["target"] = df["target"].astype(int)

# Mild capping of extreme va; for numeric columns (exclude target)
num_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != "target"]
if num_cols:
    lower_q = df[num_cols].quantile(0.001)
    upper_q = df[num_cols].quantile(0.999)
    df[num_cols] = df[num_cols].clip(lower=lower_q, upper=upper_q, axis=1)


# Stratified 60/20/20 split on the target
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.40, random_state=42)  # -> 60/40
train_idx, temp_idx = next(sss1.split(df, df["target"]))
train = df.iloc[train_idx].reset_index(drop=True)
temp = df.iloc[temp_idx].reset_index(drop=True)

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)  # -> 20/20
valid_idx, test_idx = next(sss2.split(temp, temp["target"]))
valid = temp.iloc[valid_idx].reset_index(drop=True)
test = temp.iloc[test_idx].reset_index(drop=True)


# Save outputs
# If you get an error here, install a Parquet engine:  pip install pyarrow
train.to_parquet(OUT / "train.parquet", index=False)
valid.to_parquet(OUT / "valid.parquet", index=False)
test.to_parquet(OUT / "test.parquet", index=False)
df.to_parquet(OUT / "dataset_all.parquet", index=False)

# simple schema (dtype per column)
pd.Series(df.dtypes.astype(str)).to_csv(OUT / "schema.csv")

print("Saved to:", OUT.resolve())
print("Shapes -> train:", train.shape, "valid:", valid.shape, "test:", test.shape)
print("Class balance (train):", train["target"].value_counts(normalize=True).to_dict())
