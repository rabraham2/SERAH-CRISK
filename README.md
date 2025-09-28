# SERAH - CRISK (Credit Risk Insights & Scoring Kit)

CRISK is an end-to-end credit risk pipeline on the UCI Taiwan Credit Card dataset. It cleans and prepares data, trains a leak-safe model to predict the probability of default (PD), calibrates and evaluates performance, and then converts scores into Approve/Review/Decline decisions and credit limits under explicit portfolio loss constraints.

---

## Table of Contents
1. [Overview](#overview)
2. [Primary Research Question](#primary-research-question)
3. [System Requirements](#system-requirements)
4. [Installation and Setup](#installation-and-setup)
5. [Instructions to Run](#instructions)
6. [Scripts and Roles](#scripts-and-roles)
7. [Running the Scripts](#running-the-scripts)
8. [Project Structure](#project-structure)
9. [License](#license)

---


## Overview
Assess consumer credit risk end-to-end: estimate monthly and 1-year PD (probability of default), make Approve / Review / Decline decisions, assign credit limits under an expected-loss cap, and monitor portfolio & fairness—complete with ready-to-share CSV/JSON readouts, charts, and an optional REST API.

This project turns the public UCI Taiwan Credit Card Default data into a production-style pipeline:

1. Ingest & clean the raw dataset (schema normalisation, robust clipping, stratified 60/20/20 split).
2. Train a leak-safe scikit-learn pipeline (One-Hot + Robust scaling → HistGradientBoosting), and report AUC/AP/Brier, confusion matrices, ROC/PR curves, and permutation importance.
3. Score & cohort readout: produce monthly PD, compound to 1-year PD, create cohort-level metrics and calibrated risk bands, and export slice reports by sex and age.
4. Decision policy: pick thresholds to hit target approval and bad-rate constraints (grid sweep + ROC-based “Youden-J” option), and save Approve/Review/Decline decisions.
5. Limit assignment: translate decisions into credit limits using band-based multipliers and an EL cap (EL = PD × LGD × Util × Limit); output portfolio roll-ups and summaries.
6. Portfolio optimisation: run policy sweeps and frontier analysis to find approval rates that satisfy an EL-rate ceiling; finalise a policy pack (thresholds + decisions + portfolio KPIs).
7. Fairness & governance: generate group-level metrics (approval, decline, bad-rate proxies) across protected proxies (sex, age bands) with CSV/JSON artefacts for review.
8. Serve & operate: score new applications in batch (reports & plots) or via a FastAPI REST endpoint that returns PD, decision, and recommended limit.

It’s designed to be reproducible, leak-safe, and actionable—outputs include concrete lending decisions and limit recommendations, along with portfolio-level KPIs, fairness slices, and deployment hooks.

---

## Primary Research Question
Given six months of repayment behaviour, billing and payment history, credit limits, and demographics, how should we approve/review/decline applications and set credit limits so we maximise safe growth while keeping expected loss under a portfolio cap over the next year?


- **Aims and Objectives**:

  a. Build a robust monthly default (PD) model with strict no-leakage practices.
  
  b. Validate & calibrate predictions (AUC/AP/Brier), and convert monthly PD → 1-year PD.
  
  c. Design an approval policy (Approve/Review/Decline) that meets approval% and bad-rate targets under an EL cap.
  
  d. Assign credit limits using banded multipliers and an EL = PD × LGD × Util cap; produce portfolio roll-ups.
  
  e. Communicate results via KPIs, CSV/JSON artefacts, charts, and an optional REST API for scoring.


- **Dataset (What, Why, & How)**:

  UCI Taiwan Credit Card Default dataset (2005; 30,000 cardholders; 24 features + label) captures pre-target behaviour and demographics suitable for PD modelling.
  
    A - Raw grain: one customer snapshot with six months of history:

        1. Demographics & limit: SEX, EDUCATION, MARRIAGE, AGE, LIMIT_BAL
        2. Repayment status (ordered categorical): PAY_1..PAY_6
        3. Billing amounts: BILL_AMT1..BILL_AMT6
        4. Payment amounts: PAY_AMT1..PAY_AMT6
        5. Label: default payment next month (binary)
    
    B - Modelling grain: features available at month t (the six prior months’ signals), target is default at t+1 → a monthly PD. We produce 1-year PD via compounding:

        pd_1y=1−(1−pd_month)12
    
    C - Signals: demographics, limit, most-recent delinquency codes (PAY_1 highest weight), bill & payment levels, ratios/robust scaling, one-hot for ordered categories, and risk bands (A/B/C/D) derived from pd_1y.
    
    D - Why it fits: widely used benchmark with rich pre-target behaviour, clear time anchor (statement month), reasonable class balance (~22% default), and fairness-proxy attributes (sex, age) enabling group slice monitoring. It supports policy simulation (thresholds, approval rates, EL caps) using only data available before the target month.
  
  We anchor time at the statement month. Only features from months t−1…t−6 enter the model; no target-month or post-outcome fields are used, preventing leakage.

  <i>A comprehensive Data Dictionary is included as <b>Data Dictionary.md</b>.</i>


- **Methods Used**:

  a) Feature engineering - Cast repayment status as ordered categories; robust-scale bills/payments; clip outliers; derive monthly PD (target) and 1-year PD via compounding pd_1y=1−(1−pd_month)12. Produce risk bands (A/B/C/D) from pd_1y.

  b) Modelling - scikit-learn Pipeline: ColumnTransformer with OneHotEncoder (SEX/EDUCATION/MARRIAGE + PAY_1..PAY_6) and RobustScaler (limits, bills, payments, age) → HistGradientBoostingClassifier. Stratified 60/20/20 splits; metrics on valid and test: AUC, AP, Brier, and confusion matrices. Permutation importance for interpretability.

  c) Calibration & readout - Save ROC/PR curves, compute Youden-J threshold candidate, generate scored cohort with monthly PD and derived pd_1y; produce slice reports by sex and age bands for parity checks.

  d) Policy (thresholds) - construct Approve / Review / Decline using a monthly-PD threshold with an optional review window—utilities to pick thresholds by target approval% and max bad-rate on approved.

  e) Limits (EL-cap) - propose limits per applicant via banded multipliers (by A/B/C/D) and cap with EL = PD × LGD × Util × Limit ≤ EL_budget × Limit. Portfolio roll-ups by risk band, policy sweeps and frontier to meet portfolio EL cap.

  f) Fairness & monitoring - report approval/decline rates, bad-rates, and (proxy) EL by sex/age slices; export CSV/JSON artefacts and charts.

  g) Serving – minimal FastAPI /score endpoint that takes applicant JSON, returns pd_month, pd_1y, decision, and limit using the saved model and policy thresholds.
  
---

## System Requirements
- **Operating System**: Windows, macOS, or Linux
- **Python Version:**: Python 3.10–3.12
- **Python Packages**:
  - pip install pandas numpy scikit-learn joblib pyarrow matplotlib seaborn \
            openpyxl xlrd==1.2.0 fastapi "uvicorn[standard]" "pydantic>=2,<3"
- **Memory**: ~1–2 GB is plenty for the UCI dataset (30k rows)

---

## Installation and Setup
1. **Clone the Repository**:
   - Click the green "**Code**" button in this GitHub repository.
   - Copy the HTTPS or SSH link provided (e.g., `https://github.com/rabraham2/SERAH-CRISK.git`).
   - Open your terminal or command prompt and run:
     ```bash
     git clone https://github.com/rabraham2/SERAH-CRISK.git
     ```

2. **Install Required Python Packages**:
   Open **PyCharm** or **Other IDE-Enabled Coding Platform** and install the necessary packages:

```python
   pip install pandas numpy scikit-learn joblib pyarrow matplotlib seaborn \
            openpyxl xlrd==1.2.0 fastapi "uvicorn[standard]" "pydantic>=2,<3"
```

3. **Processed and Cleaned Dataet**:
   - The original and unprocessed version of the real dataset can be found in the folder Data in GitHub.
   - If you need to run the processed and data-cleaned version of the dataset after running the Data Preparation Script directly, use the file in the folder Dataset in GitHub.

---

## Instructions

Step 1: Audit & Missingness
Run: python data_preparation.py (first half prints schema/shape info)
  - Produces: Dataset/crisk_clean/
  - schema.csv – dtypes after normalisation
  - Console audit: row counts, class balance, basic quantiles

Step 2: Cleaning & Split → crisk_clean/
Run: python data_preparation.py
  - Output: train.parquet (60%), valid.parquet (20%), test.parquet (20%)
  - Output: dataset_all.parquet (full post-clean)
  - Output: schema.csv (final column names, including target)

Step 3: Feature pipeline (no leak)
Built inside training pipeline (crisk_train.py) using ColumnTransformer
  - One-hot: sex, education, marriage, pay_1..pay_6
  - Robust-scale: limit_bal, age, bill_amt1..6, pay_amt1..6
  - Outlier clipping at 0.1% / 99.9% done in Step 2

Step 4: Train & Validate
Run: python crisk_train.py
Outputs → Dataset/crisk_models/
  - crisk_model.pkl – fitted Pipeline(preprocess + HistGB)
  - metrics.json – AUC/AP/Brier + confusion matrices (valid & test)
  - roc_valid.csv, pr_valid.csv, roc_test.csv, pr_test.csv
  - feature_importance_validation.csv (permutation importance)
  - Console: Validation/Test AUC, Youden-J threshold*, Top features

Step 5: Readout & 1-Year PD Cohort
Run: python crisk_readout.py
Outputs → Dataset/crisk_readout/
  - scored_cohort_1y.csv – base features + pd_month, pd_1y, target
  - readout_summary.json – AUC/AP/Brier, counts by risk band
  - Slices for parity: slice_by_sex.csv, slice_by_age.csv, slice_by_sex_age.csv
  - policy_rollup.csv – band/threshold summaries
  - Curves reused from Step 4

Step 6: Thresholds → Approve / Review / Decline
Run: python crisk_pick_thresholds.py
Outputs → Dataset/crisk_readout/
  - decisions_1y.csv – per-row decision with chosen thresholds
  - Console summary: approval rate, bad-rate among approved, thresholds used
  - Optionally fix the business threshold set:
  - Policy frontier: python crisk_policy_frontier.py
  - Policy sweep: python crisk_policy_sweep.py
Outputs → Dataset/crisk_policy/
- policy_frontier.csv, policy_sweep.csv – approval vs EL/bad-rate trade-offs

Step 7: Limits & Portfolio (EL-cap)
Run: python crisk_limit_strategy.py
Outputs → Dataset/crisk_policy/
  - decisions_with_limits.csv – decision, pd_1y, risk_band, limit_final
  - portfolio_by_risk_band.csv – apps, avg_pd, avg/sum limit, expected loss by band
  - portfolio_summary.json – approval rate, total/avg limit, expected-loss rate
  - If you lock a final policy (e.g., “approve ~5% with EL ≤ 12%”), run your chosen option step to emit:
  - final_policy_threshold.json – { "thr_pd_month": ..., "review_upper": ... }
  - final_policy_decisions.csv – decisions at your chosen policy

Step 8: New Batch Scoring & Reports
Run (score):
  - python crisk_new_batch_score.py --input Dataset/new_apps.csv
  - # (no input) creates a demo at Dataset/new_apps_demo.csv
Produces → Dataset/crisk_policy/
  - new_batch_decisions.csv – pd_month, pd_1y, decision, limit_final
  - new_batch_summary.json – approval/review/decline, threshold used, limit stats
Run (QA visuals & slices):
  - python crisk_new_batch_report.py
Outputs → Dataset/crisk_policy/new_batch_report/
  - decision_mix.png, counts_sex.png, counts_age_band.png, overview.json, slice_sex.csv, slice_age.csv
Run (quick re-threshold on same batch):
  - python crisk_new_batch_rethreshold.py
Outputs → Dataset/crisk_policy/
  - new_batch_decisions_tuned.csv – updated decisions based on batch quantiles

Step 9: Fairness & Governance
Run: python crisk_fairness_report.py
Outputs → Dataset/crisk_policy/fairness_report/
  - by_sex.csv, by_age_band.csv – approval/decline rates, bad-rates, proxy EL
  - fairness_summary.json – portfolio-level rates + notes
    (Charts may be emitted alongside CSVs; use them to monitor parity over time.)

Step 10: Serve
Run API:
    uvicorn crisk_service:app --host 0.0.0.0 --port 8000 --reload
Endpoints
    GET /health → model/policy health
    POST /score → per-applicant: pd_month, pd_1y, decision, limit
Consumes:
    {
      "applicants": [{
        "limit_bal": 120000, "sex": 2, "education": 2, "marriage": 1, "age": 35,
        "pay_1": 0, "pay_2": 0, "pay_3": 0, "pay_4": 0, "pay_5": 0, "pay_6": 0,
        "bill_amt1": 30000, "bill_amt2": 28000, "bill_amt3": 26000,
        "bill_amt4": 24000, "bill_amt5": 22000, "bill_amt6": 20000,
        "pay_amt1": 30000, "pay_amt2": 28000, "pay_amt3": 26000,
        "pay_amt4": 24000, "pay_amt5": 22000, "pay_amt6": 20000,
        "applicant_id": "A-001"
      }]
    }

Where things land (quick map)
  - Dataset/crisk_clean/ – cleaned splits & schema
  - Dataset/crisk_models/ – model pickle + validation metrics/curves/importances
  - Dataset/crisk_readout/ – scored cohort, 1-year PD, slices, policy rollups
  - Dataset/crisk_policy/ – thresholds, decisions, limits, portfolio summaries, sweeps/frontier, new-batch artefacts, fairness reports
All steps are idempotent and safe to re-run; artefacts are versioned by folder. Adjust LGD, UTIL, EL budget, and risk band cut-offs to your portfolio needs.

---

## Scripts and Roles

|  #  | Script (role)                         | What it does                          | Key inputs                             |                  Key outputs                     |
|-----|---------------------------------------|---------------------------------------|----------------------------------------|--------------------------------------------------|
|  1  | data_preparation.py                   | Clean & normalize the Taiwan credit   | Dataset/UCI_Credit_Card.{csv/xlsx/xls} | Dataset/crisk_clean/{train,valid,test}.parquet,  |
|     |                                       | dataset; clip outliers; stratified    |                                        | dataset_all.parquet, schema.csv                  |
|     |                                       | 60/20/20 split; write schema          |                                        |                                                  | 
|     |                                       |                                       |                                        |                                                  |
|  2  | crisk_train.py                        | Build preprocessing (OHE+RobustScaler)| Dataset/crisk_clean/                   | Dataset/crisk_models/crisk_model.pkl,            |
|     |                                       | + HistGradientBoosting model; train;  | {train,valid,test}.parquet             | metrics.json, roc_valid.csv, pr_valid.csv,       |
|     |                                       | evaluate (AUC/AP/Brier); curves &     |                                        | roc_test.csv, pr_test.csv,                       |
|     |                                       | permutation importances               |                                        | feature_importance_validation.csv                |
|     |                                       |                                       |                                        |                                                  |
|  3  | crisk_readout.py                      | Score monthly PD (pd_month) and       | crisk_model.pkl,                       | Dataset/crisk_readout/scored_cohort_1y.csv,      |
|     |                                       | annualize (pd_1y); make cohort        | dataset_all.parquet                    | readout_summary.json, slice_by_sex.csv,          |
|     |                                       | readout & fairness slices             |                                        | slice_by_age.csv, slice_by_sex_age.csv,          |
|     |                                       |                                       |                                        | policy_rollup.csv                                |
|     |                                       |                                       |                                        |                                                  |
|  4  | crisk_pick_thresholds.py              | Pick Approve/Review/Decline           | scored_cohort_1y.csv                   | Dataset/crisk_readout/decisions_1y.csv, console  |
|     |                                       | thresholds by target approval &       |                                        | summary (approval rate, bad-rate, thresholds)    |                                |     |                                       | max bad-rate; label decisions         |                                        |                                                  |
|     |                                       |                                       |                                        |                                                  |
|  5  | crisk_policy_frontier.py              | Compute approval-rate vs              | scored_cohort_1y.csv                   | Dataset/crisk_policy/policy_frontier.csv,        |
|     |                                       | expected-loss-rate fronti             |                                        |  console pick (or best observed if none meet cap) |
|     |                                       | find highest approval meeting EL-cap  |                                        |                                                  |
|     |                                       |                                       |                                        |                                                  |  
|  6  | crisk_policy_sweep.py                 | Grid search over approval targets &   | scored_cohort_1y.csv                   | Dataset/crisk_policy/policy_sweep.csv,           |
|     |                                       | bad-rate caps; evaluate expected      |                                        | console top candidates                           |    
|     |                                       | loss under limit policy               |                                        |                                                  |
|     |                                       |                                       |                                        |                                                  |
|  7  | crisk_limit_strategy.py               | Assign risk bands (by pd_1y),         | decisions_1y.csv,                      | Dataset/crisk_policy/decisions_with_limits.csv,  |
|     |                                       | compute baseline limits × band        | scored_cohort_1y.csv                   | portfolio_by_risk_band.csv,                      |
|     |                                       | multipliers, apply EL cap, and        |                                        | portfolio_summary.json                           |
|     |                                       | build portfolio KPIs                  |                                        |                                                  | 
|     |                                       |                                       |                                        |                                                  |
|  8  | crisk_policy_finalize.py (Option A/B) | Freeze a final policy (approval % /   | scored_cohort_1y.csv (and              | Dataset/crisk_policy/final_policy_threshold.json,|
|     |                                       | EL-cap); emit reusable policy pack    | frontier/sweep results)                | final_policy_decisions.csv,                      |  
|     |                                       |                                       |                                        | final_policy_portfolio.json                      |
|     |                                       |                                       |                                        |                                                  |
|  9  | crisk_new_batch_score.py              | Score new applicants; apply frozen    | Dataset/new_apps.csv (or auto          | Dataset/crisk_policy/new_batch_decisions.csv,    |
|     |                                       | policy thresholds; assign final limits| demo if missing), crisk_model.pkl,     | new_batch_summary.json                           |
|     |                                       |                                       | final_policy_threshold.json            |                                                  |
|     |                                       |                                       |                                        |                                                  |
| 10  | crisk_new_batch_report.py             | Batch QA & visuals: decision mix,     | new_batch_decisions.csv                | Dataset/crisk_policy/new_batch_report/           |
|     |                                       | approval by sex/age; export slices    |                                        | {decision_mix.png, counts_sex.png,               |
|     |                                       |                                       |                                        | counts_age_band.png, overview.json,              |
|     |                                       |                                       |                                        | slice_sex.csv, slice_age.csv}                    |
|     |                                       |                                       |                                        |                                                  |
| 11  | crisk_new_batch_rethreshold.py        | Light re-threshold on the incoming    | new_batch_decisions.csv (or            | Dataset/crisk_policy/                            |       
|     |                                       | batch (e.g., target 7.5% approvals);  | new_apps.csv)                          | new_batch_decisions_tuned.csv,                   |
|     |                                       | re-label decisions                    |                                        | console summary (new thresholds & rates)         |
|     |                                       |                                       |                                        |                                                  |
| 12  | crisk_new_batch_limits_compare.py     | Compare baseline vs tuned batch       | new_batch_decisions.csv,               | Dataset/crisk_policy/limits_baseline.csv,        |
|     |                                       | policies; write per-app limits and    | new_batch_decisions_tuned.csv          | limits_tuned.csv, limits_baseline_summary.json,  |
|     |                                       | portfolio diffs                       |                                        | limits_tuned_summary.json,                       |
|     |                                       |                                       |                                        | new_batch_limits_compare.json                    |
|     |                                       |                                       |                                        |                                                  |
| 13  | crisk_fairness_report.py              | Monitor parity: approval/decline &    | final_policy_decisions.csv (or         | Dataset/crisk_policy/fairness_report/{by_sex.csv,|                                 |     |                                       | bad-rate by sex and age bands;        | batch decisions)                       | by_age_band.csv, fairness_summary.json}          |   
|     |                                       | proxy EL                              |                                        |                                                  |                                 |     |                                       |                                       |                                        |                                                  |
| 14  | crisk_service.py                      | Minimal FastAPI for online scoring    | crisk_model.pkl,                       | Running REST API: GET /health, POST /score       |
|     |                                       | & limit decisioning                   | final_policy_threshold.json            | (returns pd_month, pd_1y, decision, limit)       |
|     |                                       |                                       |                                        |                                                  |
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Notes
• All scripts are written into Dataset/crisk_* folders and are safe to re-run; artefacts are overwritten deterministically.
• Policy/limit assumptions: LGD=0.85, UTIL=0.40, EL-cap defaults documented in each script; risk-band cuts: A(≤2%), B(2–5%), C(5–10%), D(>10%).
• If you need to change approval targets, bad-rate caps, or EL budgets, edit the constants at the top of the respective scripts (Steps 4–8 & 12).

---

## Running the Scripts

```Python Code

A]--------> ## data_preparation.py  ##

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

B]--------> ## # # crisk_train.py ##

# crisk_train.py — Train CRISK model, report metrics, save artifacts

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn import __version__ as skl_version
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import joblib


# Paths
BASE = Path("Dataset")
IN = BASE / "crisk_clean"
OUT = BASE / "crisk_models"
OUT.mkdir(parents=True, exist_ok=True)


# Load data
train = pd.read_parquet(IN / "train.parquet")
valid = pd.read_parquet(IN / "valid.parquet")
test = pd.read_parquet(IN / "test.parquet")

y_tr = train["target"].astype(int).to_numpy()
y_va = valid["target"].astype(int).to_numpy()
y_te = test["target"].astype(int).to_numpy()

X_tr = train.drop(columns=["target"])
X_va = valid.drop(columns=["target"])
X_te = test.drop(columns=["target"])

# Feature groups
pay_cols = [f"pay_{i}" for i in [1, 2, 3, 4, 5, 6]]
bill_cols = [f"bill_amt{i}" for i in [1, 2, 3, 4, 5, 6]]
payamt_cols = [f"pay_amt{i}" for i in [1, 2, 3, 4, 5, 6]]

num_cols = bill_cols + payamt_cols + ["age", "limit_bal"]
ord_cat_cols = pay_cols + ["sex", "education", "marriage"]

# Preprocessor (robust to sklearn version)
ver_major, ver_minor = map(int, skl_version.split(".")[:2])
ohe_kwargs = {"handle_unknown": "ignore"}
if (ver_major, ver_minor) >= (1, 2):
    ohe_kwargs["sparse_output"] = False
else:
    ohe_kwargs["sparse"] = False
ohe = OneHotEncoder(**ohe_kwargs)

pre = ColumnTransformer(
    transformers=[
        ("cat", ohe, ord_cat_cols),
        ("num", RobustScaler(), num_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

# Model
clf = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_depth=7,
    max_leaf_nodes=None,
    min_samples_leaf=25,
    l2_regularization=0.01,
    early_stopping=True,
    random_state=42,
)

pipe = Pipeline([
    ("pre", pre),
    ("clf", clf),
])


# Train
pipe.fit(X_tr, y_tr)

# Helpers to save curves robustly
def _pad_to_len(a: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(a).ravel()
    if len(a) >= n:
        return a[:n]
    out = np.empty(n, dtype=float)
    out[:len(a)] = a
    out[len(a):] = np.nan
    return out

def safe_save_roc_pr(y_true, proba, name: str):
    fpr, tpr, thr = roc_curve(y_true, proba)
    n = max(len(fpr), len(tpr), len(thr) + 1)  # thresholds is one shorter
    roc_df = pd.DataFrame({
        "fpr": _pad_to_len(fpr, n),
        "tpr": _pad_to_len(tpr, n),
        "thr": _pad_to_len(thr, n),  # padded with NaN on the last row
    })
    roc_df.to_csv(OUT / f"roc_{name}.csv", index=False)

    pr, rc, _ = precision_recall_curve(y_true, proba)
    pd.DataFrame({"precision": pr, "recall": rc}).to_csv(OUT / f"pr_{name}.csv", index=False)


# Evaluate
def eval_split(X, y, name: str) -> dict:
    proba = pipe.predict_proba(X)[:, 1]
    pred50 = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y, proba)
    ap = average_precision_score(y, proba)
    brier = brier_score_loss(y, proba)

    fpr, tpr, thr = roc_curve(y, proba)
    j = tpr - fpr
    j_idx = int(np.argmax(j))
    thr_star = float(thr[j_idx]) if j_idx < len(thr) else 0.5
    pred_star = (proba >= thr_star).astype(int)

    rep50 = classification_report(y, pred50, output_dict=True, zero_division=0)
    rep_star = classification_report(y, pred_star, output_dict=True, zero_division=0)
    cm50 = confusion_matrix(y, pred50).tolist()
    cm_star = confusion_matrix(y, pred_star).tolist()

    safe_save_roc_pr(y, proba, name)

    return {
        "split": name,
        "n": int(len(y)),
        "pos_rate": float(np.mean(y)),
        "AUC": float(auc),
        "AP": float(ap),
        "Brier": float(brier),
        "threshold_star": thr_star,
        "report_at_0.50": rep50,
        "report_at_star": rep_star,
        "confusion_at_0.50": cm50,
        "confusion_at_star": cm_star,
    }

metrics = {
    "valid": eval_split(X_va, y_va, "valid"),
    "test":  eval_split(X_te, y_te, "test"),
}

# Permutation importance on TRANSFORMED X and FINAL ESTIMATOR
# (fixes the length mismatch)
Xva_tr = pipe.named_steps["pre"].transform(X_va)                   # transformed matrix
feat_names = pipe.named_steps["pre"].get_feature_names_out()       # expanded names match columns
perm = permutation_importance(
    estimator=pipe.named_steps["clf"],                              # final classifier only
    X=Xva_tr,
    y=y_va,
    n_repeats=5,
    random_state=42,
    scoring="roc_auc",
)
imp = (pd.DataFrame({
    "feature": feat_names,
    "importance_mean": perm.importances_mean,
    "importance_std":  perm.importances_std,
})
       .sort_values("importance_mean", ascending=False))
imp.to_csv(OUT / "feature_importance_validation.csv", index=False)


# Persist model + metrics
joblib.dump(pipe, OUT / "crisk_model.pkl")
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

print("Saved model to:", OUT / "crisk_model.pkl")
print("Validation AUC:", round(metrics["valid"]["AUC"], 4),
      "| Test AUC:", round(metrics["test"]["AUC"], 4))
print("Validation threshold* (Youden J):", round(metrics["valid"]["threshold_star"], 4))
print("Top 10 features by permutation importance:")
print(imp.head(10))

C]--------> ## # # crisk_score_readout.py ##

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

D]--------> ## # # crisk_pick_thresholds.py ##

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


E]--------> ## # # crisk_new_batch_rethreshold.py ##

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

F]--------> ## # # crisk_limit_strategy.py ##

# crisk_limit_strategy.py — turn decisions into approve limits + portfolio view

import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("Dataset")
READOUT = BASE / "crisk_readout"
CLEAN   = BASE / "crisk_clean"
OUT     = BASE / "crisk_policy"
OUT.mkdir(parents=True, exist_ok=True)

# Load decisions + scored cohort (for features like limit_bal, age, sex)
dec = pd.read_csv(READOUT / "decisions_1y.csv")           # decision + thresholds + score/pd_month/target
scored = pd.read_csv(READOUT / "scored_cohort_1y.csv")

# Join if we have a key; else align by row order
key = next((c for c in ["applicant_id","customer_id","id"] if c in dec.columns and c in scored.columns), None)
if key:
    df = dec.merge(scored, on=key, how="left", suffixes=("","_sc"))
else:
    df = dec.copy()
    for c in scored.columns:
        if c not in df.columns:
            df[c] = scored[c].values[:len(df)]

# Choose score column
score_col = "pd_month" if "pd_month" in df.columns else ("score" if "score" in df.columns else None)
if score_col is None:
    raise RuntimeError("Need a score column (pd_month or score).")

# Ensure 1y PD
if "pd_1y" not in df.columns:
    pd_m = pd.to_numeric(df[score_col], errors="coerce").clip(0, 1).fillna(0.0)
    df["pd_1y"] = 1.0 - (1.0 - pd_m) ** 12
df["pd_1y"] = df["pd_1y"].clip(0, 1)

# Base limit if missing
if "limit_bal" not in df.columns:
    df["limit_bal"] = 50000

# Risk bands by PD
bins = [-np.inf, 0.02, 0.05, 0.10, np.inf]
labels = ["A (<=2%)","B (2–5%)","C (5–10%)","D (>10%)"]
df["risk_band"] = pd.cut(df["pd_1y"], bins=bins, labels=labels)

# EL-capped limit policy
LGD = 0.85
UTIL = 0.40
EL_BUDGET = 0.03

mult_by_band = {
    "A (<=2%)": 1.20,
    "B (2–5%)": 1.00,
    "C (5–10%)": 0.80,
    "D (>10%)": 0.50,
}

# FIX: make multiplier a float series (not categorical) before fillna
mult = df["risk_band"].astype(str).map(mult_by_band)      # object floats or NaN
mult = pd.to_numeric(mult, errors="coerce").fillna(0.7)   # <- safe fill

df["baseline_limit"] = df["limit_bal"] * mult

# EL cap
cap = (EL_BUDGET / (df["pd_1y"] * LGD * UTIL)).replace([np.inf, -np.inf], np.nan)
df["el_cap_multiplier"] = cap.clip(upper=1.0)
df["limit_rec"] = (df["baseline_limit"] * df["el_cap_multiplier"]).fillna(df["baseline_limit"])

# Practical guards
df["limit_rec"] = df["limit_rec"].clip(lower=1000, upper=200000)

# Apply decisions: declines -> 0 limit
df["limit_final"] = np.where(df["decision"].astype(str).str.lower().eq("decline"), 0, df["limit_rec"])

# Portfolio KPIs
approved = df[df["limit_final"] > 0].copy()
n_apps = len(df); n_approved = len(approved)
approval_rate = n_approved / max(1, n_apps)

approved["EL"] = approved["pd_1y"] * LGD * UTIL * approved["limit_final"]
portfolio = {
    "n_applications": int(n_apps),
    "n_approved": int(n_approved),
    "approval_rate": float(approval_rate),
    "avg_limit": float(approved["limit_final"].mean() if n_approved else 0),
    "sum_limit": float(approved["limit_final"].sum() if n_approved else 0),
    "expected_loss_total": float(approved["EL"].sum() if n_approved else 0),
    "expected_loss_rate": float(approved["EL"].sum() / max(1.0, approved["limit_final"].sum())) if n_approved else 0.0,
}

# Rollups
by_band = (approved.groupby("risk_band", observed=True)
           .agg(apps=("pd_1y","size"),
                avg_pd=("pd_1y","mean"),
                avg_limit=("limit_final","mean"),
                sum_limit=("limit_final","sum"),
                exp_loss=("EL","sum"))
           .reset_index())

# Save
OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / "decisions_with_limits.csv", index=False)
by_band.to_csv(OUT / "portfolio_by_risk_band.csv", index=False)
pd.Series(portfolio).to_json(OUT / "portfolio_summary.json", indent=2)

print("✓ Limits assigned and policy pack saved to:", OUT.resolve())
print("Portfolio:", portfolio)
print(by_band)

G]--------> ## # # crisk_policy_sweep.py ##

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

H]--------> ## # # crisk_policy_frontier.py ##

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

I]--------> ## # # crisk_policy_finalize_from_cap.py ##

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

J]--------> ## # # crisk_apply.py ##

# crisk_apply.py — score new applications, make approve/review/decline + limits

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# ---------- paths ----------
BASE   = Path("Dataset")
CLEAN  = BASE / "crisk_clean"
MODELS = BASE / "crisk_models"
POLICY = BASE / "crisk_policy"
READOUT= BASE / "crisk_readout"
POLICY.mkdir(parents=True, exist_ok=True)

MODEL_PATH   = MODELS / "crisk_model.pkl"
THR_PATH     = POLICY / "final_policy_threshold.json"   # optional (from your frontier/sweep)
PORTF_PATH   = POLICY / "final_policy_portfolio.json"   # optional (contains approval_rate)
DEFAULT_DEMO = BASE / "new_apps_demo.csv"               # created on first run if needed

# ---------- helpers ----------
REQ_COLS = (
    [f"pay_{i}" for i in [1,2,3,4,5,6]] +
    [f"bill_amt{i}" for i in [1,2,3,4,5,6]] +
    [f"pay_amt{i}"  for i in [1,2,3,4,5,6]] +
    ["sex","education","marriage","age","limit_bal"]
)

def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model not found at: {path}")
    return joblib.load(path)

def read_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suf in (".xlsx", ".xls"):
        # header=1 for UCI original excel; for generic xlsx keep header=0
        try:
            return pd.read_excel(path, engine="openpyxl")
        except Exception:
            return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suf}")

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing training columns with safe defaults."""
    out = df.copy()
    defaults = {
        # categorical-ish / ordinal
        "sex": 2,          # 1/2 in UCI; using 2 as default
        "education": 2,    # 1..4 common; 2 ~ university
        "marriage": 2,     # 1..3 common; 2 ~ single
        # numeric
        "age": 35,
        "limit_bal": 50000,
        **{f"pay_{i}": 0 for i in [1,2,3,4,5,6]},
        **{f"bill_amt{i}": 0.0 for i in [1,2,3,4,5,6]},
        **{f"pay_amt{i}":  0.0 for i in [1,2,3,4,5,6]},
    }
    for c in REQ_COLS:
        if c not in out.columns:
            out[c] = defaults[c]
    # order columns roughly like training (not strictly required, but neat)
    return out[[*(c for c in REQ_COLS), *(c for c in out.columns if c not in REQ_COLS)]]

def load_policy_threshold() -> dict:
    """Try to load a saved threshold. Return dict with keys:
       - 'thr_pd_m' (monthly PD threshold for approval) OR None
       - 'approval_rate' (float 0..1) OR None
       - 'review_band' (float, width around threshold) default 0.10
    """
    out = {"thr_pd_m": None, "approval_rate": None, "review_band": 0.10}
    try:
        if THR_PATH.exists():
            with open(THR_PATH, "r") as f:
                d = json.load(f)
            # accept several key names
            out["thr_pd_m"] = (
                d.get("thr_pd_m") or d.get("thr_approve") or d.get("threshold") or d.get("threshold_approve")
            )
        if out["approval_rate"] is None and PORTF_PATH.exists():
            with open(PORTF_PATH, "r") as f:
                p = json.load(f)
            out["approval_rate"] = p.get("approval_rate")
    except Exception:
        pass
    return out

def make_demo_if_needed() -> Path:
    if DEFAULT_DEMO.exists():
        return DEFAULT_DEMO
    src = CLEAN / "test.parquet"
    if not src.exists():
        raise FileNotFoundError(f"Cannot create demo — missing {src}")
    df = pd.read_parquet(src)
    df = df.drop(columns=["target"], errors="ignore")
    df.head(1000).to_csv(DEFAULT_DEMO, index=False)
    print(f"(No --input provided) Created demo at: {DEFAULT_DEMO}")
    return DEFAULT_DEMO

# ---- limit/portfolio helpers (same logic as your limit strategy) ----
LGD, UTIL = 0.85, 0.40
EL_BUDGET = 0.03  # EL as share of limit tolerated for EL-cap
MULT_BY_BAND = {
    "A (<=2%)": 1.20,
    "B (2–5%)": 1.00,
    "C (5–10%)": 0.80,
    "D (>10%)": 0.50,
}
BINS   = [-np.inf, 0.02, 0.05, 0.10, np.inf]
LABELS = ["A (<=2%)","B (2–5%)","C (5–10%)","D (>10%)"]

def assign_limits(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # 1y PD from monthly PD
    pd_m = pd.to_numeric(out["pd_month"], errors="coerce").clip(0,1).fillna(0.0)
    out["pd_1y"] = 1.0 - (1.0 - pd_m)**12
    out["pd_1y"] = out["pd_1y"].clip(0,1)
    # bands
    out["risk_band"] = pd.cut(out["pd_1y"], bins=BINS, labels=LABELS)
    mult = pd.to_numeric(out["risk_band"].astype(str).map(MULT_BY_BAND), errors="coerce").fillna(0.7)
    # baseline*mult
    if "limit_bal" not in out.columns:
        out["limit_bal"] = 50_000
    out["baseline_limit"] = out["limit_bal"] * mult
    # EL cap
    cap_mult = (EL_BUDGET / (out["pd_1y"] * LGD * UTIL)).replace([np.inf,-np.inf], np.nan).clip(upper=1.0)
    out["el_cap_multiplier"] = cap_mult
    out["limit_rec"] = (out["baseline_limit"] * out["el_cap_multiplier"]).fillna(out["baseline_limit"])
    out["limit_rec"] = out["limit_rec"].clip(lower=1_000, upper=200_000)
    # final: 0 for Decline, keep for Approve, 0 (or keep) for Review — here we set 0 to force manual review
    out["limit_final"] = np.where(out["decision"].str.lower().eq("decline"), 0,
                           np.where(out["decision"].str.lower().eq("review"), 0, out["limit_rec"]))
    return out

# ---------- main ----------
def main(input_path: Path | None = None):
    # 0) resolve input
    if input_path is None:
        input_path = make_demo_if_needed()
    df_raw = read_any(input_path)
    df = ensure_columns(df_raw)

    # 1) load model & score
    pipe = load_model(MODEL_PATH)
    pd_month = pipe.predict_proba(df[REQ_COLS])[:, 1]  # probability of default (= monthly PD proxy)
    scored = df.copy()
    scored["pd_month"] = pd_month

    # 2) load policy threshold (if you saved one); else derive from approval rate; else default to 5%
    pol = load_policy_threshold()
    scores = scored["pd_month"]
    thr = pol["thr_pd_m"]
    approval_rate = pol["approval_rate"]

    if thr is None:
        if approval_rate is None:
            approval_rate = 0.05  # fallback to your Option A outcome
        # lowest PD are safest => approve the lowest 'approval_rate' portion
        thr = float(scores.quantile(approval_rate))  # approve scores <= thr
    # define a review band ~10% of population above the approval threshold (cap within [0,1])
    band = float(pol.get("review_band", 0.10))
    q_low  = thr
    q_high = float(scores.quantile(min(approval_rate + band, 0.99)))

    # 3) make decisions
    decision = np.where(scores <= q_low, "Approve",
                 np.where(scores <= q_high, "Review", "Decline"))
    decided = scored.assign(decision=decision, thr_approve=thr, review_upper=q_high)

    # 4) limits for approvals (and zero for declines & reviews)
    decided = assign_limits(decided)

    # 5) outputs
    out_csv = POLICY / "new_batch_decisions.csv"
    out_json= POLICY / "new_batch_summary.json"
    decided.to_csv(out_csv, index=False)

    summary = {
        "n_applications": int(len(decided)),
        "approval_rate": float((decided["decision"] == "Approve").mean()),
        "review_rate":   float((decided["decision"] == "Review").mean()),
        "decline_rate":  float((decided["decision"] == "Decline").mean()),
        "thr_pd_month":  float(thr),
        "review_upper":  float(q_high),
        "avg_limit_approved": float(decided.loc[decided["decision"]=="Approve","limit_final"].mean()),
        "sum_limit_approved": float(decided.loc[decided["decision"]=="Approve","limit_final"].sum()),
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("✓ Scored and decided.")
    print("  Decisions ->", out_csv)
    print("  Summary   ->", out_json)
    print(summary)

# ... keep all your existing imports, helpers, and main() exactly as you have ...

if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser(
        description="CRISK: apply model & policy to new applications",
        add_help=True
    )
    ap.add_argument(
        "--input",
        help="Path to CSV/Parquet/XLSX with new applications (if omitted, a demo is created)."
    )
    # <-- This is the key line: ignore any extra args that PyCharm/pydevconsole adds
    args, unknown = ap.parse_known_args()
    if unknown:
        # Optional: print once so you know they're being ignored
        print(f"(Note) Ignoring unknown args: {unknown}")
    in_path = Path(args.input) if args.input else None
    # If you launched from the Python Console, just call main() with no args or your path
    main(in_path)

K]--------> ## # # crisk_new_batch_report.py ##

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

L]--------> ## # # crisk_new_batch_apply_limits_and_compare.py ##

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

M]--------> ## # # crisk_fairness_report.py ##

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

N]--------> ## # # crisk_service.py ##

# crisk_service.py — minimal REST API for CRISK (Pydantic v2 / FastAPI)
from pathlib import Path
from typing import List, Optional
import json

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# --- paths (adjust if your repo layout is different)
BASE = Path("Dataset")
MODEL_PATH   = BASE / "crisk_models" / "crisk_model.pkl"
POLICY_PATH  = BASE / "crisk_policy" / "final_policy_threshold.json"   # created earlier
READOUT_PATH = BASE / "crisk_readout"

# --- load model + policy
pipe = joblib.load(MODEL_PATH)
try:
    policy = json.loads(POLICY_PATH.read_text())
    THR = float(policy.get("thr_pd_month", 0.05))          # monthly PD threshold chosen
    REVIEW_UP = float(policy.get("review_upper", THR*1.45))  # optional review window
except Exception:
    THR, REVIEW_UP = 0.05, 0.07

LGD, UTIL = 0.85, 0.40
BAND_MULT = {"A (<=2%)": 1.20, "B (2–5%)": 1.00, "C (5–10%)": 0.80, "D (>10%)": 0.50}

# --- request/response schemas
FEATURE_ORDER = [
    "limit_bal","sex","education","marriage","age",
    "pay_1","pay_2","pay_3","pay_4","pay_5","pay_6",
    "bill_amt1","bill_amt2","bill_amt3","bill_amt4","bill_amt5","bill_amt6",
    "pay_amt1","pay_amt2","pay_amt3","pay_amt4","pay_amt5","pay_amt6",
]

class Applicant(BaseModel):
    # keep names consistent with your training columns
    limit_bal: float
    sex: int
    education: int
    marriage: int
    age: int
    pay_1: int
    pay_2: int
    pay_3: int
    pay_4: int
    pay_5: int
    pay_6: int
    bill_amt1: float
    bill_amt2: float
    bill_amt3: float
    bill_amt4: float
    bill_amt5: float
    bill_amt6: float
    pay_amt1: float
    pay_amt2: float
    pay_amt3: float
    pay_amt4: float
    pay_amt5: float
    pay_amt6: float
    applicant_id: Optional[str] = None

class ScoreRequest(BaseModel):
    # Pydantic v2: use Field(min_length=1) instead of conlist(..., min_items=1)
    applicants: List[Applicant] = Field(min_length=1)

class ScoreResponse(BaseModel):
    applicant_id: Optional[str] = None
    pd_month: float
    pd_1y: float
    decision: str
    limit: float

app = FastAPI(title="CRISK API", version="1.0")

def _risk_band(pd_1y: float) -> str:
    if pd_1y <= 0.02: return "A (<=2%)"
    if pd_1y <= 0.05: return "B (2–5%)"
    if pd_1y <= 0.10: return "C (5–10%)"
    return "D (>10%)"

def _limit_with_policy(row: pd.Series) -> float:
    # baseline by band
    band = _risk_band(row["pd_1y"])
    baseline = row["limit_bal"] * BAND_MULT.get(band, 0.7)
    # EL cap
    cap_mult = (0.03 / (row["pd_1y"] * LGD * UTIL)) if row["pd_1y"] > 0 else 1.0
    cap_mult = min(max(cap_mult, 0.0), 1.0)
    limit = baseline * cap_mult
    return float(np.clip(limit, 1000, 200000))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists(), "threshold": THR}

@app.post("/score", response_model=List[ScoreResponse])
def score(req: ScoreRequest):
    # to DataFrame in the exact feature order
    df = pd.DataFrame([a.model_dump() for a in req.applicants])
    X = df[FEATURE_ORDER].copy()

    # monthly PD from model
    pd_month = pipe.predict_proba(X)[:, 1]
    pd_1y = 1 - (1 - pd_month) ** 12

    # decisions
    decisions = np.where(pd_month <= THR, "Approve",
                 np.where(pd_month <= REVIEW_UP, "Review", "Decline"))

    # limits (0 for Decline)
    out: List[ScoreResponse] = []
    for i, row in df.iterrows():
        pd1y = float(pd_1y[i])
        dec  = str(decisions[i])
        lim  = _limit_with_policy(pd.Series({**row.to_dict(), "pd_1y": pd1y})) if dec != "Decline" else 0.0
        out.append(ScoreResponse(
            applicant_id=row.get("applicant_id"),
            pd_month=float(pd_month[i]),
            pd_1y=pd1y,
            decision=dec,
            limit=lim
        ))
    return out

```

## Project Structure

SERAH-CRISK/
├─ Dataset/
│  ├─ crisk_clean
│  ├─ crisk_models
│  ├─ crisk_policy
│  ├─ crisk_readout
│  ├─ taiwan_credit
│  ├─ new_apps_demo.csv
│  ├─ UCI_Credit_Card.csv
├─ Scripts/
│  ├─ data_preparation.py
│  ├─ crisk_train.py
│  ├─ crisk_score_readout.py
│  ├─ crisk_pick_thresholds.py
│  ├─ crisk_new_batch_rethreshold.py
│  ├─ crisk_limit_strategy.py
│  ├─ crisk_policy_sweep.py
│  ├─ crisk_policy_frontier.py
│  ├─ crisk_policy_finalize_from_cap.py
│  ├─ crisk_apply.py
│  ├─ crisk_new_batch_report.py
│  ├─ crisk_new_batch_apply_limits_and_compare.py
│  ├─ crisk_fairness_report.py
│  ├─ crisk_service.py
├─ README.md
├─ LICENSE


## License

MIT License

Copyright (c) 2025 rabraham2

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including, without limitation, the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


