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

A]--------> ## data_cleaning.py  ##
# Data Analysis Summary





