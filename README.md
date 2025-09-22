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


