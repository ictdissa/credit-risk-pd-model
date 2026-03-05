# Model Card — PD Model (Probability of Default)

## Overview
This project trains a Probability of Default (PD) model for consumer/retail loans and demonstrates:
- Model training + evaluation (AUC, PR-AUC, KS, Brier)
- Probability calibration (sigmoid/isotonic)
- Risk-based decisioning (profit/EL-based threshold)
- Monitoring baseline (PSI drift for predicted PD)

## Intended Use
- Rank-order loan applications by default risk (PD).
- Produce calibrated PD estimates suitable for policy simulation and risk banding.

## Not Intended Use
- This is a portfolio project; economics are simplified and do not fully model amortization timing, prepayments, or default timing.
- Do not deploy in production without compliance, governance, and rigorous validation.

## Data & Label
- Data source: `data/loans_clean.csv`
- Target: `Default` (assumed binary 0/1)

## Splitting Protocol
- Split type: `random_stratified`
- Details: {
  "split_type": "random_stratified",
  "note": "No date_col provided or found; using stratified random split."
}

## Leakage Prevention
Columns excluded from PD features:
- Explicit excludes: ['InterestRate', 'LoanID']
- Keyword-based suspected leakage: ['MaritalStatus']
- User-provided extra drops: []

Rationale: post-origination or outcome-related fields (payments, delinquency after origination, status, recoveries) can inflate performance and invalidate results.

## Modeling
- Candidates: Logistic Regression (interpretable baseline), HistGradientBoosting (performance model)
- Selection metric on policy-validation (uncalibrated): pr_auc_on_policy_validation_uncalibrated
- Chosen base model: hgb

## Calibration
- Method: `sigmoid`
- Evaluation includes before/after calibration overlay plots and Brier score changes.

## Final Test Performance (Calibrated)
{
  "roc_auc": 0.7297306599089347,
  "pr_auc": 0.2918666001605721,
  "brier": 0.09358245462304639,
  "ks": 0.3416685041315029
}

## Monitoring
- PSI drift computed for predicted PD between base-train and test.
- Drift response plan is described in MONITORING_REPORT.md
