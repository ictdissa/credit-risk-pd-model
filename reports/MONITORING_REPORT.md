# Monitoring Report — PD Model

## What we monitor
### 1) Input / Population Drift (proxy)
- **Predicted PD PSI** between baseline (base-train) and a later sample (test).
- Current PSI: **0.000223**

Typical PSI interpretation (rule-of-thumb):
- < 0.10: minor/no drift
- 0.10–0.25: moderate drift (investigate)
- > 0.25: major drift (action needed)

### 2) Performance Drift (when labels arrive)
When new defaults are observed, monitor:
- AUC/PR-AUC/KS on recent cohorts
- Calibration drift (Brier / calibration curve)

## Alerting & Actions
- PSI > 0.10: inspect feature distribution shifts, segment changes, policy change effects.
- PSI > 0.25: consider retraining on recent data and re-validating calibration.
- If calibration drift is observed: re-calibrate or retrain + re-calibrate.

## Retraining policy (portfolio project)
- Retrain schedule: quarterly or when PSI is high for 2 consecutive periods.
- Recalibration: whenever risk grade default rates deviate materially from expected.

## Notes
This is a portfolio-grade monitoring plan; real institutions add governance, approvals, challenger models, and documented sign-offs.
