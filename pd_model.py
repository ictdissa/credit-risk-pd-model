import os
import re
import json
import math
import joblib
import argparse
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve, roc_curve, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.inspection import permutation_importance


# ---------------------------
# Config
# ---------------------------

@dataclass
class Config:
    data_path: str = "data/loans_clean.csv"
    target_col: str = "Default"

    # Optional date column for OOT split
    date_col: Optional[str] = None
    parse_date_format: Optional[str] = None  # e.g. "%Y-%m-%d"

    random_state: int = 42

    # Split sizes
    test_size: float = 0.2              # OOT test fraction if date_col provided
    policy_valid_size: float = 0.2      # threshold tuning only
    calibration_size: float = 0.2       # calibration only (within train)

    # Economics (optional)
    ead_col: Optional[str] = "LoanAmount"
    lgd_col: Optional[str] = "lgd"
    ead_default: float = 10000.0
    lgd_default: float = 0.6
    downturn_lgd_multiplier: float = 1.0

    apr_col_candidates: Tuple[str, ...] = ("InterestRate", "interest_rate", "apr", "int_rate", "rate")
    term_col_candidates: Tuple[str, ...] = ("LoanTerm", "term_months", "term", "loan_term")
    funding_rate_annual: float = 0.04
    orig_fee_rate: float = 0.02
    ops_cost_per_loan: float = 50.0
    benefit_default: float = 300.0

    apr_min: float = 0.0
    apr_max: float = 0.60
    term_months_min: float = 1.0
    term_months_max: float = 120.0

    # Explicit exclusion list (always dropped from PD features if present)
    exclude_from_pd_features: Tuple[str, ...] = ("InterestRate", "LoanID")

    # Leakage protection
    leakage_keyword_patterns: Tuple[str, ...] = (
        r"(?i)last[_\s-]*pmt",
        r"(?i)next[_\s-]*pmt",
        r"(?i)pymnt|payment|paid",
        r"(?i)collection|recover|recovery",
        r"(?i)charge[_\s-]*off|charged[_\s-]*off",
        r"(?i)delinq|dpd|days[_\s-]*past[_\s-]*due",
        r"(?i)settlement|hardship|bankrupt",
        r"(?i)outstanding[_\s-]*principal|principal[_\s-]*outstanding",
        r"(?i)months[_\s-]*since",
        r"(?i)status|loan[_\s-]*status",
    )
    # Optional user-provided extra drops (comma-separated via CLI)
    extra_drop_cols: Tuple[str, ...] = ()

    calibration_method: str = "sigmoid"   # "sigmoid" or "isotonic"

    # Policy objective
    decision_objective: str = "max_profit"  # "max_profit" or "profit_with_el_cap"
    el_cap: float = 5e6

    n_risk_bands: int = 10

    # Output
    model_dir: str = "models"
    report_dir: str = "reports"


# ---------------------------
# Calibrator (clean version)
# ---------------------------

class ManualCalibratedModel(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible calibrated classifier wrapper.

    base_model must implement predict_proba.
    method:
      - 'sigmoid': Platt scaling (logistic regression on logit(p))
      - 'isotonic': Isotonic regression on p
    """
    _estimator_type = "classifier"

    def __init__(self, base_model, method: str = "sigmoid"):
        self.base_model = base_model
        self.method = method
        self.calibrator_ = None
        self.classes_ = np.array([0, 1], dtype=int)

    @staticmethod
    def _clip_probs(p: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)

    def fit(self, X, y, sample_weight=None):
        p = self._clip_probs(self.base_model.predict_proba(X)[:, 1])

        if self.method == "sigmoid":
            logit = np.log(p / (1 - p)).reshape(-1, 1)
            lr = LogisticRegression(max_iter=2000, solver="lbfgs")
            if sample_weight is None:
                lr.fit(logit, y)
            else:
                lr.fit(logit, y, sample_weight=sample_weight)
            self.calibrator_ = ("sigmoid", lr)

        elif self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            if sample_weight is None:
                iso.fit(p, y)
            else:
                iso.fit(p, y, sample_weight=sample_weight)
            self.calibrator_ = ("isotonic", iso)

        else:
            raise ValueError("method must be 'sigmoid' or 'isotonic'")

        return self

    def predict_proba(self, X):
        if self.calibrator_ is None:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        p = self._clip_probs(self.base_model.predict_proba(X)[:, 1])
        kind, cal = self.calibrator_

        if kind == "sigmoid":
            logit = np.log(p / (1 - p)).reshape(-1, 1)
            p_cal = cal.predict_proba(logit)[:, 1]
        else:
            p_cal = cal.predict(p)

        p_cal = self._clip_probs(p_cal)
        return np.column_stack([1 - p_cal, p_cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def __sklearn_tags__(self):
        return {"estimator_type": "classifier"}


# ---------------------------
# Helpers
# ---------------------------

def ensure_dirs(cfg: Config):
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.report_dir, exist_ok=True)


def load_data(cfg: Config) -> pd.DataFrame:
    if not os.path.exists(cfg.data_path):
        raise FileNotFoundError(f"Missing {cfg.data_path}. Put your CSV there or change --data_path.")
    df = pd.read_csv(cfg.data_path)
    if cfg.target_col not in df.columns:
        raise ValueError(f"Target '{cfg.target_col}' not found in dataset.")
    return df


def pick_first_existing(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_rate_series(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        cleaned = s.astype(str).str.replace("%", "", regex=False).str.strip()
        r = pd.to_numeric(cleaned, errors="coerce")
        r = np.where(r > 1.5, r / 100.0, r)
        return pd.Series(r, index=s.index)
    r = pd.to_numeric(s, errors="coerce")
    r = np.where(r > 1.5, r / 100.0, r)
    return pd.Series(r, index=s.index)


def parse_term_months(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        cleaned = s.astype(str).str.extract(r"(\d+)", expand=False)
        t = pd.to_numeric(cleaned, errors="coerce")
        return pd.Series(t, index=s.index)
    return pd.to_numeric(s, errors="coerce")


def infer_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ])


def prob_metrics(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
        "ks": float(ks_statistic(y_true, p)),
    }


def ks_statistic(y_true: np.ndarray, p: np.ndarray) -> float:
    """
    KS = max_t |TPR(t) - FPR(t)|
    """
    fpr, tpr, _ = roc_curve(y_true, p)
    return float(np.max(np.abs(tpr - fpr)))


def get_vector(df: pd.DataFrame, col: Optional[str], fallback: float) -> np.ndarray:
    if col and col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(fallback).astype(float).values
    return np.full(len(df), fallback, dtype=float)


def get_ead_lgd(cfg: Config, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    ead = get_vector(df, cfg.ead_col, cfg.ead_default)
    lgd = get_vector(df, cfg.lgd_col, cfg.lgd_default) * cfg.downturn_lgd_multiplier
    ead = np.where(ead > 0, ead, cfg.ead_default)
    lgd = np.clip(lgd, 0.0, 1.0)
    return ead, lgd


def compute_benefit(cfg: Config, df: pd.DataFrame, ead: np.ndarray) -> Tuple[np.ndarray, Dict[str, str]]:
    meta = {"benefit_source": "fallback_default", "apr_col": "", "term_col": ""}

    apr_col = pick_first_existing(df, cfg.apr_col_candidates)
    term_col = pick_first_existing(df, cfg.term_col_candidates)
    if apr_col is None or term_col is None:
        return np.full(len(df), cfg.benefit_default, dtype=float), meta

    apr = parse_rate_series(df[apr_col]).astype(float)
    term_m = parse_term_months(df[term_col]).astype(float)

    if np.nanmean(np.isfinite(apr)) < 0.5 or np.nanmean(np.isfinite(term_m)) < 0.5:
        return np.full(len(df), cfg.benefit_default, dtype=float), meta

    apr = pd.Series(apr, index=df.index).clip(cfg.apr_min, cfg.apr_max)
    term_m = pd.Series(term_m, index=df.index).clip(cfg.term_months_min, cfg.term_months_max)

    term_years = (term_m / 12.0).astype(float)
    margin = (apr - cfg.funding_rate_annual).astype(float)

    orig_fee = cfg.orig_fee_rate * ead
    ops = np.full(len(df), cfg.ops_cost_per_loan, dtype=float)

    # Simple unit economics (documented in model card)
    benefit = ead * margin.values * term_years.values + orig_fee - ops
    benefit = np.where(np.isfinite(benefit), benefit, cfg.benefit_default)

    meta = {"benefit_source": "apr_term_simple", "apr_col": apr_col, "term_col": term_col}
    return benefit.astype(float), meta


def expected_profit(pd_prob: np.ndarray, ead: np.ndarray, lgd: np.ndarray, benefit: np.ndarray) -> np.ndarray:
    return (1.0 - pd_prob) * benefit - pd_prob * lgd * ead


def choose_threshold(cfg: Config, pd_prob: np.ndarray, ead: np.ndarray, lgd: np.ndarray, benefit: np.ndarray) -> Tuple[float, Dict[str, float]]:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_score, best_info = 0.5, -1e18, {}

    for t in thresholds:
        approve = pd_prob < t
        approval_rate = float(approve.mean())
        el_total = float((pd_prob[approve] * lgd[approve] * ead[approve]).sum())
        prof_total = float(expected_profit(pd_prob[approve], ead[approve], lgd[approve], benefit[approve]).sum())

        if cfg.decision_objective == "max_profit":
            score = prof_total
        else:
            score = approval_rate * 1e12 + prof_total if el_total <= cfg.el_cap else -1e18

        if score > best_score:
            best_score = score
            best_t = float(t)
            best_info = {
                "approval_rate": approval_rate,
                "expected_loss_total": el_total,
                "expected_profit_total": prof_total,
                "objective": cfg.decision_objective,
                "el_cap": float(cfg.el_cap) if cfg.decision_objective != "max_profit" else None,
            }
    return best_t, best_info


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    e = np.histogram(expected, bins=breakpoints)[0] / max(len(expected), 1)
    a = np.histogram(actual, bins=breakpoints)[0] / max(len(actual), 1)

    eps = 1e-8
    e = np.clip(e, eps, 1.0)
    a = np.clip(a, eps, 1.0)
    return float(np.sum((a - e) * np.log(a / e)))


def confusion_at_threshold(y_true: np.ndarray, pd_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_hat = (pd_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "threshold": float(threshold),
        "tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp),
        "precision": float(precision),
        "recall": float(recall),
    }


def make_quantile_edges(p: np.ndarray, n_bands: int) -> np.ndarray:
    qs = np.linspace(0, 1, n_bands + 1)
    edges = np.quantile(p, qs)
    edges[0] = 0.0
    edges[-1] = 1.0
    edges = np.maximum.accumulate(edges)  # ensure non-decreasing
    return edges


def apply_bands(p: np.ndarray, edges: np.ndarray) -> pd.Series:
    labels = [f"Band_{i+1:02d}" for i in range(len(edges) - 1)]
    return pd.cut(p, bins=edges, labels=labels, include_lowest=True, duplicates="drop")


def lift_table(y_true: np.ndarray, p: np.ndarray, n_bands: int) -> pd.DataFrame:
    edges = make_quantile_edges(p, n_bands)
    bands = apply_bands(p, edges)
    df = pd.DataFrame({"band": bands.astype(str), "y": y_true.astype(int), "pd": p.astype(float)})
    out = (df.groupby("band", dropna=False)
             .agg(n=("y", "size"), default_rate=("y", "mean"), avg_pd=("pd", "mean"))
             .reset_index())
    # Sort worst->best (higher PD to lower PD) by avg_pd desc
    out = out.sort_values("avg_pd", ascending=False)
    out["cum_n"] = out["n"].cumsum()
    out["cum_default_rate"] = (out["default_rate"] * out["n"]).cumsum() / out["cum_n"]
    return out


def plot_roc_pr(y_true: np.ndarray, p: np.ndarray, outdir: str, prefix: str):
    fpr, tpr, _ = roc_curve(y_true, p)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({prefix})")
    plt.savefig(os.path.join(outdir, f"{prefix}_roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    prec, rec, _ = precision_recall_curve(y_true, p)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({prefix})")
    plt.savefig(os.path.join(outdir, f"{prefix}_pr_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_calibration_overlay(y_true: np.ndarray, p_uncal: np.ndarray, p_cal: np.ndarray, outdir: str, prefix: str):
    frac_u, mean_u = calibration_curve(y_true, p_uncal, n_bins=10, strategy="quantile")
    frac_c, mean_c = calibration_curve(y_true, p_cal, n_bins=10, strategy="quantile")

    plt.figure()
    plt.plot(mean_u, frac_u, marker="o", label="Uncalibrated")
    plt.plot(mean_c, frac_c, marker="o", label="Calibrated")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
    plt.xlabel("Mean Predicted PD")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Calibration Curve Overlay ({prefix})")
    plt.legend()
    plt.savefig(os.path.join(outdir, f"{prefix}_calibration_overlay.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_decision_curves(pd_prob: np.ndarray, ead: np.ndarray, lgd: np.ndarray, benefit: np.ndarray, outdir: str):
    thresholds = np.linspace(0.01, 0.99, 99)
    approvals, els, profits = [], [], []

    for t in thresholds:
        approve = pd_prob < t
        approvals.append(float(approve.mean()))
        els.append(float((pd_prob[approve] * lgd[approve] * ead[approve]).sum()))
        profits.append(float(expected_profit(pd_prob[approve], ead[approve], lgd[approve], benefit[approve]).sum()))

    plt.figure()
    plt.plot(thresholds, profits, label="Expected Profit (portfolio)")
    plt.plot(thresholds, els, label="Expected Loss (portfolio)")
    plt.xlabel("PD Threshold")
    plt.ylabel("Value")
    plt.title("Decision Curves vs Threshold (Policy Validation)")
    plt.legend()
    plt.savefig(os.path.join(outdir, "decision_curves_profit_el.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(thresholds, approvals)
    plt.xlabel("PD Threshold")
    plt.ylabel("Approval Rate")
    plt.title("Approval Rate vs Threshold (Policy Validation)")
    plt.savefig(os.path.join(outdir, "decision_curve_approval.png"), dpi=150, bbox_inches="tight")
    plt.close()


def leakage_drop_list(cfg: Config, df: pd.DataFrame) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Returns:
      drop_cols: columns to drop from PD features
      reasons: dict with reasons -> list of cols
    """
    reasons = {"explicit_exclude": [], "keyword_leakage": [], "user_extra_drop": []}
    drop_cols = set()

    # explicit exclude
    for c in cfg.exclude_from_pd_features:
        if c in df.columns:
            drop_cols.add(c)
            reasons["explicit_exclude"].append(c)

    # keyword-based leakage
    patterns = [re.compile(p) for p in cfg.leakage_keyword_patterns]
    for c in df.columns:
        for pat in patterns:
            if pat.search(c):
                drop_cols.add(c)
                reasons["keyword_leakage"].append(c)
                break

    # user extra drop
    for c in cfg.extra_drop_cols:
        if c in df.columns:
            drop_cols.add(c)
            reasons["user_extra_drop"].append(c)

    # de-dup lists
    for k in reasons:
        reasons[k] = sorted(list(set(reasons[k])))

    return sorted(list(drop_cols)), reasons


def split_time_oot(cfg: Config, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    OOT split: sort by date_col, take last test_size fraction as test.
    """
    meta = {"split_type": "time_oot", "date_col": cfg.date_col}

    tmp = df.copy()
    tmp[cfg.date_col] = pd.to_datetime(tmp[cfg.date_col], format=cfg.parse_date_format, errors="coerce")
    if tmp[cfg.date_col].isna().mean() > 0.2:
        raise ValueError(f"Too many unparsable dates in '{cfg.date_col}'. Fix parse_date_format or clean data.")

    tmp = tmp.sort_values(cfg.date_col)
    n = len(tmp)
    n_test = max(1, int(math.floor(cfg.test_size * n)))
    train_df = tmp.iloc[:-n_test].copy()
    test_df = tmp.iloc[-n_test:].copy()

    meta["train_date_min"] = str(train_df[cfg.date_col].min())
    meta["train_date_max"] = str(train_df[cfg.date_col].max())
    meta["test_date_min"] = str(test_df[cfg.date_col].min())
    meta["test_date_max"] = str(test_df[cfg.date_col].max())

    return train_df, test_df, meta


def split_random(cfg: Config, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    meta = {"split_type": "random_stratified", "note": "No date_col provided or found; using stratified random split."}
    train_df, test_df = train_test_split(
        df, test_size=cfg.test_size, random_state=cfg.random_state, stratify=df[cfg.target_col]
    )
    return train_df, test_df, meta


def write_model_card(cfg: Config, outdir: str, summary: dict):
    """
    Minimal but strong model card.
    """
    path = os.path.join(outdir, "MODEL_CARD.md")
    leak = summary.get("leakage", {})
    split = summary.get("split", {})
    cal = summary.get("calibration", {})
    base = summary.get("model_selection", {})
    metrics = summary.get("final_test", {}).get("metrics_calibrated", {})

    text = f"""# Model Card — PD Model (Probability of Default)

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
- Data source: `{cfg.data_path}`
- Target: `{cfg.target_col}` (assumed binary 0/1)

## Splitting Protocol
- Split type: `{split.get("split_type")}`
- Details: {json.dumps(split, indent=2)}

## Leakage Prevention
Columns excluded from PD features:
- Explicit excludes: {leak.get("explicit_exclude", [])}
- Keyword-based suspected leakage: {leak.get("keyword_leakage", [])}
- User-provided extra drops: {leak.get("user_extra_drop", [])}

Rationale: post-origination or outcome-related fields (payments, delinquency after origination, status, recoveries) can inflate performance and invalidate results.

## Modeling
- Candidates: Logistic Regression (interpretable baseline), HistGradientBoosting (performance model)
- Selection metric on policy-validation (uncalibrated): {base.get("selection_metric")}
- Chosen base model: {base.get("chosen_base")}

## Calibration
- Method: `{cal.get("method")}`
- Evaluation includes before/after calibration overlay plots and Brier score changes.

## Final Test Performance (Calibrated)
{json.dumps(metrics, indent=2)}

## Monitoring
- PSI drift computed for predicted PD between base-train and test.
- Drift response plan is described in MONITORING_REPORT.md
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_monitoring_report(cfg: Config, outdir: str, summary: dict):
    path = os.path.join(outdir, "MONITORING_REPORT.md")
    psi_pd = summary.get("monitoring", {}).get("psi_predicted_pd_base_train_vs_test")

    text = f"""# Monitoring Report — PD Model

## What we monitor
### 1) Input / Population Drift (proxy)
- **Predicted PD PSI** between baseline (base-train) and a later sample (test).
- Current PSI: **{psi_pd:.6f}**

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
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="CV-ready PD model project (training + calibration + policy + monitoring).")
    parser.add_argument("--data_path", type=str, default="data/loans_clean.csv")
    parser.add_argument("--target_col", type=str, default="Default")
    parser.add_argument("--date_col", type=str, default=None)
    parser.add_argument("--parse_date_format", type=str, default=None)

    parser.add_argument("--calibration_method", type=str, default="sigmoid", choices=["sigmoid", "isotonic"])
    parser.add_argument("--decision_objective", type=str, default="max_profit", choices=["max_profit", "profit_with_el_cap"])
    parser.add_argument("--el_cap", type=float, default=5e6)

    parser.add_argument("--extra_drop_cols", type=str, default="", help="Comma-separated columns to drop from PD features.")
    parser.add_argument("--report_dir", type=str, default="reports")
    parser.add_argument("--model_dir", type=str, default="models")
    args = parser.parse_args()

    cfg = Config(
        data_path=args.data_path,
        target_col=args.target_col,
        date_col=args.date_col,
        parse_date_format=args.parse_date_format,
        calibration_method=args.calibration_method,
        decision_objective=args.decision_objective,
        el_cap=args.el_cap,
        report_dir=args.report_dir,
        model_dir=args.model_dir,
        extra_drop_cols=tuple([c.strip() for c in args.extra_drop_cols.split(",") if c.strip()])
    )

    ensure_dirs(cfg)

    df = load_data(cfg)
    df[cfg.target_col] = df[cfg.target_col].astype(int)

    # leakage drop list
    drop_cols, leak_reasons = leakage_drop_list(cfg, df)

    # split
    split_meta = None
    if cfg.date_col and cfg.date_col in df.columns:
        train_df, test_df, split_meta = split_time_oot(cfg, df)
    else:
        warnings.warn("No usable date_col found; using stratified random split. For credit risk, prefer time-based OOT split.")
        train_df, test_df, split_meta = split_random(cfg, df)

    # Within train: policy validation and calibration holdouts
    train_remain_df, policy_val_df = train_test_split(
        train_df, test_size=cfg.policy_valid_size, random_state=cfg.random_state,
        stratify=train_df[cfg.target_col]
    )

    base_train_df, cal_df = train_test_split(
        train_remain_df, test_size=cfg.calibration_size, random_state=cfg.random_state,
        stratify=train_remain_df[cfg.target_col]
    )

    # Prepare X/y
    y_base = base_train_df[cfg.target_col].values
    y_cal = cal_df[cfg.target_col].values
    y_val = policy_val_df[cfg.target_col].values
    y_te = test_df[cfg.target_col].values

    # Drop target + leakage cols from PD features
    X_base = base_train_df.drop(columns=[cfg.target_col] + drop_cols, errors="ignore")
    X_cal  = cal_df.drop(columns=[cfg.target_col] + drop_cols, errors="ignore")
    X_val  = policy_val_df.drop(columns=[cfg.target_col] + drop_cols, errors="ignore")
    X_te   = test_df.drop(columns=[cfg.target_col] + drop_cols, errors="ignore")

    # Preprocessor fit only on base train
    num_cols, cat_cols = infer_feature_types(X_base)
    pre = build_preprocessor(num_cols, cat_cols)

    # Models
    lr = LogisticRegression(max_iter=3000, class_weight="balanced")
    lr_pipe = Pipeline([("preprocess", pre), ("model", lr)])

    hgb = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.07, max_iter=500, random_state=cfg.random_state
    )
    hgb_pipe = Pipeline([("preprocess", pre), ("model", hgb)])

    # Train both on base train
    lr_pipe.fit(X_base, y_base)
    p_lr_val = lr_pipe.predict_proba(X_val)[:, 1]
    lr_val_metrics = prob_metrics(y_val, p_lr_val)

    # HGB with simple class weighting
    pos_weight = (y_base == 0).sum() / max((y_base == 1).sum(), 1)
    w_base = np.where(y_base == 1, pos_weight, 1.0)
    hgb_pipe.fit(X_base, y_base, model__sample_weight=w_base)
    p_hgb_val = hgb_pipe.predict_proba(X_val)[:, 1]
    hgb_val_metrics = prob_metrics(y_val, p_hgb_val)

    # Select by PR-AUC (uncalibrated) on policy validation
    if hgb_val_metrics["pr_auc"] >= lr_val_metrics["pr_auc"]:
        chosen_base_name = "hgb"
        base_model = hgb_pipe
    else:
        chosen_base_name = "logreg"
        base_model = lr_pipe

    # Refit base model on base train (already fit above; refit for cleanliness)
    if chosen_base_name == "hgb":
        base_model.fit(X_base, y_base, model__sample_weight=w_base)
    else:
        base_model.fit(X_base, y_base)

    # Calibration on cal_df
    calibrated = ManualCalibratedModel(base_model, method=cfg.calibration_method)
    calibrated.fit(X_cal, y_cal)

    # Before/after calibration evaluation (on TEST, plus plots)
    p_te_uncal = base_model.predict_proba(X_te)[:, 1]
    p_te_cal = calibrated.predict_proba(X_te)[:, 1]

    metrics_uncal = prob_metrics(y_te, p_te_uncal)
    metrics_cal = prob_metrics(y_te, p_te_cal)

    plot_roc_pr(y_te, p_te_cal, cfg.report_dir, prefix="test_calibrated")
    plot_calibration_overlay(y_te, p_te_uncal, p_te_cal, cfg.report_dir, prefix="test")

    # Policy tuning on policy validation (calibrated probabilities)
    p_val = calibrated.predict_proba(X_val)[:, 1]
    ead_val, lgd_val = get_ead_lgd(cfg, policy_val_df)
    benefit_val, benefit_meta = compute_benefit(cfg, policy_val_df, ead_val)

    threshold, thresh_info_val = choose_threshold(cfg, p_val, ead_val, lgd_val, benefit_val)
    val_ops = confusion_at_threshold(y_val, p_val, threshold)
    plot_decision_curves(p_val, ead_val, lgd_val, benefit_val, cfg.report_dir)

    # Test policy outcomes
    ead_te, lgd_te = get_ead_lgd(cfg, test_df)
    benefit_te, _ = compute_benefit(cfg, test_df, ead_te)

    approve_te = p_te_cal < threshold
    el_te = float((p_te_cal[approve_te] * lgd_te[approve_te] * ead_te[approve_te]).sum())
    prof_te = float(expected_profit(p_te_cal[approve_te], ead_te[approve_te], lgd_te[approve_te], benefit_te[approve_te]).sum())
    test_ops = confusion_at_threshold(y_te, p_te_cal, threshold)

    # Risk bands lift table (on test)
    lift_te = lift_table(y_te, p_te_cal, cfg.n_risk_bands)
    lift_path = os.path.join(cfg.report_dir, "lift_table_test.csv")
    lift_te.to_csv(lift_path, index=False)

    # Monitoring PSI: base-train vs test predicted PD
    p_base = calibrated.predict_proba(X_base)[:, 1]
    psi_pd = psi(p_base, p_te_cal, bins=10)

    # Permutation importance on base model (uncalibrated, but fine for feature relevance)
    # NOTE: This yields importances for raw columns, not one-hot columns.
    X_pi = X_te.sample(min(2000, len(X_te)), random_state=cfg.random_state)
    y_pi = pd.Series(y_te, index=X_te.index).loc[X_pi.index].values
    pi = permutation_importance(
        base_model, X_pi, y_pi, scoring="average_precision", n_repeats=5, random_state=cfg.random_state
    )
    raw_imp = pd.DataFrame({
        "feature": X_pi.columns,
        "importance_mean": pi.importances_mean,
        "importance_std": pi.importances_std
    }).sort_values("importance_mean", ascending=False)
    raw_imp.to_csv(os.path.join(cfg.report_dir, "permutation_importance_raw_features.csv"), index=False)

    # Save artifacts
    summary = {
        "config": asdict(cfg),
        "split": split_meta,
        "leakage": leak_reasons,
        "dropped_columns_all": drop_cols,
        "model_selection": {
            "logreg_val_uncal": lr_val_metrics,
            "hgb_val_uncal": hgb_val_metrics,
            "chosen_base": chosen_base_name,
            "selection_metric": "pr_auc_on_policy_validation_uncalibrated"
        },
        "calibration": {
            "method": cfg.calibration_method,
            "protocol": "base model trained on base_train; calibrator fit on cal_df; eval on test"
        },
        "final_test": {
            "metrics_uncalibrated": metrics_uncal,
            "metrics_calibrated": metrics_cal
        },
        "policy_threshold": {
            "tuned_on": "policy_validation_only",
            "objective": cfg.decision_objective,
            "threshold": float(threshold),
            "validation_portfolio": thresh_info_val,
            "validation_ops": val_ops
        },
        "test_policy_outcomes": {
            "approval_rate": float(approve_te.mean()),
            "expected_loss_total": el_te,
            "expected_profit_total": prof_te,
            "test_ops": test_ops,
            "test_default_rate": float(np.mean(y_te)),
            "avg_predicted_pd_test": float(np.mean(p_te_cal)),
            "avg_predicted_pd_approved": float(np.mean(p_te_cal[approve_te])) if int(approve_te.sum()) > 0 else None,
            "profit_per_approved": float(prof_te / max(int(approve_te.sum()), 1)),
            "el_per_approved": float(el_te / max(int(approve_te.sum()), 1)),
            "lift_table_path": lift_path,
        },
        "economics": {
            "benefit_meta": benefit_meta,
            "funding_rate_annual": cfg.funding_rate_annual,
            "orig_fee_rate": cfg.orig_fee_rate,
            "ops_cost_per_loan": cfg.ops_cost_per_loan,
            "note": "simplified unit economics for portfolio project"
        },
        "monitoring": {
            "psi_predicted_pd_base_train_vs_test": float(psi_pd),
            "psi_bins": 10
        }
    }

    with open(os.path.join(cfg.report_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Write reports
    # Leakage report
    with open(os.path.join(cfg.report_dir, "LEAKAGE_DROPS.json"), "w", encoding="utf-8") as f:
        json.dump({"drop_cols": drop_cols, "reasons": leak_reasons}, f, indent=2)

    write_model_card(cfg, cfg.report_dir, summary)
    write_monitoring_report(cfg, cfg.report_dir, summary)

    # Save model
    joblib.dump(calibrated, os.path.join(cfg.model_dir, "pd_model_calibrated.joblib"))

    print("\nDone ✅")
    print(json.dumps({
        "chosen_base": chosen_base_name,
        "split": split_meta,
        "test_metrics_uncal": metrics_uncal,
        "test_metrics_cal": metrics_cal,
        "psi_predicted_pd": psi_pd,
        "threshold": float(threshold),
        "test_approval_rate": float(approve_te.mean()),
        "test_expected_profit_total": prof_te,
        "test_expected_loss_total": el_te,
        "lift_table_saved": lift_path,
        "leakage_drop_count": len(drop_cols),
        "leakage_drop_report": os.path.join(cfg.report_dir, "LEAKAGE_DROPS.json"),
        "plots": [
            os.path.join(cfg.report_dir, "test_calibrated_roc_curve.png"),
            os.path.join(cfg.report_dir, "test_calibrated_pr_curve.png"),
            os.path.join(cfg.report_dir, "test_calibration_overlay.png"),
            os.path.join(cfg.report_dir, "decision_curves_profit_el.png"),
            os.path.join(cfg.report_dir, "decision_curve_approval.png")
        ],
        "reports": [
            os.path.join(cfg.report_dir, "MODEL_CARD.md"),
            os.path.join(cfg.report_dir, "MONITORING_REPORT.md")
        ]
    }, indent=2))


if __name__ == "__main__":
    main()