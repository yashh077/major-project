"""Alternative evaluation script with 0.96 KPI thresholds."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    accuracy_score,
)


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "Datasets" / "finalDataset.csv"
MODEL_PATH = ROOT / "fake_classifier.pkl"
RESULTS_PATH = Path(__file__).resolve().parent / "test_midels_results.json"
RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERIC_FEATURES = [
    "Age",
    "Education_Level_Code",
    "Years of Experience",
    "Senior",
    "Salary_PPP_Adjusted",
    "PPP_Index",
]
CATEGORICAL_FEATURES = ["Gender", "Education_Level", "Job Title", "Country", "Race"]

THRESHOLD = 0.96  # universal acceptance threshold


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    return df.replace({np.inf: np.nan, -np.inf: np.nan}).dropna(subset=["Fake_Job_Risk"])


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["Fake_Job_Risk"].astype(int)
    stratify = y if y.nunique() == 2 else None
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )
    return X_test, y_test


def evaluate(X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "fake_classifier.pkl is missing. Run `python train_models.py` first."
        )
    model = joblib.load(MODEL_PATH)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    return {
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        "PR_AUC": average_precision_score(y_test, y_prob),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy_score(y_test, y_pred),
    }


def main() -> None:
    df = load_dataset()
    X_test, y_test = split_data(df)
    metrics = evaluate(X_test, y_test)

    passes = {metric: value >= THRESHOLD for metric, value in metrics.items()}
    results = {"threshold": THRESHOLD, "metrics": metrics, "status": passes}
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("=== TEST MIDELS SUMMARY ===")
    print(f"Threshold: {THRESHOLD:.2f}")
    for metric, value in metrics.items():
        label = "PASS" if passes[metric] else "FAIL"
        print(f"  {metric:>10s}: {value:.4f} [{label}]")
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

