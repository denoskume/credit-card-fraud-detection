import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _validated_arrays(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true)
    score = np.asarray(y_score, dtype=float)

    if true.shape[0] != score.shape[0]:
        raise ValueError("y_true and y_score must have the same length.")
    if true.ndim != 1 or score.ndim != 1:
        raise ValueError("y_true and y_score must be one-dimensional.")
    if not np.isfinite(score).all():
        raise ValueError("y_score must contain only finite values.")
    if not set(np.unique(true)).issubset({0, 1}) or len(np.unique(true)) != 2:
        raise ValueError("y_true must be binary and contain both classes.")

    return true.astype(int), score


def classification_metrics(y_true, y_score, threshold: float = 0.5) -> dict[str, float | int]:
    true, score = _validated_arrays(y_true, y_score)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    prediction = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(true, prediction, labels=[0, 1]).ravel()

    return {
        "roc_auc": float(roc_auc_score(true, score)),
        "pr_auc": float(average_precision_score(true, score)),
        "precision": float(precision_score(true, prediction, zero_division=0)),
        "recall": float(recall_score(true, prediction, zero_division=0)),
        "f1": float(f1_score(true, prediction, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def threshold_table(y_true, y_score, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    rows = [classification_metrics(y_true, y_score, float(threshold)) for threshold in thresholds]
    return pd.DataFrame(rows)


def best_f1_threshold(y_true, y_score) -> float:
    table = threshold_table(y_true, y_score)
    best_index = table["f1"].idxmax()
    return float(table.loc[best_index, "threshold"])
