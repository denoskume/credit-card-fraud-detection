import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _validate_inputs(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)

    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("y_true and y_score must be one-dimensional")
    if len(labels) != len(scores):
        raise ValueError("y_true and y_score must have the same length")
    if len(labels) == 0:
        raise ValueError("y_true and y_score must not be empty")
    if not np.isfinite(scores).all():
        raise ValueError("y_score must contain only finite values")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("y_true must contain only 0 and 1")
    if len(np.unique(labels)) != 2:
        raise ValueError("y_true must contain both classes")

    return labels.astype(int), scores


def classification_metrics(y_true, y_score, threshold: float = 0.5) -> dict[str, float | int]:
    labels, scores = _validate_inputs(y_true, y_score)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_table(y_true, y_score, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    labels, scores = _validate_inputs(y_true, y_score)
    candidate_thresholds = (
        np.linspace(0.0, 1.0, 201) if thresholds is None else np.asarray(thresholds, dtype=float)
    )

    if candidate_thresholds.ndim != 1 or len(candidate_thresholds) == 0:
        raise ValueError("thresholds must be a non-empty one-dimensional array")
    if not np.isfinite(candidate_thresholds).all() or np.any((candidate_thresholds < 0) | (candidate_thresholds > 1)):
        raise ValueError("thresholds must contain finite values between 0 and 1")

    rows = []
    for threshold in candidate_thresholds:
        predictions = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(labels, predictions, zero_division=0)),
                "recall": float(recall_score(labels, predictions, zero_division=0)),
                "f1": float(f1_score(labels, predictions, zero_division=0)),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

    return pd.DataFrame(rows)


def best_f1_threshold(y_true, y_score) -> float:
    labels, scores = _validate_inputs(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(labels, scores)

    precision = precision[:-1]
    recall = recall[:-1]
    denominator = precision + recall
    f1_values = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )

    return float(thresholds[int(np.argmax(f1_values))])
