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


def _validate_binary_inputs(fraud_labels, fraud_scores) -> tuple[np.ndarray, np.ndarray]:
    label_array = np.asarray(fraud_labels)
    score_array = np.asarray(fraud_scores, dtype=float)

    if label_array.shape[0] != score_array.shape[0]:
        raise ValueError("Labels and scores must have the same length.")
    if label_array.shape[0] == 0:
        raise ValueError("Labels and scores must not be empty.")
    if set(np.unique(label_array)) - {0, 1}:
        raise ValueError("Labels must contain only binary values 0 and 1.")
    if not np.isfinite(score_array).all():
        raise ValueError("Scores must be finite.")
    if ((score_array < 0) | (score_array > 1)).any():
        raise ValueError("Scores must be between 0 and 1.")

    return label_array.astype(int), score_array


def _validate_threshold(threshold: float) -> float:
    threshold_value = float(threshold)
    if not 0 <= threshold_value <= 1:
        raise ValueError("Threshold must be between 0 and 1.")
    return threshold_value


def classification_metrics(
    fraud_labels,
    fraud_scores,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    label_array, score_array = _validate_binary_inputs(fraud_labels, fraud_scores)
    threshold_value = _validate_threshold(threshold)
    predicted_labels = (score_array >= threshold_value).astype(int)

    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        label_array,
        predicted_labels,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": threshold_value,
        "roc_auc": float(roc_auc_score(label_array, score_array)),
        "pr_auc": float(average_precision_score(label_array, score_array)),
        "precision": float(precision_score(label_array, predicted_labels, zero_division=0)),
        "recall": float(recall_score(label_array, predicted_labels, zero_division=0)),
        "f1": float(f1_score(label_array, predicted_labels, zero_division=0)),
        "tn": int(true_negative),
        "fp": int(false_positive),
        "fn": int(false_negative),
        "tp": int(true_positive),
    }


def threshold_table(
    fraud_labels,
    fraud_scores,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    label_array, score_array = _validate_binary_inputs(fraud_labels, fraud_scores)
    threshold_values = (
        np.linspace(0.01, 0.99, 99)
        if thresholds is None
        else np.asarray(thresholds, dtype=float)
    )

    if threshold_values.ndim != 1 or threshold_values.size == 0:
        raise ValueError("Thresholds must be a non-empty one-dimensional array.")

    rows = []
    for threshold_value in threshold_values:
        metrics = classification_metrics(label_array, score_array, threshold_value)
        rows.append(metrics)

    return pd.DataFrame(rows)


def best_f1_threshold(fraud_labels, fraud_scores) -> float:
    label_array, score_array = _validate_binary_inputs(fraud_labels, fraud_scores)
    precision_values, recall_values, threshold_values = precision_recall_curve(
        label_array,
        score_array,
    )

    if threshold_values.size == 0:
        return 0.5

    precision_values = precision_values[:-1]
    recall_values = recall_values[:-1]
    denominator = precision_values + recall_values
    f1_values = np.divide(
        2 * precision_values * recall_values,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )

    strongest_index = int(np.argmax(f1_values))
    return float(threshold_values[strongest_index])
