from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


@dataclass(frozen=True)
class BusinessCostAssumptions:
    missed_fraud_cost_multiplier: float = 1.0
    false_positive_review_cost: float = 5.0
    detected_fraud_recovery_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.missed_fraud_cost_multiplier < 0:
            raise ValueError("missed_fraud_cost_multiplier must be non-negative")
        if self.false_positive_review_cost < 0:
            raise ValueError("false_positive_review_cost must be non-negative")
        if not 0.0 <= self.detected_fraud_recovery_rate <= 1.0:
            raise ValueError("detected_fraud_recovery_rate must be between 0 and 1")


def _validate_cost_inputs(y_true, y_score, amounts, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    transaction_amounts = np.asarray(amounts, dtype=float)

    if labels.ndim != 1 or scores.ndim != 1 or transaction_amounts.ndim != 1:
        raise ValueError("y_true, y_score and amounts must be one-dimensional")
    if not (len(labels) == len(scores) == len(transaction_amounts)):
        raise ValueError("y_true, y_score and amounts must have the same length")
    if len(labels) == 0:
        raise ValueError("inputs must not be empty")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("y_true must contain only 0 and 1")
    if not np.isfinite(scores).all():
        raise ValueError("y_score must contain only finite values")
    if not np.isfinite(transaction_amounts).all() or np.any(transaction_amounts < 0):
        raise ValueError("amounts must contain finite non-negative values")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    return labels.astype(int), scores, transaction_amounts


def expected_cost(
    y_true,
    y_score,
    amounts,
    threshold: float,
    assumptions: BusinessCostAssumptions,
) -> dict[str, float | int]:
    labels, scores, transaction_amounts = _validate_cost_inputs(y_true, y_score, amounts, threshold)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    false_negative_mask = (labels == 1) & (predictions == 0)
    true_positive_mask = (labels == 1) & (predictions == 1)

    missed_fraud_cost = float(
        transaction_amounts[false_negative_mask].sum() * assumptions.missed_fraud_cost_multiplier
    )
    false_positive_cost = float(fp * assumptions.false_positive_review_cost)
    recovered_fraud_value = float(
        transaction_amounts[true_positive_mask].sum() * assumptions.detected_fraud_recovery_rate
    )
    net_cost = missed_fraud_cost + false_positive_cost - recovered_fraud_value

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "missed_fraud_cost": missed_fraud_cost,
        "false_positive_cost": false_positive_cost,
        "recovered_fraud_value": recovered_fraud_value,
        "net_cost": float(net_cost),
    }


def cost_sensitivity_table(
    y_true,
    y_score,
    amounts,
    thresholds: np.ndarray,
    assumptions: BusinessCostAssumptions,
) -> pd.DataFrame:
    candidate_thresholds = np.asarray(thresholds, dtype=float)
    if candidate_thresholds.ndim != 1 or len(candidate_thresholds) == 0:
        raise ValueError("thresholds must be a non-empty one-dimensional array")

    rows = []
    for threshold in candidate_thresholds:
        row = expected_cost(y_true, y_score, amounts, float(threshold), assumptions)
        rows.append({"threshold": float(threshold), **row})

    return pd.DataFrame(rows)
