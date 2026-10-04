from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BusinessCostAssumptions:
    missed_fraud_multiplier: float = 1.0
    false_positive_review_cost: float = 5.0
    recoverable_fraud_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.missed_fraud_multiplier < 0:
            raise ValueError("missed_fraud_multiplier must be non-negative.")
        if self.false_positive_review_cost < 0:
            raise ValueError("false_positive_review_cost must be non-negative.")
        if not 0.0 <= self.recoverable_fraud_fraction <= 1.0:
            raise ValueError("recoverable_fraud_fraction must be between 0 and 1.")


def expected_cost(
    y_true,
    y_score,
    amounts,
    threshold: float,
    assumptions: BusinessCostAssumptions,
) -> dict[str, float | int]:
    true = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    transaction_amounts = np.asarray(amounts, dtype=float)

    if not (len(true) == len(score) == len(transaction_amounts)):
        raise ValueError("y_true, y_score, and amounts must have the same length.")
    if not np.isfinite(score).all() or not np.isfinite(transaction_amounts).all():
        raise ValueError("Scores and amounts must contain only finite values.")
    if (transaction_amounts < 0).any():
        raise ValueError("Transaction amounts must be non-negative.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    prediction = (score >= threshold).astype(int)
    false_positive_mask = (true == 0) & (prediction == 1)
    false_negative_mask = (true == 1) & (prediction == 0)
    true_positive_mask = (true == 1) & (prediction == 1)

    false_positives = int(false_positive_mask.sum())
    false_negatives = int(false_negative_mask.sum())
    true_positives = int(true_positive_mask.sum())
    true_negatives = int(((true == 0) & (prediction == 0)).sum())

    missed_fraud_cost = float(
        transaction_amounts[false_negative_mask].sum() * assumptions.missed_fraud_multiplier
    )
    false_positive_cost = float(false_positives * assumptions.false_positive_review_cost)
    recovered_fraud_value = float(
        transaction_amounts[true_positive_mask].sum() * assumptions.recoverable_fraud_fraction
    )
    estimated_net_cost = missed_fraud_cost + false_positive_cost - recovered_fraud_value

    return {
        "threshold": float(threshold),
        "tn": true_negatives,
        "fp": false_positives,
        "fn": false_negatives,
        "tp": true_positives,
        "missed_fraud_cost": missed_fraud_cost,
        "false_positive_cost": false_positive_cost,
        "recovered_fraud_value": recovered_fraud_value,
        "estimated_net_cost": float(estimated_net_cost),
    }


def cost_sensitivity_table(
    y_true,
    y_score,
    amounts,
    thresholds: np.ndarray,
    assumptions: list[BusinessCostAssumptions],
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for assumption_index, assumption in enumerate(assumptions):
        for threshold in thresholds:
            result = expected_cost(y_true, y_score, amounts, float(threshold), assumption)
            result["assumption_id"] = assumption_index
            result.update(asdict(assumption))
            rows.append(result)
    return pd.DataFrame(rows)
