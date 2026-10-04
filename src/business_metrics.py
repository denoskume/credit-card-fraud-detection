from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BusinessCostAssumptions:
    missed_fraud_multiplier: float = 1.0
    false_positive_review_cost: float = 0.0
    recoverable_fraud_fraction: float = 0.0

    def __post_init__(self):
        if self.missed_fraud_multiplier < 0:
            raise ValueError("missed_fraud_multiplier must be non-negative.")
        if self.false_positive_review_cost < 0:
            raise ValueError("false_positive_review_cost must be non-negative.")
        if not 0.0 <= self.recoverable_fraud_fraction <= 1.0:
            raise ValueError("recoverable_fraud_fraction must be between 0 and 1.")


def expected_cost(
    fraud_labels,
    probability_scores,
    transaction_amounts,
    threshold: float,
    assumptions: BusinessCostAssumptions,
) -> dict[str, float | int]:
    fraud_labels = np.asarray(fraud_labels)
    probability_scores = np.asarray(probability_scores, dtype=float)
    transaction_amounts = np.asarray(transaction_amounts, dtype=float)

    if not (
        len(fraud_labels)
        == len(probability_scores)
        == len(transaction_amounts)
    ):
        raise ValueError("Labels, probability scores, and amounts must have equal length.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    if not np.isfinite(probability_scores).all():
        raise ValueError("Probability scores must be finite.")
    if not np.isfinite(transaction_amounts).all():
        raise ValueError("Transaction amounts must be finite.")
    if (transaction_amounts < 0).any():
        raise ValueError("Transaction amounts must be non-negative.")

    predicted_fraud = probability_scores >= threshold
    actual_fraud = fraud_labels == 1

    false_negative_mask = actual_fraud & ~predicted_fraud
    false_positive_mask = ~actual_fraud & predicted_fraud

    false_negatives = int(false_negative_mask.sum())
    false_positives = int(false_positive_mask.sum())

    unrecovered_fraction = 1.0 - assumptions.recoverable_fraud_fraction
    missed_fraud_cost = float(
        transaction_amounts[false_negative_mask].sum()
        * assumptions.missed_fraud_multiplier
        * unrecovered_fraction
    )
    false_positive_cost = float(
        false_positives * assumptions.false_positive_review_cost
    )

    return {
        "threshold": float(threshold),
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "missed_fraud_cost": missed_fraud_cost,
        "false_positive_cost": false_positive_cost,
        "total_expected_cost": missed_fraud_cost + false_positive_cost,
    }


def cost_sensitivity_table(
    fraud_labels,
    probability_scores,
    transaction_amounts,
    assumptions: BusinessCostAssumptions,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    threshold_values = (
        np.linspace(0.0, 1.0, 101)
        if thresholds is None
        else np.asarray(thresholds, dtype=float)
    )

    results = [
        expected_cost(
            fraud_labels,
            probability_scores,
            transaction_amounts,
            threshold=float(threshold),
            assumptions=assumptions,
        )
        for threshold in threshold_values
    ]
    return pd.DataFrame(results)
