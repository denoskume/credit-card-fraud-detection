import numpy as np

from src.business_metrics import (
    BusinessCostAssumptions,
    cost_sensitivity_table,
    expected_cost,
)


def test_expected_cost_matches_hand_computed_example():
    fraud_labels = np.array([1, 1, 0, 0])
    probability_scores = np.array([0.9, 0.2, 0.8, 0.1])
    transaction_amounts = np.array([100.0, 200.0, 50.0, 80.0])
    assumptions = BusinessCostAssumptions(
        missed_fraud_multiplier=1.0,
        false_positive_review_cost=10.0,
        recoverable_fraud_fraction=0.0,
    )

    result = expected_cost(
        fraud_labels,
        probability_scores,
        transaction_amounts,
        threshold=0.5,
        assumptions=assumptions,
    )

    assert result["false_negatives"] == 1
    assert result["false_positives"] == 1
    assert result["missed_fraud_cost"] == 200.0
    assert result["false_positive_cost"] == 10.0
    assert result["total_expected_cost"] == 210.0


def test_expected_cost_handles_zero_false_positives():
    fraud_labels = np.array([1, 0, 0])
    probability_scores = np.array([0.9, 0.1, 0.2])
    transaction_amounts = np.array([100.0, 50.0, 60.0])
    assumptions = BusinessCostAssumptions(false_positive_review_cost=7.5)

    result = expected_cost(
        fraud_labels,
        probability_scores,
        transaction_amounts,
        threshold=0.5,
        assumptions=assumptions,
    )

    assert result["false_positives"] == 0
    assert result["false_positive_cost"] == 0.0


def test_expected_cost_handles_zero_false_negatives():
    fraud_labels = np.array([1, 1, 0])
    probability_scores = np.array([0.9, 0.8, 0.1])
    transaction_amounts = np.array([100.0, 150.0, 50.0])
    assumptions = BusinessCostAssumptions(missed_fraud_multiplier=2.0)

    result = expected_cost(
        fraud_labels,
        probability_scores,
        transaction_amounts,
        threshold=0.5,
        assumptions=assumptions,
    )

    assert result["false_negatives"] == 0
    assert result["missed_fraud_cost"] == 0.0


def test_recoverable_fraction_reduces_missed_fraud_cost():
    fraud_labels = np.array([1])
    probability_scores = np.array([0.1])
    transaction_amounts = np.array([100.0])
    assumptions = BusinessCostAssumptions(
        missed_fraud_multiplier=1.0,
        recoverable_fraud_fraction=0.25,
    )

    result = expected_cost(
        fraud_labels,
        probability_scores,
        transaction_amounts,
        threshold=0.5,
        assumptions=assumptions,
    )

    assert result["missed_fraud_cost"] == 75.0


def test_cost_sensitivity_table_returns_one_row_per_threshold():
    fraud_labels = np.array([1, 0, 1, 0])
    probability_scores = np.array([0.9, 0.6, 0.4, 0.1])
    transaction_amounts = np.array([100.0, 50.0, 120.0, 40.0])
    assumptions = BusinessCostAssumptions(false_positive_review_cost=5.0)
    thresholds = np.array([0.3, 0.5, 0.7])

    sensitivity_table = cost_sensitivity_table(
        fraud_labels,
        probability_scores,
        transaction_amounts,
        assumptions,
        thresholds=thresholds,
    )

    assert sensitivity_table["threshold"].tolist() == thresholds.tolist()
    assert sensitivity_table["total_expected_cost"].notna().all()
