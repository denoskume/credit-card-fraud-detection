import numpy as np

from src.business_metrics import BusinessCostAssumptions, cost_sensitivity_table, expected_cost


def test_expected_cost_matches_hand_computed_example():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.9, 0.2, 0.8])
    amounts = np.array([10.0, 20.0, 30.0, 40.0])
    assumptions = BusinessCostAssumptions(
        missed_fraud_multiplier=1.0,
        false_positive_review_cost=5.0,
        recoverable_fraud_fraction=0.5,
    )

    result = expected_cost(y_true, y_score, amounts, 0.5, assumptions)

    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["tp"] == 1
    assert result["missed_fraud_cost"] == 30.0
    assert result["false_positive_cost"] == 5.0
    assert result["recovered_fraud_value"] == 20.0
    assert result["estimated_net_cost"] == 15.0


def test_expected_cost_handles_zero_false_positives():
    result = expected_cost(
        np.array([0, 1]),
        np.array([0.1, 0.9]),
        np.array([10.0, 25.0]),
        0.5,
        BusinessCostAssumptions(false_positive_review_cost=7.0),
    )

    assert result["fp"] == 0
    assert result["false_positive_cost"] == 0.0


def test_expected_cost_handles_zero_false_negatives():
    result = expected_cost(
        np.array([0, 1]),
        np.array([0.9, 0.9]),
        np.array([10.0, 25.0]),
        0.5,
        BusinessCostAssumptions(),
    )

    assert result["fn"] == 0
    assert result["missed_fraud_cost"] == 0.0


def test_cost_sensitivity_table_returns_threshold_assumption_grid():
    table = cost_sensitivity_table(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.8, 0.4, 0.9]),
        np.array([10.0, 20.0, 30.0, 40.0]),
        thresholds=np.array([0.3, 0.6]),
        assumptions=[BusinessCostAssumptions(), BusinessCostAssumptions(false_positive_review_cost=10.0)],
    )

    assert len(table) == 4
    assert set(table["threshold"]) == {0.3, 0.6}
