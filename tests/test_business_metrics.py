import numpy as np
import pytest

from src.business_metrics import BusinessCostAssumptions, cost_sensitivity_table, expected_cost


def test_expected_cost_matches_hand_calculated_fp_fn_case():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.8, 0.4, 0.9])
    amounts = np.array([20.0, 30.0, 100.0, 200.0])
    assumptions = BusinessCostAssumptions(
        missed_fraud_cost_multiplier=1.0,
        false_positive_review_cost=10.0,
        detected_fraud_recovery_rate=0.25,
    )

    result = expected_cost(y_true, y_score, amounts, 0.5, assumptions)

    assert (result["fp"], result["fn"], result["tp"]) == (1, 1, 1)
    assert result["missed_fraud_cost"] == pytest.approx(100.0)
    assert result["false_positive_cost"] == pytest.approx(10.0)
    assert result["recovered_fraud_value"] == pytest.approx(50.0)
    assert result["net_cost"] == pytest.approx(60.0)


def test_expected_cost_handles_zero_fp_and_zero_fn():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    amounts = np.array([20.0, 30.0, 100.0, 200.0])
    assumptions = BusinessCostAssumptions(false_positive_review_cost=12.0)

    result = expected_cost(y_true, y_score, amounts, 0.5, assumptions)

    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["false_positive_cost"] == 0.0
    assert result["missed_fraud_cost"] == 0.0


def test_cost_sensitivity_table_compares_thresholds():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.6, 0.4, 0.9])
    amounts = np.array([10.0, 20.0, 100.0, 200.0])
    assumptions = BusinessCostAssumptions()

    table = cost_sensitivity_table(
        y_true,
        y_score,
        amounts,
        thresholds=np.array([0.3, 0.7]),
        assumptions=assumptions,
    )

    assert table["threshold"].tolist() == pytest.approx([0.3, 0.7])
    assert np.isfinite(table["net_cost"]).all()
