import numpy as np
import pytest

from src.evaluation import best_f1_threshold, classification_metrics, threshold_table


def test_classification_metrics_returns_expected_values():
    fraud_labels = np.array([0, 0, 1, 1])
    fraud_scores = np.array([0.1, 0.4, 0.7, 0.9])

    metrics = classification_metrics(fraud_labels, fraud_scores, threshold=0.5)

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["tp"] == 2


def test_classification_metrics_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        classification_metrics([0, 1], [0.2], threshold=0.5)


def test_classification_metrics_rejects_scores_outside_probability_range():
    with pytest.raises(ValueError, match="between 0 and 1"):
        classification_metrics([0, 1], [0.2, 1.2], threshold=0.5)


def test_threshold_table_returns_confusion_counts_for_known_case():
    fraud_labels = np.array([0, 0, 1, 1])
    fraud_scores = np.array([0.2, 0.6, 0.4, 0.9])

    threshold_results = threshold_table(
        fraud_labels,
        fraud_scores,
        thresholds=np.array([0.5]),
    )

    threshold_row = threshold_results.iloc[0]
    assert threshold_row["tn"] == 1
    assert threshold_row["fp"] == 1
    assert threshold_row["fn"] == 1
    assert threshold_row["tp"] == 1
    assert threshold_row["precision"] == pytest.approx(0.5)
    assert threshold_row["recall"] == pytest.approx(0.5)
    assert threshold_row["f1"] == pytest.approx(0.5)


def test_best_f1_threshold_selects_strongest_candidate():
    fraud_labels = np.array([0, 0, 1, 1])
    fraud_scores = np.array([0.1, 0.3, 0.6, 0.9])

    selected_threshold = best_f1_threshold(fraud_labels, fraud_scores)

    assert 0.3 < selected_threshold <= 0.6


def test_best_f1_threshold_can_select_score_above_point_99():
    fraud_labels = np.array([0, 0, 1, 1])
    fraud_scores = np.array([0.10, 0.995, 0.996, 0.999])

    selected_threshold = best_f1_threshold(fraud_labels, fraud_scores)

    assert selected_threshold > 0.99
    assert selected_threshold <= 0.996
