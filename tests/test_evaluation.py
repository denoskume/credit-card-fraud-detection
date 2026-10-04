import numpy as np
import pandas as pd
import pytest

from src.evaluation import best_f1_threshold, classification_metrics, threshold_table


def test_classification_metrics_returns_expected_finite_values():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])

    metrics = classification_metrics(y_true, y_score, threshold=0.5)

    assert set(metrics) == {
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "tn",
        "fp",
        "fn",
        "tp",
        "threshold",
    }
    assert all(np.isfinite(value) for value in metrics.values())
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_classification_metrics_known_confusion_counts():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.7, 0.4, 0.9])

    metrics = classification_metrics(y_true, y_score, threshold=0.5)

    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1


def test_classification_metrics_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        classification_metrics(np.array([0, 1]), np.array([0.2]))


def test_threshold_table_returns_one_row_per_threshold():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.6, 0.9])
    thresholds = np.array([0.25, 0.5, 0.75])

    table = threshold_table(y_true, y_score, thresholds)

    assert isinstance(table, pd.DataFrame)
    assert table["threshold"].tolist() == thresholds.tolist()
    assert np.isfinite(table.select_dtypes(include="number").to_numpy()).all()


def test_best_f1_threshold_returns_candidate_from_sweep():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.6, 0.9])

    threshold = best_f1_threshold(y_true, y_score)

    assert 0.0 < threshold < 1.0
