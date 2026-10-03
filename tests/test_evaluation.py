import numpy as np
import pytest

from src.evaluation import best_f1_threshold, classification_metrics, threshold_table


EXPECTED_KEYS = {
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "tn",
    "fp",
    "fn",
    "tp",
}


def test_classification_metrics_returns_expected_finite_values():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.6, 0.9])

    metrics = classification_metrics(y_true, y_score)

    assert set(metrics) == EXPECTED_KEYS
    assert all(np.isfinite(value) for value in metrics.values())
    for key in ["roc_auc", "pr_auc", "precision", "recall", "f1"]:
        assert 0.0 <= metrics[key] <= 1.0
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (2, 0, 0, 2)


def test_classification_metrics_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        classification_metrics([0, 1], [0.2])


def test_threshold_table_has_known_confusion_counts():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.2, 0.7, 0.4, 0.9])

    table = threshold_table(y_true, y_score, thresholds=np.array([0.5]))
    row = table.iloc[0]

    assert row["threshold"] == pytest.approx(0.5)
    assert (row["tn"], row["fp"], row["fn"], row["tp"]) == (1, 1, 1, 1)
    assert np.isfinite(row[["precision", "recall", "f1"]].to_numpy(dtype=float)).all()


def test_best_f1_threshold_returns_threshold_from_sweep():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.3, 0.4, 0.9])

    threshold = best_f1_threshold(y_true, y_score)

    assert 0.0 <= threshold <= 1.0
    assert classification_metrics(y_true, y_score, threshold)["f1"] == pytest.approx(1.0)
