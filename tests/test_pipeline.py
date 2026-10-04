import numpy as np
import pandas as pd

from src.business_metrics import BusinessCostAssumptions, expected_cost
from src.data import EXPECTED_COLUMNS, split_dataset
from src.evaluation import classification_metrics
from src.models import build_logistic_regression
from src.preprocessing import build_linear_preprocessor, fit_transform_splits


def build_small_transaction_dataset() -> pd.DataFrame:
    transaction_count = 40
    transaction_dataset = pd.DataFrame(
        np.zeros((transaction_count, len(EXPECTED_COLUMNS))),
        columns=EXPECTED_COLUMNS,
    )
    transaction_dataset["Time"] = np.arange(transaction_count, dtype=float)
    transaction_dataset["Amount"] = np.linspace(1.0, 200.0, transaction_count)
    transaction_dataset["Class"] = [0] * 32 + [1] * 8
    return transaction_dataset


def test_pipeline_fit_predict_and_metrics_path():
    transaction_dataset = build_small_transaction_dataset()

    (
        training_features,
        validation_features,
        test_features,
        training_labels,
        validation_labels,
        _,
    ) = split_dataset(transaction_dataset, random_state=42)

    preprocessor = build_linear_preprocessor(training_features.columns.tolist())
    (
        transformed_training_features,
        transformed_validation_features,
        _,
        _,
    ) = fit_transform_splits(
        preprocessor,
        training_features,
        validation_features,
        test_features,
    )

    model = build_logistic_regression(random_state=42)
    model.fit(transformed_training_features, training_labels)
    validation_scores = model.predict_proba(transformed_validation_features)[:, 1]

    metrics = classification_metrics(validation_labels, validation_scores, threshold=0.5)

    expected_keys = {
        "threshold",
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
    assert expected_keys == set(metrics)
    assert np.isfinite(validation_scores).all()


def test_pipeline_business_cost_known_case():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.8, 0.4, 0.9])
    amounts = np.array([10.0, 20.0, 100.0, 200.0])
    assumptions = BusinessCostAssumptions(
        missed_fraud_multiplier=1.0,
        false_positive_review_cost=5.0,
        recoverable_fraud_fraction=0.0,
    )

    result = expected_cost(labels, scores, amounts, 0.5, assumptions)

    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["missed_fraud_cost"] == 100.0
    assert result["false_positive_cost"] == 5.0
    assert result["total_expected_cost"] == 105.0
