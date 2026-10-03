import numpy as np
import pandas as pd

from src.business_metrics import BusinessCostAssumptions, expected_cost
from src.data import EXPECTED_COLUMNS, split_dataset
from src.evaluation import classification_metrics
from src.models import build_logistic_regression
from src.preprocessing import build_linear_preprocessor


def make_dataset(rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {"Time": np.arange(rows, dtype=float)}
    for index in range(1, 29):
        data[f"V{index}"] = rng.normal(size=rows)
    data["Amount"] = rng.uniform(1.0, 500.0, size=rows)
    data["Class"] = np.array([1 if index % 10 == 0 else 0 for index in range(rows)])
    return pd.DataFrame(data, columns=EXPECTED_COLUMNS)


def test_end_to_end_classical_pipeline_produces_metrics_and_cost():
    frame = make_dataset()
    X_train, X_validation, _, y_train, y_validation, _ = split_dataset(frame)

    preprocessor = build_linear_preprocessor(X_train.columns.tolist())
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_validation_transformed = preprocessor.transform(X_validation)

    model = build_logistic_regression()
    model.fit(X_train_transformed, y_train)
    scores = model.predict_proba(X_validation_transformed)[:, 1]

    metrics = classification_metrics(y_validation, scores, threshold=0.5)
    cost = expected_cost(
        y_validation,
        scores,
        X_validation["Amount"].to_numpy(),
        threshold=0.5,
        assumptions=BusinessCostAssumptions(),
    )

    assert np.isfinite(X_train_transformed).all()
    assert np.isfinite(scores).all()
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert metrics["tp"] + metrics["fn"] == int(y_validation.sum())
    assert np.isfinite(cost["net_cost"])
