import numpy as np
import torch

from src.models import (
    FraudMLP,
    build_logistic_regression,
    build_random_forest,
    build_xgboost,
)


def make_binary_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 6))
    y = np.array([0] * 30 + [1] * 10)
    return X, y


def test_logistic_regression_is_trainable_and_probability_based():
    X, y = make_binary_data()
    model = build_logistic_regression()

    model.fit(X, y)
    probabilities = model.predict_proba(X)[:, 1]

    assert model.class_weight == "balanced"
    assert probabilities.shape == (40,)
    assert np.isfinite(probabilities).all()


def test_random_forest_is_trainable_and_probability_based():
    X, y = make_binary_data()
    model = build_random_forest()

    model.fit(X, y)
    probabilities = model.predict_proba(X)[:, 1]

    assert model.class_weight == "balanced_subsample"
    assert probabilities.shape == (40,)
    assert np.isfinite(probabilities).all()


def test_xgboost_is_trainable_and_probability_based():
    X, y = make_binary_data()
    model = build_xgboost()

    model.fit(X, y)
    probabilities = model.predict_proba(X)[:, 1]

    assert probabilities.shape == (40,)
    assert np.isfinite(probabilities).all()


def test_fraud_mlp_produces_one_logit_per_row():
    model = FraudMLP(input_dim=6)
    features = torch.zeros((8, 6), dtype=torch.float32)

    logits = model(features)

    assert logits.shape == (8,)
    assert torch.isfinite(logits).all()
