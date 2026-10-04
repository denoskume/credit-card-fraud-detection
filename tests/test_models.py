import numpy as np
import torch

from src.models import (
    FraudMLP,
    build_logistic_regression,
    build_random_forest,
    build_xgboost,
)


def test_logistic_regression_constructor_is_trainable_and_seeded():
    model = build_logistic_regression(random_state=7)

    assert hasattr(model, "fit")
    assert hasattr(model, "predict_proba")
    assert model.random_state == 7
    assert model.class_weight == "balanced"


def test_random_forest_constructor_is_trainable_and_seeded():
    model = build_random_forest(random_state=7)

    assert hasattr(model, "fit")
    assert hasattr(model, "predict_proba")
    assert model.random_state == 7
    assert model.class_weight == "balanced_subsample"


def test_xgboost_constructor_is_trainable_and_seeded():
    model = build_xgboost(random_state=7)

    assert hasattr(model, "fit")
    assert hasattr(model, "predict_proba")
    assert model.random_state == 7
    assert model.eval_metric == "aucpr"


def test_fraud_mlp_returns_one_logit_per_sample():
    model = FraudMLP(input_dim=30)
    inputs = torch.zeros((5, 30), dtype=torch.float32)

    logits = model(inputs)

    assert logits.shape == (5,)
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
