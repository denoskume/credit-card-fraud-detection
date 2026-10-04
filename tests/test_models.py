import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.models import (
    FraudMLP,
    build_logistic_regression,
    build_random_forest,
    build_xgboost,
)


def test_logistic_regression_constructor_returns_configured_estimator():
    model = build_logistic_regression(random_state=17)

    assert isinstance(model, LogisticRegression)
    assert model.class_weight == "balanced"
    assert model.random_state == 17
    assert model.max_iter >= 1000


def test_random_forest_constructor_returns_configured_estimator():
    model = build_random_forest(random_state=17)

    assert isinstance(model, RandomForestClassifier)
    assert model.class_weight == "balanced"
    assert model.random_state == 17
    assert model.n_estimators >= 100


def test_xgboost_constructor_returns_probability_classifier():
    model = build_xgboost(random_state=17)

    assert isinstance(model, XGBClassifier)
    assert model.random_state == 17
    assert model.eval_metric == "aucpr"
    assert model.objective == "binary:logistic"


def test_fraud_mlp_returns_one_logit_per_transaction():
    model = FraudMLP(input_dim=30)
    transaction_batch = torch.randn(8, 30)

    fraud_logits = model(transaction_batch)

    assert fraud_logits.shape == (8,)
    assert all(isinstance(layer, torch.nn.Module) for layer in model.network)
