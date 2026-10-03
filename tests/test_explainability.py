import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.explainability import build_shap_explainer, positive_class_explanation


def make_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 4))
    y = np.array([0] * 20 + [1] * 10)
    return X, y


def test_linear_shap_explainer_matches_input_shape():
    X, y = make_data()
    model = LogisticRegression(max_iter=500).fit(X, y)
    explainer = build_shap_explainer(model, X[:10])

    explanation = positive_class_explanation(explainer(X[:5]))

    assert explanation.values.shape == (5, 4)
    assert np.isfinite(explanation.values).all()


def test_tree_shap_explainer_returns_positive_class_matrix():
    X, y = make_data()
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    explainer = build_shap_explainer(model, X[:10])

    explanation = positive_class_explanation(explainer(X[:5]))

    assert explanation.values.shape == (5, 4)
    assert np.isfinite(explanation.values).all()
