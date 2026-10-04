import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.explainability import build_shap_explainer, compute_shap_values


def test_shap_helpers_return_finite_sample_feature_matrix():
    training_features = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.2, 0.1],
            [0.9, 0.8],
        ]
    )
    training_labels = np.array([0, 0, 0, 1, 0, 1])
    explained_samples = training_features[:3]

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(training_features, training_labels)

    explainer = build_shap_explainer(model, training_features)
    shap_values = compute_shap_values(explainer, explained_samples)

    assert shap_values.shape == explained_samples.shape
    assert np.isfinite(shap_values).all()


def test_xgboost_shap_helpers_return_finite_sample_feature_matrix():
    training_features = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.2, 0.1],
            [0.9, 0.8],
        ]
    )
    training_labels = np.array([0, 0, 0, 1, 0, 1])
    explained_samples = training_features[:3]

    model = XGBClassifier(
        n_estimators=5,
        max_depth=2,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(training_features, training_labels)

    explainer = build_shap_explainer(model, training_features)
    shap_values = compute_shap_values(explainer, explained_samples)

    assert shap_values.shape == explained_samples.shape
    assert np.isfinite(shap_values).all()
