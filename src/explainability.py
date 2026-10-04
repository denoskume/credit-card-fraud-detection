import numpy as np
import shap
from xgboost import XGBClassifier


def build_shap_explainer(model, background):
    if isinstance(model, XGBClassifier):
        return shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent",
        )

    return shap.Explainer(model, background)


def compute_shap_values(explainer, samples) -> np.ndarray:
    explanation = explainer(samples)
    shap_values = np.asarray(explanation.values, dtype=float)

    if shap_values.ndim == 3:
        if shap_values.shape[-1] == 2:
            shap_values = shap_values[:, :, 1]
        elif shap_values.shape[0] == 2:
            shap_values = shap_values[1]
        else:
            raise ValueError("Unexpected multi-output SHAP value shape.")

    if shap_values.ndim != 2:
        raise ValueError("SHAP values must form a sample-by-feature matrix.")
    if not np.isfinite(shap_values).all():
        raise ValueError("SHAP values must be finite.")

    return shap_values
