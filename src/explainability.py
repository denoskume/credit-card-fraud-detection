import numpy as np
import shap


def build_shap_explainer(model, background):
    background_array = np.asarray(background, dtype=float)
    if background_array.ndim != 2 or background_array.shape[0] == 0:
        raise ValueError("background must be a non-empty two-dimensional array")
    if not np.isfinite(background_array).all():
        raise ValueError("background must contain only finite values")
    return shap.Explainer(model, background_array)


def positive_class_explanation(explanation: shap.Explanation) -> shap.Explanation:
    values = np.asarray(explanation.values)
    if values.ndim == 2:
        return explanation
    if values.ndim == 3 and values.shape[-1] == 2:
        return explanation[..., 1]
    raise ValueError(f"Unsupported SHAP explanation shape: {values.shape}")
