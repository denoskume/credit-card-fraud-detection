import numpy as np
import pandas as pd

from src.preprocessing import (
    build_linear_preprocessor,
    build_tree_preprocessor,
    fit_transform_splits,
)


FEATURE_NAMES = ["Time", "V1", "V2", "Amount"]


def make_split(offset: float, rows: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time": np.arange(rows, dtype=float) + offset,
            "V1": np.linspace(-1.0, 1.0, rows) + offset,
            "V2": np.linspace(1.0, 2.0, rows) - offset,
            "Amount": np.linspace(10.0, 60.0, rows) + offset,
        }
    )


def test_linear_preprocessor_returns_finite_equal_width_splits():
    preprocessor = build_linear_preprocessor(FEATURE_NAMES)

    train, validation, test, fitted = fit_transform_splits(
        preprocessor,
        make_split(0.0),
        make_split(100.0),
        make_split(200.0),
    )

    assert train.shape[1] == validation.shape[1] == test.shape[1] == len(FEATURE_NAMES)
    assert np.isfinite(train).all()
    assert np.isfinite(validation).all()
    assert np.isfinite(test).all()
    assert fitted is preprocessor


def test_validation_and_test_transforms_do_not_refit_scaler():
    preprocessor = build_linear_preprocessor(FEATURE_NAMES)
    X_train = make_split(0.0)
    X_validation = make_split(100.0)
    X_test = make_split(200.0)

    _, _, _, fitted = fit_transform_splits(preprocessor, X_train, X_validation, X_test)
    scaler = fitted.named_transformers_["scaled"]

    assert np.allclose(scaler.mean_, X_train[["Time", "Amount"]].mean().to_numpy())
    assert not np.allclose(scaler.mean_, X_validation[["Time", "Amount"]].mean().to_numpy())
    assert not np.allclose(scaler.mean_, X_test[["Time", "Amount"]].mean().to_numpy())


def test_tree_preprocessor_preserves_feature_values():
    X_train = make_split(0.0)
    preprocessor = build_tree_preprocessor(FEATURE_NAMES)

    train, _, _, _ = fit_transform_splits(preprocessor, X_train, X_train, X_train)

    assert np.allclose(train, X_train[FEATURE_NAMES].to_numpy())
