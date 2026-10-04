import numpy as np
import pandas as pd

from src.preprocessing import (
    build_linear_preprocessor,
    build_tree_preprocessor,
    fit_transform_splits,
)


def make_features(offset: float = 0.0, rows: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time": np.arange(rows, dtype=float) + offset,
            **{f"V{i}": np.linspace(i, i + 1, rows) for i in range(1, 29)},
            "Amount": np.linspace(10.0, 100.0, rows) + offset,
        }
    )


def test_linear_preprocessor_returns_finite_equal_width_splits():
    train = make_features()
    validation = make_features(offset=1000.0, rows=6)
    test = make_features(offset=2000.0, rows=6)

    transformed_train, transformed_validation, transformed_test, _ = fit_transform_splits(
        build_linear_preprocessor(list(train.columns)), train, validation, test
    )

    assert transformed_train.shape[1] == 30
    assert transformed_validation.shape[1] == 30
    assert transformed_test.shape[1] == 30
    assert np.isfinite(transformed_train).all()
    assert np.isfinite(transformed_validation).all()
    assert np.isfinite(transformed_test).all()


def test_validation_and_test_transforms_do_not_refit_scaler():
    train = make_features()
    validation = make_features(offset=1000.0, rows=6)
    test = make_features(offset=2000.0, rows=6)

    _, _, _, fitted = fit_transform_splits(
        build_linear_preprocessor(list(train.columns)), train, validation, test
    )
    scaler = fitted.named_transformers_["scale"]

    np.testing.assert_allclose(scaler.mean_, train[["Time", "Amount"]].mean().to_numpy())


def test_tree_preprocessor_preserves_numeric_values():
    train = make_features()
    validation = make_features(offset=50.0, rows=6)
    test = make_features(offset=100.0, rows=6)

    transformed_train, transformed_validation, transformed_test, _ = fit_transform_splits(
        build_tree_preprocessor(list(train.columns)), train, validation, test
    )

    np.testing.assert_allclose(transformed_train, train.to_numpy())
    np.testing.assert_allclose(transformed_validation, validation.to_numpy())
    np.testing.assert_allclose(transformed_test, test.to_numpy())
