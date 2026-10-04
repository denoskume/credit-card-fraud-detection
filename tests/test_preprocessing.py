import numpy as np
import pandas as pd

from src.preprocessing import (
    build_linear_preprocessor,
    build_tree_preprocessor,
    fit_transform_splits,
)


def make_feature_splits():
    training_features = pd.DataFrame({
        "Time": [0.0, 100.0, 200.0, 300.0],
        "V1": [0.1, 0.2, 0.3, 0.4],
        "V2": [1.0, 1.5, 2.0, 2.5],
        "Amount": [10.0, 20.0, 30.0, 40.0],
    })
    validation_features = pd.DataFrame({
        "Time": [10_000.0, 20_000.0],
        "V1": [0.5, 0.6],
        "V2": [3.0, 3.5],
        "Amount": [500.0, 600.0],
    })
    test_features = pd.DataFrame({
        "Time": [30_000.0, 40_000.0],
        "V1": [0.7, 0.8],
        "V2": [4.0, 4.5],
        "Amount": [700.0, 800.0],
    })
    return training_features, validation_features, test_features


def test_linear_preprocessor_returns_finite_equal_width_splits():
    training_features, validation_features, test_features = make_feature_splits()
    preprocessor = build_linear_preprocessor(training_features.columns.tolist())

    (
        transformed_training_features,
        transformed_validation_features,
        transformed_test_features,
        fitted_preprocessor,
    ) = fit_transform_splits(
        preprocessor,
        training_features,
        validation_features,
        test_features,
    )

    expected_width = training_features.shape[1]
    assert transformed_training_features.shape == (4, expected_width)
    assert transformed_validation_features.shape == (2, expected_width)
    assert transformed_test_features.shape == (2, expected_width)
    assert np.isfinite(transformed_training_features).all()
    assert np.isfinite(transformed_validation_features).all()
    assert np.isfinite(transformed_test_features).all()
    assert fitted_preprocessor is preprocessor


def test_validation_and_test_transforms_do_not_refit_scaler():
    training_features, validation_features, test_features = make_feature_splits()
    preprocessor = build_linear_preprocessor(training_features.columns.tolist())

    preprocessor.fit(training_features)
    training_scaler_mean = preprocessor.named_transformers_["scaled"].mean_.copy()

    preprocessor.transform(validation_features)
    preprocessor.transform(test_features)

    np.testing.assert_array_equal(
        preprocessor.named_transformers_["scaled"].mean_,
        training_scaler_mean,
    )


def test_linear_preprocessor_scales_only_time_and_amount():
    training_features, _, _ = make_feature_splits()
    preprocessor = build_linear_preprocessor(training_features.columns.tolist())
    transformed_training_features = preprocessor.fit_transform(training_features)

    np.testing.assert_allclose(
        transformed_training_features[:, :2].mean(axis=0),
        np.zeros(2),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        transformed_training_features[:, 2:],
        training_features[["V1", "V2"]].to_numpy(),
    )


def test_tree_preprocessor_preserves_feature_values():
    training_features, validation_features, test_features = make_feature_splits()
    preprocessor = build_tree_preprocessor(training_features.columns.tolist())

    (
        transformed_training_features,
        transformed_validation_features,
        transformed_test_features,
        _,
    ) = fit_transform_splits(
        preprocessor,
        training_features,
        validation_features,
        test_features,
    )

    np.testing.assert_allclose(
        transformed_training_features,
        training_features.to_numpy(),
    )
    np.testing.assert_allclose(
        transformed_validation_features,
        validation_features.to_numpy(),
    )
    np.testing.assert_allclose(
        transformed_test_features,
        test_features.to_numpy(),
    )
