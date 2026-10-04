import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

SCALED_FEATURES = ["Time", "Amount"]


def build_linear_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    passthrough_features = [
        feature_name
        for feature_name in feature_names
        if feature_name not in SCALED_FEATURES
    ]

    return ColumnTransformer(
        transformers=[
            ("scaled", StandardScaler(), SCALED_FEATURES),
            ("passthrough", "passthrough", passthrough_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_tree_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[("passthrough", "passthrough", feature_names)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_transform_splits(
    preprocessor,
    training_features,
    validation_features,
    test_features,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    transformed_training_features = preprocessor.fit_transform(training_features)
    transformed_validation_features = preprocessor.transform(validation_features)
    transformed_test_features = preprocessor.transform(test_features)

    return (
        np.asarray(transformed_training_features, dtype=float),
        np.asarray(transformed_validation_features, dtype=float),
        np.asarray(transformed_test_features, dtype=float),
        preprocessor,
    )
