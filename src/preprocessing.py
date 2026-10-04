import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


def build_linear_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    scaled_features = [feature for feature in ["Time", "Amount"] if feature in feature_names]
    passthrough_features = [feature for feature in feature_names if feature not in scaled_features]

    return ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), scaled_features),
            ("passthrough", "passthrough", passthrough_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_tree_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[("features", "passthrough", feature_names)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_transform_splits(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ColumnTransformer]:
    transformed_train = np.asarray(preprocessor.fit_transform(X_train), dtype=float)
    transformed_validation = np.asarray(preprocessor.transform(X_validation), dtype=float)
    transformed_test = np.asarray(preprocessor.transform(X_test), dtype=float)

    return transformed_train, transformed_validation, transformed_test, preprocessor
