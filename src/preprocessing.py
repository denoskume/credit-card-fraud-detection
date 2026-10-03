import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


SCALED_FEATURES = ["Time", "Amount"]


def build_linear_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    missing = [feature for feature in SCALED_FEATURES if feature not in feature_names]
    if missing:
        raise ValueError(f"Missing features required for scaling: {', '.join(missing)}")

    passthrough_features = [feature for feature in feature_names if feature not in SCALED_FEATURES]
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
        transformers=[("features", "passthrough", feature_names)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_transform_splits(
    preprocessor,
    X_train,
    X_validation,
    X_test,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    train = np.asarray(preprocessor.fit_transform(X_train), dtype=float)
    validation = np.asarray(preprocessor.transform(X_validation), dtype=float)
    test = np.asarray(preprocessor.transform(X_test), dtype=float)
    return train, validation, test, preprocessor
