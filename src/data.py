from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


EXPECTED_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]


def validate_dataset(df: pd.DataFrame) -> None:
    missing = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    unexpected = [column for column in df.columns if column not in EXPECTED_COLUMNS]

    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"Unexpected columns: {', '.join(unexpected)}")
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError("Dataset columns must match the expected order")
    if df["Class"].isna().any():
        raise ValueError("Class must not contain missing values")
    if not set(df["Class"].unique()).issubset({0, 1}):
        raise ValueError("Class must contain only 0 and 1")


def load_dataset(path: str | Path) -> pd.DataFrame:
    dataset = pd.read_csv(path)
    validate_dataset(dataset)
    return dataset


def split_dataset(
    df: pd.DataFrame,
    random_state: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    validate_dataset(df)

    features = df.drop(columns="Class")
    target = df["Class"]

    X_train, X_remaining, y_train, y_remaining = train_test_split(
        features,
        target,
        test_size=0.40,
        stratify=target,
        random_state=random_state,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.50,
        stratify=y_remaining,
        random_state=random_state,
    )

    return X_train, X_validation, X_test, y_train, y_validation, y_test
