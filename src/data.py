from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
EXPECTED_COLUMNS = [
    "Time",
    *[f"V{index}" for index in range(1, 29)],
    "Amount",
    "Class",
]


def validate_dataset(transaction_dataset: pd.DataFrame) -> None:
    observed_columns = list(transaction_dataset.columns)

    if observed_columns != EXPECTED_COLUMNS:
        missing_columns = [
            column_name
            for column_name in EXPECTED_COLUMNS
            if column_name not in observed_columns
        ]
        extra_columns = [
            column_name
            for column_name in observed_columns
            if column_name not in EXPECTED_COLUMNS
        ]
        raise ValueError(
            "Dataset schema does not match the expected credit-card format. "
            f"Missing columns: {missing_columns}. Extra columns: {extra_columns}."
        )

    if transaction_dataset["Class"].isna().any():
        raise ValueError("Class must not contain missing values.")

    observed_classes = set(transaction_dataset["Class"].unique())
    if not observed_classes.issubset({0, 1}):
        raise ValueError("Class must contain only binary values 0 and 1.")

    if observed_classes != {0, 1}:
        raise ValueError("Class must contain both legitimate and fraudulent transactions.")


def load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    transaction_dataset = pd.read_csv(dataset_path)
    validate_dataset(transaction_dataset)
    return transaction_dataset


def split_dataset(
    transaction_dataset: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    validate_dataset(transaction_dataset)

    transaction_features = transaction_dataset.drop(columns="Class")
    fraud_labels = transaction_dataset["Class"].astype(int)

    (
        training_features,
        remaining_features,
        training_labels,
        remaining_labels,
    ) = train_test_split(
        transaction_features,
        fraud_labels,
        test_size=0.40,
        stratify=fraud_labels,
        random_state=random_state,
    )

    (
        validation_features,
        test_features,
        validation_labels,
        test_labels,
    ) = train_test_split(
        remaining_features,
        remaining_labels,
        test_size=0.50,
        stratify=remaining_labels,
        random_state=random_state,
    )

    return (
        training_features.reset_index(drop=True),
        validation_features.reset_index(drop=True),
        test_features.reset_index(drop=True),
        training_labels.reset_index(drop=True),
        validation_labels.reset_index(drop=True),
        test_labels.reset_index(drop=True),
    )
