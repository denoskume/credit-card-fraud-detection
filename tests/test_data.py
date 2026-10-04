import pandas as pd
import pytest

from src.data import EXPECTED_COLUMNS, load_dataset, split_dataset, validate_dataset


def make_valid_dataset(row_count: int = 100) -> pd.DataFrame:
    transaction_data = {
        "Time": range(row_count),
        **{f"V{index}": [float(index)] * row_count for index in range(1, 29)},
        "Amount": [25.0] * row_count,
        "Class": [0] * (row_count - 10) + [1] * 10,
    }
    return pd.DataFrame(transaction_data, columns=EXPECTED_COLUMNS)


def test_validate_dataset_accepts_expected_schema():
    transaction_dataset = make_valid_dataset()
    validate_dataset(transaction_dataset)


def test_validate_dataset_rejects_missing_column():
    transaction_dataset = make_valid_dataset().drop(columns="V28")

    with pytest.raises(ValueError, match="schema"):
        validate_dataset(transaction_dataset)


def test_validate_dataset_rejects_extra_column():
    transaction_dataset = make_valid_dataset()
    transaction_dataset["Customer ID"] = range(len(transaction_dataset))

    with pytest.raises(ValueError, match="schema"):
        validate_dataset(transaction_dataset)


def test_validate_dataset_rejects_non_binary_target():
    transaction_dataset = make_valid_dataset()
    transaction_dataset.loc[0, "Class"] = 2

    with pytest.raises(ValueError, match="Class"):
        validate_dataset(transaction_dataset)


def test_validate_dataset_rejects_missing_target():
    transaction_dataset = make_valid_dataset()
    transaction_dataset.loc[0, "Class"] = pd.NA

    with pytest.raises(ValueError, match="Class"):
        validate_dataset(transaction_dataset)


def test_split_dataset_preserves_both_classes_and_expected_sizes():
    transaction_dataset = make_valid_dataset()

    (
        training_features,
        validation_features,
        test_features,
        training_labels,
        validation_labels,
        test_labels,
    ) = split_dataset(transaction_dataset, random_state=42)

    assert len(training_features) == 60
    assert len(validation_features) == 20
    assert len(test_features) == 20
    assert set(training_labels.unique()) == {0, 1}
    assert set(validation_labels.unique()) == {0, 1}
    assert set(test_labels.unique()) == {0, 1}
    assert "Class" not in training_features.columns
    assert len(training_features) == len(training_labels)
    assert len(validation_features) == len(validation_labels)
    assert len(test_features) == len(test_labels)


def test_load_dataset_reads_and_validates_csv(tmp_path):
    transaction_dataset = make_valid_dataset()
    dataset_path = tmp_path / "creditcard.csv"
    transaction_dataset.to_csv(dataset_path, index=False)

    loaded_dataset = load_dataset(dataset_path)

    pd.testing.assert_frame_equal(loaded_dataset, transaction_dataset)
