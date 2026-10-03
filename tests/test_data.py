import pandas as pd
import pytest

from src.data import EXPECTED_COLUMNS, load_dataset, split_dataset, validate_dataset


def make_valid_frame(rows: int = 100) -> pd.DataFrame:
    data = {column: [float(index) for index in range(rows)] for column in EXPECTED_COLUMNS}
    data["Class"] = [1 if index % 10 == 0 else 0 for index in range(rows)]
    return pd.DataFrame(data, columns=EXPECTED_COLUMNS)


def test_validate_dataset_accepts_exact_expected_schema():
    validate_dataset(make_valid_frame())


def test_validate_dataset_rejects_missing_columns():
    frame = make_valid_frame().drop(columns=["V28"])

    with pytest.raises(ValueError, match="Missing required columns: V28"):
        validate_dataset(frame)


def test_validate_dataset_rejects_extra_columns():
    frame = make_valid_frame().assign(CustomerId=1)

    with pytest.raises(ValueError, match="Unexpected columns: CustomerId"):
        validate_dataset(frame)


def test_validate_dataset_rejects_non_binary_target():
    frame = make_valid_frame()
    frame.loc[0, "Class"] = 2

    with pytest.raises(ValueError, match="Class must contain only 0 and 1"):
        validate_dataset(frame)


def test_validate_dataset_rejects_missing_target_values():
    frame = make_valid_frame()
    frame.loc[0, "Class"] = pd.NA

    with pytest.raises(ValueError, match="Class must not contain missing values"):
        validate_dataset(frame)


def test_load_dataset_reads_and_validates_csv(tmp_path):
    path = tmp_path / "creditcard.csv"
    frame = make_valid_frame()
    frame.to_csv(path, index=False)

    loaded = load_dataset(path)

    assert list(loaded.columns) == EXPECTED_COLUMNS
    assert loaded.shape == frame.shape


def test_split_dataset_uses_stratified_60_20_20_partitions():
    frame = make_valid_frame()

    X_train, X_validation, X_test, y_train, y_validation, y_test = split_dataset(frame)

    assert (len(X_train), len(X_validation), len(X_test)) == (60, 20, 20)
    assert (len(y_train), len(y_validation), len(y_test)) == (60, 20, 20)
    assert "Class" not in X_train.columns
    assert set(y_train.unique()) == {0, 1}
    assert set(y_validation.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}
