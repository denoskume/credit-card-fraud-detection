from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import EXPECTED_COLUMNS, load_dataset, split_dataset, validate_dataset


def make_dataset(rows_per_class: int = 10) -> pd.DataFrame:
    rows = rows_per_class * 2
    data = {
        "Time": np.arange(rows, dtype=float),
        **{f"V{i}": np.linspace(i, i + 1, rows) for i in range(1, 29)},
        "Amount": np.linspace(1.0, 100.0, rows),
        "Class": np.array([0] * rows_per_class + [1] * rows_per_class),
    }
    return pd.DataFrame(data, columns=EXPECTED_COLUMNS)


def test_validate_dataset_accepts_exact_expected_schema():
    validate_dataset(make_dataset())


def test_validate_dataset_rejects_missing_column():
    frame = make_dataset().drop(columns=["V28"])

    with pytest.raises(ValueError, match="schema"):
        validate_dataset(frame)


def test_validate_dataset_rejects_extra_column():
    frame = make_dataset()
    frame["Unexpected"] = 1

    with pytest.raises(ValueError, match="schema"):
        validate_dataset(frame)


def test_validate_dataset_rejects_non_binary_target():
    frame = make_dataset()
    frame.loc[0, "Class"] = 2

    with pytest.raises(ValueError, match="binary"):
        validate_dataset(frame)


def test_validate_dataset_rejects_missing_target_value():
    frame = make_dataset()
    frame.loc[0, "Class"] = np.nan

    with pytest.raises(ValueError, match="missing"):
        validate_dataset(frame)


def test_load_dataset_reads_and_validates_csv(tmp_path: Path):
    source = tmp_path / "creditcard.csv"
    expected = make_dataset()
    expected.to_csv(source, index=False)

    loaded = load_dataset(source)

    pd.testing.assert_frame_equal(loaded, expected)


def test_split_dataset_preserves_both_classes_in_all_partitions():
    frame = make_dataset(rows_per_class=20)

    X_train, X_validation, X_test, y_train, y_validation, y_test = split_dataset(frame)

    assert len(X_train) == 24
    assert len(X_validation) == 8
    assert len(X_test) == 8
    assert set(y_train.unique()) == {0, 1}
    assert set(y_validation.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}
    assert "Class" not in X_train.columns
    assert "Class" not in X_validation.columns
    assert "Class" not in X_test.columns
