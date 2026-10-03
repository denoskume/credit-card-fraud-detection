import pandas as pd
import pytest

from src.data import EXPECTED_COLUMNS, load_dataset, split_features_target


def make_valid_frame() -> pd.DataFrame:
    row = {column: 0.0 for column in EXPECTED_COLUMNS}
    row["Class"] = 0
    return pd.DataFrame([row])


def test_load_dataset_accepts_expected_schema(tmp_path):
    path = tmp_path / "creditcard.csv"
    frame = make_valid_frame()
    frame.to_csv(path, index=False)

    loaded = load_dataset(path)

    assert list(loaded.columns) == EXPECTED_COLUMNS
    assert loaded.shape == (1, 31)


def test_load_dataset_rejects_missing_columns(tmp_path):
    path = tmp_path / "creditcard.csv"
    frame = make_valid_frame().drop(columns=["V28"])
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Missing required columns: V28"):
        load_dataset(path)


def test_load_dataset_rejects_non_binary_target(tmp_path):
    path = tmp_path / "creditcard.csv"
    frame = make_valid_frame()
    frame.loc[0, "Class"] = 2
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Class must contain only 0 and 1"):
        load_dataset(path)


def test_split_features_target_separates_class_column():
    frame = make_valid_frame()

    features, target = split_features_target(frame)

    assert "Class" not in features.columns
    assert target.name == "Class"
    assert target.tolist() == [0]
