from pathlib import Path

import pandas as pd

from scripts.download_data import (
    convert_arff_to_csv,
    normalize_downloaded_dataset,
    validate_downloaded_dataset,
)


def test_convert_arff_to_csv_preserves_expected_columns(tmp_path: Path):
    arff_path = tmp_path / "creditcard.arff"
    csv_path = tmp_path / "creditcard.csv"
    expected_columns = ["Time", "V1", "Amount", "Class"]

    arff_path.write_text(
        "@RELATION creditcard\n"
        "@ATTRIBUTE Time NUMERIC\n"
        "@ATTRIBUTE V1 NUMERIC\n"
        "@ATTRIBUTE Amount NUMERIC\n"
        "@ATTRIBUTE Class {0,1}\n"
        "@DATA\n"
        "0,0.1,25.0,0\n"
        "1,-0.2,100.0,1\n",
        encoding="utf-8",
    )

    convert_arff_to_csv(arff_path, csv_path, expected_columns)

    converted_dataset = pd.read_csv(csv_path)
    assert converted_dataset.columns.tolist() == expected_columns
    assert converted_dataset.shape == (2, 4)
    assert converted_dataset["Class"].tolist() == [0, 1]


def test_normalize_downloaded_dataset_converts_quoted_class_labels():
    transaction_dataset = pd.DataFrame(
        {
            "Time": [0.0, 1.0],
            "V1": [0.1, -0.2],
            "Amount": [25.0, 100.0],
            "Class": ["'0'", "'1'"],
        }
    )

    normalized_dataset = normalize_downloaded_dataset(transaction_dataset)

    assert normalized_dataset["Class"].tolist() == [0, 1]
    assert pd.api.types.is_integer_dtype(normalized_dataset["Class"])


def test_validate_downloaded_dataset_accepts_real_dataset_signature():
    expected_columns = ["Time", *[f"V{index}" for index in range(1, 29)], "Amount", "Class"]
    transaction_dataset = pd.DataFrame(0.0, index=range(284_807), columns=expected_columns)
    transaction_dataset["Class"] = 0
    transaction_dataset.loc[:491, "Class"] = 1

    validate_downloaded_dataset(transaction_dataset)
