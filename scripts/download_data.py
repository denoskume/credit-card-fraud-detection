from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

OPENML_DATASET_ID = 1597
OPENML_ARFF_URL = "https://openml.org/data/v1/download/1673544/creditcard.arff"
EXPECTED_COLUMNS = [
    "Time",
    *[f"V{index}" for index in range(1, 29)],
    "Amount",
    "Class",
]
EXPECTED_ROWS = 284_807
EXPECTED_FRAUDS = 492


def convert_arff_to_csv(
    arff_path: Path,
    csv_path: Path,
    expected_columns: list[str] = EXPECTED_COLUMNS,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    data_section_started = False

    with arff_path.open("r", encoding="utf-8") as source_file, csv_path.open(
        "w", encoding="utf-8", newline=""
    ) as destination_file:
        destination_file.write(",".join(expected_columns) + "\n")

        for line in source_file:
            stripped_line = line.strip()

            if not data_section_started:
                if stripped_line.lower() == "@data":
                    data_section_started = True
                continue

            if stripped_line and not stripped_line.startswith("%"):
                destination_file.write(stripped_line + "\n")

    if not data_section_started:
        raise ValueError("Downloaded OpenML file does not contain an ARFF data section.")


def normalize_downloaded_dataset(transaction_dataset: pd.DataFrame) -> pd.DataFrame:
    normalized_dataset = transaction_dataset.copy()
    normalized_class = (
        normalized_dataset["Class"]
        .astype(str)
        .str.strip()
        .str.strip("'\"")
    )
    normalized_dataset["Class"] = pd.to_numeric(
        normalized_class,
        errors="raise",
    ).astype(int)
    return normalized_dataset


def validate_downloaded_dataset(transaction_dataset: pd.DataFrame) -> None:
    if transaction_dataset.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError("Downloaded dataset columns do not match the ULB dataset schema.")

    if len(transaction_dataset) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} transactions, found {len(transaction_dataset):,}."
        )

    observed_classes = set(transaction_dataset["Class"].unique())
    if observed_classes != {0, 1}:
        raise ValueError(
            f"Expected binary Class values {{0, 1}}, found {sorted(observed_classes)}."
        )

    fraud_count = int(transaction_dataset["Class"].sum())
    if fraud_count != EXPECTED_FRAUDS:
        raise ValueError(
            f"Expected {EXPECTED_FRAUDS} fraud transactions, found {fraud_count}."
        )


def download_dataset(destination_path: Path) -> Path:
    temporary_arff_path = destination_path.with_suffix(".arff")

    print(f"Downloading real ULB credit-card dataset from OpenML ID {OPENML_DATASET_ID}...")
    urlretrieve(OPENML_ARFF_URL, temporary_arff_path)

    try:
        convert_arff_to_csv(temporary_arff_path, destination_path)
        transaction_dataset = pd.read_csv(destination_path)
        transaction_dataset = normalize_downloaded_dataset(transaction_dataset)
        validate_downloaded_dataset(transaction_dataset)
        transaction_dataset.to_csv(destination_path, index=False)
    except Exception:
        destination_path.unlink(missing_ok=True)
        raise
    finally:
        temporary_arff_path.unlink(missing_ok=True)

    print(f"Saved: {destination_path}")
    print(f"Transactions: {len(transaction_dataset):,}")
    print(f"Frauds: {int(transaction_dataset['Class'].sum()):,}")
    return destination_path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    download_dataset(project_root / "data" / "creditcard.csv")
