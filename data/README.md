# Data

This project uses the public ULB / Worldline credit card fraud dataset commonly distributed as `creditcard.csv`.

Expected file:

```text
data/creditcard.csv
```

The dataset contains anonymized transaction features `V1` to `V28`, plus `Time`, `Amount`, and the binary target `Class`.

- `Class = 0`: legitimate transaction
- `Class = 1`: fraudulent transaction

The raw CSV is intentionally excluded from Git. Download the dataset from its public source and place `creditcard.csv` in this directory before running the analysis.

The project validates the expected schema before any model is trained.
