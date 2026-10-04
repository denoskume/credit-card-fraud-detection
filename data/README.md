# Data

This project uses the public ULB / Worldline credit card fraud dataset released through OpenML as dataset ID `1597`.

The dataset contains 284,807 real credit-card transactions made by European cardholders over two days in September 2013, including 492 fraud cases.

Expected local file:

```text
data/creditcard.csv
```

Download and verify it automatically with:

```bash
python scripts/download_data.py
```

The downloader retrieves the public OpenML copy, converts it to the CSV format expected by the project, and verifies:

- 284,807 rows
- 31 columns
- 492 fraudulent transactions
- columns `Time`, `V1` to `V28`, `Amount`, `Class`

`Class = 0` means a legitimate transaction and `Class = 1` means fraud.

The raw dataset is intentionally excluded from Git and must remain local.
