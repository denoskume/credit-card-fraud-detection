# Dataset

This project uses the public **Credit Card Fraud Detection** dataset released for research by the Machine Learning Group of Université Libre de Bruxelles in collaboration with Worldline.

Download `creditcard.csv` from the Kaggle dataset page:

- Dataset: `mlg-ulb/creditcardfraud`

Place the file locally at:

```text
data/creditcard.csv
```

The raw CSV is intentionally excluded from version control. The pipeline expects exactly these columns: `Time`, `V1`–`V28`, `Amount`, and binary target `Class`.
