# Credit Card Fraud Detection

Machine learning benchmark for fraud detection using the public ULB / Worldline credit-card transaction dataset.

The project compares Logistic Regression, Random Forest, XGBoost, and a compact PyTorch MLP under the same leakage-safe train/validation/test split. Because fraud is extremely rare, PR-AUC is used as the main model-selection metric instead of accuracy.

## Dataset

The dataset contains 284,807 transactions, including 492 fraud cases. Features `V1` to `V28` are anonymized transformed variables, with additional `Time` and `Amount` fields and binary target `Class`.

The raw `creditcard.csv` file is downloaded locally and is intentionally excluded from Git.

```bash
python scripts/download_data.py
```

## Validation benchmark

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **XGBoost** | **0.8072** | **0.9831** | 0.8736 | **0.7755** | 0.8216 |
| Random Forest | 0.8070 | 0.9637 | **0.8929** | 0.7653 | **0.8242** |
| Logistic Regression | 0.6858 | 0.9806 | 0.8605 | 0.7551 | 0.8043 |
| PyTorch MLP | 0.6687 | 0.9693 | 0.8182 | 0.7347 | 0.7742 |

XGBoost is selected because it has the highest validation PR-AUC. Random Forest is essentially tied on PR-AUC and slightly better on F1, so the selection should be read as a narrow decision rather than a dominant win.

## Threshold selection

The XGBoost threshold maximizing validation F1 is approximately `0.9331`.

A separate business-cost scenario uses illustrative assumptions only:

- missed fraud multiplier: `1.0`
- false-positive review cost: `5.0`
- recoverable fraud fraction: `0.0`

Under this scenario, the validation threshold minimizing expected cost is approximately `0.64845`. This lower threshold increases recall by accepting more false positives.

## Final held-out test

The model and business threshold are frozen before evaluating the test set.

| Metric | Test value |
| --- | ---: |
| ROC-AUC | 0.9763 |
| PR-AUC | **0.8557** |
| Precision | 0.7545 |
| Recall | **0.8384** |
| F1 | 0.7943 |
| True positives | 83 |
| False positives | 27 |
| False negatives | 16 |
| True negatives | 56,836 |

The final test result supports the validation conclusion while keeping model and threshold selection separate from the held-out evaluation.

## Explainability

SHAP is used with the selected XGBoost model for global and local explanations. The anonymized components `V1`–`V28` are not assigned invented semantic meanings; the explanations are treated strictly as feature-attribution signals.

## Project structure

```text
credit-card-fraud-detection/
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_baseline_logistic_regression.ipynb
│   ├── 03_ml_benchmark.ipynb
│   ├── 04_explainable_ai.ipynb
│   └── 05_business_impact.ipynb
├── outputs/
│   ├── figures/
│   └── metrics/
├── report/
│   └── findings.md
├── scripts/
│   └── download_data.py
├── src/
│   ├── business_metrics.py
│   ├── data.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── models.py
│   └── preprocessing.py
└── tests/
```

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_data.py
pytest -q
```

Then execute the notebooks in order from `01` to `05`.

## Tools

Python, pandas, NumPy, scikit-learn, XGBoost, PyTorch, SHAP, Matplotlib, Jupyter, pytest.

## Limitations

This is an offline benchmark on a historical anonymized dataset. It does not model concept drift, delayed labels, customer history, production latency, or real operational review costs. The business-cost assumptions are illustrative and should not be interpreted as observed business facts.
