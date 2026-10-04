# Findings

## Problem

This project evaluates fraud detection on the public ULB / Worldline credit-card transaction dataset. The task is highly imbalanced, so model selection is based primarily on precision-recall performance rather than accuracy.

The workflow uses a stratified 60/20/20 train, validation, and test split. Preprocessing is fitted on training data only. Model and threshold decisions use validation data; the held-out test set is reserved for the final evaluation.

## Validation benchmark

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 0.8072 | 0.9831 | 0.8736 | 0.7755 | 0.8216 |
| Random Forest | 0.8070 | 0.9637 | 0.8929 | 0.7653 | 0.8242 |
| Logistic Regression | 0.6858 | 0.9806 | 0.8605 | 0.7551 | 0.8043 |
| PyTorch MLP | 0.6687 | 0.9693 | 0.8182 | 0.7347 | 0.7742 |

XGBoost is selected because it has the highest validation PR-AUC, although Random Forest is effectively tied on that metric and has a slightly higher validation F1 score. The result should therefore be interpreted as a narrow model-selection decision rather than a large performance advantage.

## Threshold selection

The validation-selected F1 threshold for XGBoost is approximately `0.9331`. A separate business-cost scenario is then evaluated using explicit illustrative assumptions:

- missed fraud multiplier: `1.0`
- false-positive review cost: `5.0`
- recoverable fraud fraction: `0.0`

These values are scenario assumptions, not observed operational costs.

Under this scenario, the validation threshold minimizing expected cost is approximately `0.64845`. At that threshold, validation recall increases to `0.8163`, with precision `0.7018` and F1 `0.7547`. This illustrates the expected trade-off: a lower threshold detects more fraud but creates more false positives.

## Final held-out test result

The selected XGBoost model and validation-derived business threshold are evaluated once on the held-out test split.

| Metric | Test value |
| --- | ---: |
| ROC-AUC | 0.9763 |
| PR-AUC | 0.8557 |
| Precision | 0.7545 |
| Recall | 0.8384 |
| F1 | 0.7943 |
| True positives | 83 |
| False positives | 27 |
| False negatives | 16 |
| True negatives | 56,836 |

The final test result is consistent with the validation conclusions. The model retains strong precision-recall discrimination and detects most fraudulent transactions in the held-out split under the selected cost scenario.

## Explainability

SHAP is used with the selected XGBoost model for global and local explanations. Because `V1` through `V28` are anonymized transformed components, the project does not assign invented real-world meanings to them. SHAP values are interpreted only as model-attribution signals for the available features.

## Limitations

The dataset is historical and anonymized. The project does not model concept drift, delayed labels, transaction sequences, customer history, or production latency. The business-cost analysis is illustrative because real fraud-loss recovery rates and manual-review costs are not available. The final test result therefore demonstrates a reproducible offline benchmark, not production readiness.
