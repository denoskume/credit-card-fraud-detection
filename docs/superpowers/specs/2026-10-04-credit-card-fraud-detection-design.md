# Credit Card Fraud Detection — Project Design Specification

## 1. Project Goal

Build a reproducible machine-learning project for highly imbalanced credit-card fraud detection using the public ULB / Worldline dataset. The project must demonstrate a complete applied ML workflow suitable for an MSc-level portfolio: data understanding, robust preprocessing, baseline modelling, multi-model benchmarking, threshold-aware evaluation, explainability, and business-impact analysis.

The project is intended to show practical Machine Learning Engineering and Data Science ability without overstating production experience.

## 2. Core Question

Which model detects fraudulent transactions most effectively while controlling the operational and financial cost of false positives and false negatives?

## 3. Dataset

Primary dataset: ULB / Worldline Credit Card Fraud Detection dataset.

Expected characteristics:
- 284,807 transactions.
- 492 fraud cases.
- Binary target: `Class` (`1` = fraud, `0` = legitimate).
- Features `V1` to `V28` are anonymized PCA-transformed variables.
- `Time` and `Amount` remain interpretable original variables.
- Strong class imbalance requires evaluation beyond raw accuracy.

The raw dataset will not be committed to GitHub. `data/README.md` will explain how to obtain it and where to place it locally.

## 4. Scope

### Included

- Dataset validation and exploratory analysis.
- Train / validation / test split with stratification.
- Leakage-safe preprocessing.
- Logistic Regression baseline.
- Random Forest.
- XGBoost.
- Feed-forward neural network.
- Common model-evaluation framework.
- ROC-AUC and PR-AUC.
- Precision, Recall and F1-score.
- Confusion matrices.
- Threshold analysis.
- SHAP-based explainability for the strongest suitable model.
- Business-cost model for false negatives and false positives.
- Reproducible outputs and concise final findings.
- Essential automated tests for pipeline integrity.

### Excluded

- Real banking deployment.
- Real-time fraud scoring service.
- Cloud infrastructure.
- Streaming architecture.
- Customer-identifiable data.
- Claims that the model is production-ready.
- Synthetic performance numbers.

These exclusions keep the project credible, focused, and aligned with a student portfolio.

## 5. Repository Architecture

```text
credit-card-fraud-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_baseline_logistic_regression.ipynb
│   ├── 03_ml_benchmark.ipynb
│   ├── 04_explainable_ai.ipynb
│   └── 05_business_impact.ipynb
│
├── src/
│   ├── data.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   ├── explainability.py
│   └── business_metrics.py
│
├── outputs/
│   ├── figures/
│   └── metrics/
│
├── tests/
│   └── test_pipeline.py
│
├── report/
│   └── findings.md
│
└── docs/
    └── superpowers/
        └── specs/
```

## 6. Data Flow

1. Load the local CSV through `src/data.py`.
2. Validate expected schema and target values.
3. Inspect imbalance, amount distribution, missing values and duplicates.
4. Create stratified train / validation / test partitions.
5. Fit preprocessing only on the training partition.
6. Train all candidate models using the same data split.
7. Produce predicted probabilities, not only hard labels.
8. Evaluate all models with the same metric functions.
9. Select the strongest candidate using PR-AUC as the primary discrimination metric, supported by ROC-AUC, recall, precision and F1.
10. Analyze decision thresholds on validation data.
11. Freeze the selected threshold before final test evaluation.
12. Explain the selected model with SHAP where technically appropriate.
13. Convert prediction errors into an explicit business-cost analysis.
14. Export final metrics, figures and written findings.

## 7. Preprocessing Design

Preprocessing must be model-aware but leakage-safe.

### Shared rules

- Validate that `Class` contains only 0 and 1.
- Separate features and target before fitting transformations.
- Use stratified splitting because fraud prevalence is extremely low.
- Keep the held-out test set untouched until final evaluation.
- Set explicit random seeds for reproducibility.

### Scaling

`Amount` and `Time` will be scaled where required, especially for Logistic Regression and the neural network. PCA-derived features will not be transformed unnecessarily unless the selected modelling pipeline benefits from consistent scaling.

Tree models do not require feature scaling, so their preprocessing should remain minimal.

### Class imbalance

The first benchmark will preserve the natural class distribution and use model-native class weighting where appropriate.

Resampling techniques such as SMOTE will not be introduced by default. They may be tested only if they provide a justified improvement without contaminating validation/test data. Any resampling must occur inside the training pipeline only.

## 8. Models

### Logistic Regression

Purpose: interpretable baseline and reference point for more complex models.

Expected configuration:
- scaled numerical inputs,
- class weighting,
- probability predictions,
- modest regularization tuning.

### Random Forest

Purpose: nonlinear tree-based reference.

Focus:
- class weighting,
- number and depth of trees,
- generalization rather than maximal training score.

### XGBoost

Purpose: primary gradient-boosting candidate for tabular fraud detection.

Focus:
- imbalance-aware weighting,
- controlled tree complexity,
- learning rate,
- early stopping where applicable,
- validation PR-AUC.

### Neural Network

Purpose: compare a compact deep-learning model with classical tabular methods.

Design:
- small fully connected architecture,
- normalized inputs,
- dropout and/or regularization where useful,
- class-weighted loss,
- early stopping.

The neural network will remain intentionally compact. Complexity must be justified by evidence rather than portfolio appearance.

## 9. Evaluation Framework

Accuracy is not a primary metric because a trivial classifier predicting every transaction as legitimate would appear highly accurate.

### Primary metric

**PR-AUC (Average Precision)**

This is the main comparison metric because it focuses on performance for the rare positive class and is more informative under extreme imbalance.

### Secondary metrics

- ROC-AUC
- Precision
- Recall
- F1-score
- Confusion matrix

### Threshold-aware evaluation

The default probability threshold of 0.5 is not assumed to be optimal.

The validation set will be used to examine:
- precision-recall trade-off,
- F1 across thresholds,
- false-negative count,
- false-positive count,
- estimated business cost.

The final threshold must be selected before evaluating the test set.

## 10. Explainability

SHAP will be the primary explainability technique.

The goal is not to claim causal interpretation. The analysis will explain how input features influence model predictions within the trained model.

Planned outputs:
- global feature-importance view,
- SHAP summary plot,
- selected local explanations for representative fraud / legitimate cases where useful.

Because `V1`–`V28` are anonymized PCA components, interpretation will be framed carefully. The project will not invent semantic meanings for anonymized components.

## 11. Business-Impact Analysis

A technically superior classifier is not automatically the best operational model.

The business layer will make assumptions explicit rather than presenting invented financial facts.

### Core quantities

- False negative: fraudulent transaction not detected.
- False positive: legitimate transaction incorrectly flagged.
- True positive: fraud detected.
- True negative: legitimate transaction allowed.

### Cost model

The project will expose configurable assumptions such as:

- cost of a missed fraud,
- investigation / review cost of a flagged legitimate transaction,
- optional fraction of fraudulent transaction value considered recoverable.

A generic expected-cost function will compare models and thresholds under the same assumptions.

Where transaction `Amount` is used, it will be clearly distinguished from assumed operational costs.

The final report will include sensitivity analysis so conclusions are not tied to one arbitrary cost setting.

## 12. Notebook Responsibilities

### `01_data_understanding.ipynb`

- dataset validation,
- class distribution,
- missing values / duplicates,
- transaction amount analysis,
- concise EDA,
- documented modelling implications.

### `02_baseline_logistic_regression.ipynb`

- train / validation / test strategy,
- preprocessing,
- Logistic Regression baseline,
- baseline metrics,
- baseline interpretation.

### `03_ml_benchmark.ipynb`

- Random Forest,
- XGBoost,
- neural network,
- standardized metric comparison,
- ROC and precision-recall curves,
- model-selection decision.

### `04_explainable_ai.ipynb`

- SHAP analysis of the selected suitable model,
- global explanation,
- representative local cases,
- interpretation limitations.

### `05_business_impact.ipynb`

- threshold analysis,
- error-cost assumptions,
- model / threshold cost comparison,
- sensitivity analysis,
- final recommendation.

Notebooks will orchestrate analysis and presentation. Reusable logic belongs in `src/` rather than being duplicated across notebooks.

## 13. Source Modules

### `src/data.py`

Responsibilities:
- load dataset,
- validate columns,
- validate binary target,
- return clean feature / target structures.

### `src/preprocessing.py`

Responsibilities:
- construct leakage-safe preprocessing pipelines,
- scale only where required,
- centralize train-time transformations.

### `src/models.py`

Responsibilities:
- construct model configurations,
- centralize random seeds,
- keep model creation reproducible.

### `src/evaluation.py`

Responsibilities:
- compute common metrics,
- generate threshold statistics,
- support ROC / PR curves,
- serialize comparison results.

### `src/explainability.py`

Responsibilities:
- SHAP explainer setup,
- global and local explanation helpers.

### `src/business_metrics.py`

Responsibilities:
- compute confusion-derived business costs,
- accept explicit cost assumptions,
- support threshold / cost sensitivity comparisons.

## 14. Outputs

Generated figures will be stored in `outputs/figures/` and machine-readable metric tables in `outputs/metrics/`.

Expected figures include:
- class distribution,
- amount distribution by class,
- ROC curves,
- precision-recall curves,
- confusion matrix for final candidate,
- threshold-performance curve,
- SHAP summary,
- business-cost comparison.

Generated outputs must correspond to actual executed analysis. No placeholder results will be committed as final evidence.

## 15. Testing Strategy

`tests/test_pipeline.py` will focus on high-value checks rather than excessive unit-test volume.

Required checks:
- expected dataset schema validation,
- target validation,
- preprocessing produces finite numeric data,
- train/test transformation does not refit on test data,
- metric functions return expected keys and valid ranges,
- business-cost function produces known values on a synthetic confusion example,
- model constructors return trainable estimators.

Notebook execution and final output consistency will also be checked before declaring the project complete.

## 16. Reproducibility

The repository must provide:
- Python dependency list,
- fixed random seeds,
- dataset acquisition instructions,
- deterministic split definition where practical,
- commands for running tests,
- clear notebook execution order.

Raw credit-card data will remain outside version control.

## 17. README Design

The final `README.md` will be concise and recruiter-friendly.

It will contain:
- project title,
- short problem statement,
- dataset summary,
- methodological pipeline,
- benchmark table populated only with real results,
- explainability summary,
- business-impact conclusion,
- repository structure,
- reproducibility instructions,
- technologies,
- limitations.

The README will avoid exaggerated language such as “production-grade”, “enterprise-ready”, or unsupported claims of expertise.

## 18. Technical Stack

Primary tools:
- Python
- NumPy
- pandas
- scikit-learn
- XGBoost
- PyTorch or TensorFlow/Keras for the compact neural network
- SHAP
- Matplotlib
- Jupyter
- pytest

The implementation should prefer libraries already coherent with the user's existing ML portfolio. PyTorch is preferred for the neural network unless a concrete implementation reason favors Keras.

## 19. Success Criteria

The project is complete only when all of the following are true:

1. Dataset loading and validation are reproducible.
2. The split strategy prevents obvious target/data leakage.
3. Logistic Regression, Random Forest, XGBoost and the neural network are evaluated on the same held-out framework.
4. PR-AUC is used as the principal comparison metric.
5. Threshold choice is justified rather than assumed.
6. Final performance is measured once on the held-out test set after model / threshold decisions.
7. SHAP explanations are generated without assigning fabricated semantic meaning to PCA features.
8. Business-cost conclusions expose their assumptions.
9. All published metrics and figures come from executed code.
10. Essential tests pass.
11. README and findings are concise, natural, technically accurate and consistent with outputs.
12. No raw dataset or secrets are committed.

## 20. Portfolio Outcome

The completed project should demonstrate the following abilities naturally:

- working with severely imbalanced classification data,
- building and comparing classical ML and neural-network models,
- selecting appropriate evaluation metrics,
- reasoning about prediction thresholds,
- applying explainability methods responsibly,
- connecting model errors to business decisions,
- structuring reproducible ML code and notebooks,
- communicating findings without overstating experience.

The portfolio narrative is therefore broader than “fraud classifier”: it demonstrates an end-to-end decision-oriented ML workflow.