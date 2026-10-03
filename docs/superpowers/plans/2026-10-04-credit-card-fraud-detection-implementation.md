# Credit Card Fraud Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible MSc-level credit-card fraud detection benchmark using Logistic Regression, Random Forest, XGBoost, and a compact neural network, with threshold-aware evaluation, SHAP explainability, and explicit business-cost analysis.

**Architecture:** Reusable logic lives in focused `src/` modules, while notebooks orchestrate analysis and presentation. All models use the same leakage-safe data split and common evaluation functions; the final model and threshold are selected on validation data before a single held-out test evaluation.

**Tech Stack:** Python, NumPy, pandas, scikit-learn, XGBoost, PyTorch, SHAP, Matplotlib, Jupyter, pytest.

**Spec:** `docs/superpowers/specs/2026-10-04-credit-card-fraud-detection-design.md`

## Global Constraints

- Raw `creditcard.csv` must never be committed.
- Primary comparison metric: PR-AUC / Average Precision.
- Accuracy is not a primary model-selection metric.
- Use stratified train / validation / test partitions.
- Fit preprocessing only on training data.
- Tune model and threshold choices on validation data only.
- Evaluate the frozen final choice once on held-out test data.
- Use explicit random seeds for reproducibility.
- Do not invent semantic meanings for anonymized PCA components `V1`–`V28`.
- Business-cost assumptions must be configurable and clearly separated from observed `Amount` values.
- Published metrics and figures must come from executed code, never placeholders.
- Prefer PyTorch for the compact neural network.

## Review Focus

- Missing or extra dataset columns must fail with a clear schema error.
- Non-binary or missing target values must fail before training.
- Extremely imbalanced splits must preserve both classes in train, validation, and test.
- Metric functions must reject mismatched labels/probabilities and return finite values.
- Business-cost calculations must behave correctly when one confusion-matrix cell is zero.

---

### Task 1: Repository Foundation and Data Contract

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `data/README.md`
- Create: `src/__init__.py`
- Create: `src/data.py`
- Create: `tests/test_data.py`

**Interfaces:**
- Consumes: local `data/creditcard.csv`.
- Produces: `load_dataset(path: str | Path) -> pd.DataFrame`, `validate_dataset(df: pd.DataFrame) -> None`, `split_dataset(df: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]`.

- [ ] **Step 1: Write failing tests** for exact expected columns, binary `Class`, missing target values, missing columns, and class-preserving splits.
- [ ] **Step 2: Run** `pytest tests/test_data.py -v` and verify failure because `src.data` is absent.
- [ ] **Step 3: Implement** the three data functions with strict schema validation and stratified 60/20/20 splitting.
- [ ] **Step 4: Add** `.gitignore`, dependency list, and dataset acquisition instructions without committing raw data.
- [ ] **Step 5: Run** `pytest tests/test_data.py -v` and verify all tests pass.
- [ ] **Step 6: Commit** with `feat: add validated fraud dataset pipeline`.

### Task 2: Leakage-Safe Preprocessing

**Files:**
- Create: `src/preprocessing.py`
- Create: `tests/test_preprocessing.py`

**Interfaces:**
- Consumes: feature frames from Task 1.
- Produces: `build_linear_preprocessor(feature_names: list[str]) -> ColumnTransformer`, `build_tree_preprocessor(feature_names: list[str]) -> ColumnTransformer`, `fit_transform_splits(preprocessor, X_train, X_validation, X_test) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]`.

- [ ] **Step 1: Write failing tests** asserting finite numeric outputs, equal feature width across splits, and unchanged fitted scaler statistics after validation/test transformation.
- [ ] **Step 2: Run** `pytest tests/test_preprocessing.py -v` and verify failure.
- [ ] **Step 3: Implement** scaling for `Time` and `Amount` in linear/neural pipelines and pass-through preprocessing for tree models.
- [ ] **Step 4: Run** `pytest tests/test_preprocessing.py -v` and verify all tests pass.
- [ ] **Step 5: Commit** with `feat: add leakage-safe preprocessing`.

### Task 3: Common Evaluation and Threshold Analysis

**Files:**
- Create: `src/evaluation.py`
- Create: `tests/test_evaluation.py`

**Interfaces:**
- Produces: `classification_metrics(y_true, y_score, threshold: float = 0.5) -> dict[str, float | int]`, `threshold_table(y_true, y_score, thresholds: np.ndarray | None = None) -> pd.DataFrame`, `best_f1_threshold(y_true, y_score) -> float`.

- [ ] **Step 1: Write failing tests** for expected metric keys, valid ranges, mismatched lengths, all-finite outputs, and a synthetic threshold case with known confusion counts.
- [ ] **Step 2: Run** `pytest tests/test_evaluation.py -v` and verify failure.
- [ ] **Step 3: Implement** ROC-AUC, PR-AUC/Average Precision, precision, recall, F1, TN/FP/FN/TP, and threshold sweeps from predicted probabilities.
- [ ] **Step 4: Run** `pytest tests/test_evaluation.py -v` and verify all tests pass.
- [ ] **Step 5: Commit** with `feat: add common fraud evaluation metrics`.

### Task 4: Business-Cost Model

**Files:**
- Create: `src/business_metrics.py`
- Create: `tests/test_business_metrics.py`

**Interfaces:**
- Produces: `BusinessCostAssumptions` dataclass, `expected_cost(y_true, y_score, amounts, threshold: float, assumptions: BusinessCostAssumptions) -> dict[str, float | int]`, `cost_sensitivity_table(...) -> pd.DataFrame`.

- [ ] **Step 1: Write failing tests** using synthetic labels, scores, amounts, and hand-computed FP/FN costs, including zero-FP and zero-FN cases.
- [ ] **Step 2: Run** `pytest tests/test_business_metrics.py -v` and verify failure.
- [ ] **Step 3: Implement** configurable missed-fraud, review-cost, and recoverable-fraud assumptions without treating them as observed facts.
- [ ] **Step 4: Run** `pytest tests/test_business_metrics.py -v` and verify all tests pass.
- [ ] **Step 5: Commit** with `feat: add business cost evaluation`.

### Task 5: Model Constructors and Logistic Baseline

**Files:**
- Create: `src/models.py`
- Create: `tests/test_models.py`
- Create: `notebooks/02_baseline_logistic_regression.ipynb`

**Interfaces:**
- Produces: `build_logistic_regression(random_state: int = 42)`, `build_random_forest(random_state: int = 42)`, `build_xgboost(random_state: int = 42)`, `FraudMLP(input_dim: int)`.

- [ ] **Step 1: Write failing tests** that each constructor returns a trainable estimator/module with deterministic seed configuration where supported.
- [ ] **Step 2: Run** `pytest tests/test_models.py -v` and verify failure.
- [ ] **Step 3: Implement** conservative model constructors with class weighting / imbalance-aware defaults and a compact PyTorch MLP.
- [ ] **Step 4: Run** `pytest tests/test_models.py -v` and verify all tests pass.
- [ ] **Step 5: Build and execute** the baseline notebook using the shared data, preprocessing, and evaluation modules; export real baseline metrics only.
- [ ] **Step 6: Commit** with `feat: add model builders and logistic baseline`.

### Task 6: Data Understanding and Multi-Model Benchmark

**Files:**
- Create: `notebooks/01_data_understanding.ipynb`
- Create: `notebooks/03_ml_benchmark.ipynb`
- Create: `outputs/figures/.gitkeep`
- Create: `outputs/metrics/.gitkeep`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: executed EDA, benchmark metrics table, ROC/PR figures, validation model-selection decision.

- [ ] **Step 1: Implement and execute** `01_data_understanding.ipynb` with schema validation, imbalance, missing/duplicate checks, amount analysis, and concise modelling implications.
- [ ] **Step 2: Implement** Random Forest and XGBoost training through the shared modules and evaluate them on validation data.
- [ ] **Step 3: Implement** compact PyTorch MLP training with class-weighted loss and early stopping on validation PR-AUC or validation loss.
- [ ] **Step 4: Execute** `03_ml_benchmark.ipynb`, exporting an actual comparison table plus ROC and precision-recall curves.
- [ ] **Step 5: Verify** all four models were evaluated on the same split and the selected candidate is justified primarily by validation PR-AUC.
- [ ] **Step 6: Commit** with `feat: benchmark fraud detection models`.

### Task 7: Explainability and Business Threshold Selection

**Files:**
- Create: `src/explainability.py`
- Create: `tests/test_explainability.py`
- Create: `notebooks/04_explainable_ai.ipynb`
- Create: `notebooks/05_business_impact.ipynb`

**Interfaces:**
- Produces: `build_shap_explainer(model, background)`, `compute_shap_values(explainer, samples)`, validation threshold decision, final held-out test metrics.

- [ ] **Step 1: Write failing explainability tests** on a tiny supported estimator asserting finite SHAP output with expected sample/feature dimensions.
- [ ] **Step 2: Run** `pytest tests/test_explainability.py -v` and verify failure.
- [ ] **Step 3: Implement** SHAP helper functions without semantic relabeling of `V1`–`V28`.
- [ ] **Step 4: Run** `pytest tests/test_explainability.py -v` and verify pass.
- [ ] **Step 5: Execute** `04_explainable_ai.ipynb` for the selected suitable model and export real global/local explanation figures.
- [ ] **Step 6: Execute** `05_business_impact.ipynb`, compare thresholds on validation data under explicit cost assumptions, freeze one threshold, and evaluate it once on test data.
- [ ] **Step 7: Commit** with `feat: add explainability and business threshold analysis`.

### Task 8: Final Reporting, README, and QA

**Files:**
- Modify: `README.md`
- Create: `report/findings.md`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: all actual outputs from Tasks 1–7.
- Produces: recruiter-facing README, concise technical findings, final integration checks.

- [ ] **Step 1: Write integration tests** covering data validation, preprocessing, model fit/predict-proba path, metric keys, and known business-cost example.
- [ ] **Step 2: Run** `pytest -v` and fix only project-related failures until the full suite passes.
- [ ] **Step 3: Update** `README.md` with the problem, dataset, pipeline, actual benchmark table, explainability summary, business conclusion, structure, reproducibility instructions, technologies, and limitations.
- [ ] **Step 4: Write** `report/findings.md` with evidence-backed final model/threshold conclusions and explicit caveats.
- [ ] **Step 5: Execute notebooks in order** `01` through `05` from a clean kernel/environment and confirm stored outputs match README/report values.
- [ ] **Step 6: Verify** raw data is absent from Git, all figures/metrics are real, and no placeholder performance numbers remain.
- [ ] **Step 7: Run** `pytest -v` one final time and require a clean pass.
- [ ] **Step 8: Commit** with `docs: finalize fraud detection portfolio project`.
