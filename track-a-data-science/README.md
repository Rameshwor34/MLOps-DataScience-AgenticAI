# Track A — Data Science & MLOps

An end-to-end Data Science and MLOps pipeline for customer churn prediction, covering data validation, preprocessing, model training, evaluation, experiment tracking, model registry, deployment, serving, and production-style monitoring.

---

## 1. Project Overview

This project implements a complete machine learning lifecycle for predicting customer churn using the IBM Telco Customer Churn dataset.

The objective is not only to train a classifier, but to demonstrate how a machine learning model can be developed, evaluated, tracked, registered, promoted, served through an API, and monitored for changes in incoming data.

The complete pipeline implemented in this project is:

```text
Raw Dataset
     │
     ▼
Data Validation
     │
     ▼
Deterministic Data Cleaning
     │
     ▼
Train / Validation / Test Split
     │
     ▼
Feature Preprocessing
     │
     ▼
 ┌───┴───────────────┐
 │                   │
 ▼                   ▼
Logistic          Random Forest
Regression
 │
 └──────────┬──────────┐
            │          │
            ▼          ▼
       Gradient     Model
       Boosting    Evaluation
            │          │
            └────┬─────┘
                 ▼
          MLflow Tracking
                 │
                 ▼
          Model Comparison
                 │
                 ▼
          Best Model Selection
                 │
                 ▼
          MLflow Model Registry
                 │
          ┌──────┴──────┐
          ▼             ▼
       Staging      Production
                        │
                        ▼
                  FastAPI API
                        │
                        ▼
                  Predictions
                        │
                        ▼
              Production Monitoring
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
        Data Drift             Target Drift
             │                     │
             └──────────┬──────────┘
                        ▼
                   Evidently
                        │
                        ▼
                  HTML Reports
                        │
                        ▼
                    MLflow
````

The implementation follows the principle:

```text
IMPLEMENT
    ↓
RUN
    ↓
VERIFY OUTPUT
    ↓
TEST
    ↓
COMMIT
```

No model performance, monitoring result, or deployment state is presented without an actual local verification.

---

# 2. Project Goals

The project was designed to cover the following Data Science and MLOps lifecycle:

1. Environment setup
2. Project structure
3. Dataset acquisition
4. Dataset validation
5. Data cleaning
6. Feature engineering and preprocessing
7. Train/validation/test splitting
8. Multiple model training
9. Common model evaluation
10. MLflow experiment tracking
11. Metric logging
12. Artifact logging
13. Experiment execution
14. Model comparison
15. Best-model selection
16. Model Registry
17. Model registration
18. Staging promotion
19. Production promotion
20. Prediction serving
21. API testing
22. Monitoring reference/current datasets
23. Numerical drift monitoring
24. Categorical drift monitoring
25. Target drift monitoring
26. Custom monitoring metrics and HTML reporting
27. Automated testing and final engineering audit

---

# 3. Dataset

## 3.1 Dataset Used

The project uses the IBM Telco Customer Churn dataset.

The raw dataset contains:

```text
7043 rows
21 columns
```

The target variable is:

```text
Churn
```

with two classes:

```text
No
Yes
```

The customer identifier is:

```text
customerID
```

The identifier is removed before model training because it is an identifier rather than a predictive feature.

---

# 4. Dataset Columns

The raw dataset contains the following columns:

```text
customerID
gender
SeniorCitizen
Partner
Dependents
tenure
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaperlessBilling
PaymentMethod
MonthlyCharges
TotalCharges
Churn
```

---

# 5. Feature Types

The semantic numerical features are:

```text
SeniorCitizen
tenure
MonthlyCharges
TotalCharges
```

The categorical features are:

```text
gender
Partner
Dependents
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaperlessBilling
PaymentMethod
```

The following column is excluded from training:

```text
customerID
```

The target is:

```text
Churn
```

---

# 6. Dataset Validation

Dataset validation is implemented in:

```text
src/data/validate.py
```

The validation layer checks:

* Dataset is not empty
* Required columns are present
* Missing values
* Duplicate customer IDs
* Unexpected target values
* Numeric validity of `TotalCharges`
* Target contains at least two classes
* Dataset shape
* Target distribution

The validation intentionally does not silently repair invalid source data.

---

## 6.1 Important Raw Dataset Issue

The raw dataset contains 11 records where `TotalCharges` cannot be directly converted to a numerical value.

These records correspond to customers with:

```text
tenure = 0
TotalCharges = blank
Churn = No
```

The raw dataset is therefore correctly identified as containing invalid numerical representations.

Instead of weakening the validation logic, the raw dataset is preserved as-is.

The preprocessing pipeline handles these values deterministically by converting them to missing numerical values and subsequently imputing them using statistics learned only from the training data.

This preserves the distinction between:

```text
Raw-data validation
```

and:

```text
Model-training preprocessing
```

---

# 7. Data Preprocessing

Preprocessing is implemented in:

```text
src/data/preprocess.py
```

The preprocessing pipeline is constructed using:

```text
ColumnTransformer
+
Pipeline
```

This ensures that preprocessing is part of the trained model pipeline.

---

# 8. Numerical Preprocessing

Numerical features use:

```text
SimpleImputer(strategy="median")
        ↓
StandardScaler()
```

The numerical columns are:

```text
SeniorCitizen
tenure
MonthlyCharges
TotalCharges
```

Missing numerical values are therefore handled using the median calculated from the training data.

---

# 9. Categorical Preprocessing

Categorical features use:

```text
SimpleImputer(strategy="most_frequent")
        ↓
OneHotEncoder(handle_unknown="ignore")
```

The use of:

```text
handle_unknown="ignore"
```

ensures that unseen categorical values during inference do not cause the prediction service to fail.

---

# 10. Prevention of Data Leakage

A major design decision is that preprocessing is fitted only on the training data.

The complete model structure is:

```text
Raw Features
     ↓
Preprocessor
     ↓
Classifier
```

implemented as a single scikit-learn:

```text
Pipeline
```

Therefore:

```text
Training Data
     ↓
fit(preprocessor)
     ↓
fit(classifier)
```

while validation and test data only pass through:

```text
transform(preprocessor)
     ↓
predict(classifier)
```

This prevents validation and test information from influencing preprocessing statistics.

---

# 11. Feature Transformation

Before encoding:

```text
7043 samples
19 predictive features
```

After preprocessing:

```text
7043 samples
45 transformed features
```

The final dimensionality is produced by numerical scaling combined with categorical one-hot encoding.

---

# 12. Train / Validation / Test Split

The dataset is divided using stratified splitting.

The random seed is:

```text
42
```

The final datasets contain:

```text
Training:    4507 samples
Validation:  1127 samples
Test:        1409 samples
```

The split is stratified so that the churn class distribution remains approximately consistent across all three datasets.

Approximate class distribution:

```text
No Churn:  ~73.46%
Churn:     ~26.54%
```

---

# 13. Candidate Machine Learning Models

Three classification algorithms were selected to provide different modeling characteristics.

---

## 13.1 Logistic Regression

Configuration:

```text
LogisticRegression(
    solver="liblinear",
    max_iter=1000,
    random_state=42
)
```

Logistic Regression provides a strong linear baseline and is computationally efficient.

---

## 13.2 Random Forest

Configuration:

```text
RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
```

Random Forest provides a nonlinear tree-based ensemble capable of modeling feature interactions.

---

## 13.3 Gradient Boosting

Configuration:

```text
GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
```

Gradient Boosting provides another nonlinear ensemble approach based on sequentially improved weak learners.

---

# 14. Training Architecture

Training is implemented in:

```text
src/training/train.py
```

Each candidate model is combined with the same preprocessing pipeline:

```text
Pipeline(
    preprocessor,
    classifier
)
```

This ensures that all models are trained under the same preprocessing conditions.

The classifier is therefore never trained on manually transformed data that could differ from production preprocessing.

---

# 15. Evaluation Metrics

Evaluation is implemented in:

```text
src/evaluation/metrics.py
src/evaluation/evaluate.py
```

The following metrics are calculated:

```text
Accuracy
Precision
Recall
F1 Score
ROC-AUC
Confusion Matrix
```

---

# 16. Model Selection Policy

The model selection policy is defined before comparing the models.

Primary metric:

```text
F1 Score
```

Secondary metric:

```text
ROC-AUC
```

Therefore:

```text
1. Highest F1 Score
2. If tied, highest ROC-AUC
```

F1 is used as the primary metric because churn prediction requires balancing:

```text
Precision
+
Recall
```

rather than optimizing accuracy alone.

---

# 17. Validation Results

The verified validation results are:

| Model               | Accuracy | Precision | Recall |     F1 | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -----: | ------: |
| Logistic Regression |   0.8163 |    0.6966 | 0.5452 | 0.6116 |  0.8481 |
| Random Forest       |   0.7959 |    0.6605 | 0.4749 | 0.5525 |  0.8262 |
| Gradient Boosting   |   0.8092 |    0.6981 | 0.4950 | 0.5793 |  0.8522 |

---

# 18. Confusion Matrices

Confusion matrices are generated for all three candidate models.

The artifacts are:

```text
artifacts/plots/logistic_regression_confusion_matrix.png
artifacts/plots/random_forest_confusion_matrix.png
artifacts/plots/gradient_boosting_confusion_matrix.png
```

The verified Logistic Regression confusion matrix is:

```text
[[757,  71],
 [136, 163]]
```

The verified Random Forest confusion matrix is:

```text
[[755,  73],
 [157, 142]]
```

The verified Gradient Boosting confusion matrix is:

```text
[[764,  64],
 [151, 148]]
```

---

# 19. Best Model

According to the predefined selection policy:

```text
Primary criterion: F1
Secondary criterion: ROC-AUC
```

the selected model is:

```text
Logistic Regression
```

Its validation performance is:

```text
Accuracy:   0.8163
Precision:  0.6966
Recall:     0.5452
F1:         0.6116
ROC-AUC:    0.8481
```

Gradient Boosting achieved the highest ROC-AUC:

```text
0.8522
```

but its F1 score:

```text
0.5793
```

was lower than Logistic Regression's:

```text
0.6116
```

Therefore Logistic Regression is selected under the explicitly defined F1-first policy.

---

# 20. MLflow Experiment Tracking

MLflow is used for experiment tracking.

The main experiment is:

```text
track-a-telco-churn
```

MLflow records:

* Model parameters
* Accuracy
* Precision
* Recall
* F1
* ROC-AUC
* Confusion matrix
* Trained model artifacts

The experiment tracking backend uses a local SQLite database.

The generated MLflow runtime state is intentionally excluded from version control.

---

# 21. MLflow Runs

The successfully completed model runs include:

### Logistic Regression

```text
Run ID:
7e4a8d4d84d2468d90fa914b3b59f6eb
```

### Random Forest

```text
Run ID:
63986d1d777d4d8a97ad128027b49c20
```

### Gradient Boosting

```text
Run ID:
0a0773451835491baef76fb41782ac7a
```

These runs contain the corresponding parameters, metrics, confusion matrix artifacts, and serialized model artifacts.

---

# 22. Model Comparison

Model comparison is implemented in:

```text
src/evaluation/compare.py
```

The comparison process retrieves finished MLflow runs and orders them using:

```text
F1 Score descending
ROC-AUC descending
```

This ensures that the model selection logic is consistent with the predefined evaluation policy.

---

# 23. MLflow Model Registry

The selected model is registered under:

```text
TelcoChurnClassifier
```

The currently registered model is:

```text
TelcoChurnClassifier
Version 1
```

The registered model corresponds to the selected Logistic Regression experiment.

The model artifact is referenced through its MLflow run.

---

# 24. Model Lifecycle

The model lifecycle is represented using MLflow aliases:

```text
staging
production
```

Current verified state:

```text
TelcoChurnClassifier
        │
        ├── staging
        │      ↓
        │    Version 1
        │
        └── production
               ↓
             Version 1
```

The same validated model version was promoted through both environments during the local lifecycle demonstration.

---

# 25. Staging Verification

The staging alias points to:

```text
Version 1
```

The associated run is:

```text
7e4a8d4d84d2468d90fa914b3b59f6eb
```

The staging state was verified through MLflow.

---

# 26. Production Verification

The production alias also points to:

```text
Version 1
```

The production model therefore corresponds to the same selected and validated Logistic Regression model.

The following lifecycle was verified:

```text
Best Model
    ↓
Registry
    ↓
Staging
    ↓
Production
```

---

# 27. FastAPI Model Serving

The prediction API is implemented in:

```text
src/serving/app.py
```

Request schemas are defined in:

```text
src/serving/schemas.py
```

The API loads:

```text
models:/TelcoChurnClassifier@production
```

The production model is cached so that the model does not need to be reloaded for every request.

---

# 28. API Endpoints

## Health Endpoint

```http
GET /health
```

The endpoint reports that the API is healthy and identifies the production model alias.

---

## Prediction Endpoint

```http
POST /predict
```

The endpoint accepts customer information and returns:

```json
{
  "churn_prediction": 1,
  "churn_label": "Yes",
  "churn_probability": 0.6005373259547537
}
```

The exact probability above is from a verified local prediction test.

---

# 29. API Verification

The serving layer was verified using:

```text
GET /health
    ↓
HTTP 200
```

and:

```text
POST /predict
    ↓
HTTP 200
```

The prediction response successfully contained:

```text
churn_prediction
churn_label
churn_probability
```

---

# 30. Monitoring Architecture

Monitoring is implemented under:

```text
src/monitoring/
```

The monitoring system evaluates both:

```text
Feature/Data Drift
```

and:

```text
Target Drift
```

Evidently is used for feature-level distribution monitoring and HTML report generation.

Custom Python logic is used for target drift.

---

# 31. Monitoring Datasets

The monitoring system maintains three datasets:

```text
data/monitoring/reference.csv
data/monitoring/current.csv
data/monitoring/drifted.csv
```

Their sizes are:

```text
Reference: 4507 rows
Current:   1409 rows
Drifted:   1409 rows
```

---

# 32. Reference Dataset

The reference dataset is derived from the training data.

It represents the baseline feature and target distribution against which subsequent datasets are compared.

The reference churn rate is approximately:

```text
0.2654
```

---

# 33. Current Dataset

The current monitoring dataset is based on the held-out test data.

Its churn rate is approximately:

```text
0.2654
```

This provides a controlled baseline where the current distribution is close to the reference distribution.

---

# 34. Synthetic Drift Dataset

The `drifted.csv` dataset is deliberately modified to test whether the monitoring system can detect known distribution changes.

This is a controlled monitoring fixture.

It is not claimed to represent real production drift.

---

# 35. Numerical Drift

The synthetic drift experiment modifies:

```text
MonthlyCharges
```

by increasing values by approximately:

```text
20%
```

The reference mean MonthlyCharges is approximately:

```text
64.70
```

The current mean is approximately:

```text
64.09
```

The synthetic drifted mean is approximately:

```text
76.91
```

This creates a deliberate numerical distribution shift.

---

# 36. Categorical Drift

The synthetic drift experiment also modifies:

```text
Contract
```

by moving non-month-to-month contract values toward:

```text
Month-to-month
```

This produces a controlled categorical distribution shift.

---

# 37. Evidently Data Drift

Evidently is used to evaluate numerical and categorical feature drift.

The main report is generated as:

```text
artifacts/reports/telco_data_drift_report.html
```

The report is generated using:

```text
DataDriftPreset
```

with monitoring tests enabled.

---

# 38. Normal Data Drift Result

For the normal reference/current comparison:

```text
Drifted columns: 0
Drift share:     0.0000
```

Therefore the normal current dataset does not trigger the configured feature-drift condition.

---

# 39. Synthetic Drift Result

For the controlled drifted dataset:

```text
Drifted columns: 2
Drift share:     0.1053
```

This demonstrates that the monitoring pipeline detects the intentionally introduced distribution changes.

---

# 40. Target Drift Monitoring

Target drift is monitored separately because the target variable is semantically different from input features.

The monitored target is:

```text
Churn
```

The monitoring logic compares:

```text
Reference churn rate
```

against:

```text
Current churn rate
```

---

# 41. Target Drift Threshold

The configured target-drift threshold is:

```text
0.05
```

which corresponds to:

```text
5 percentage points
```

Target drift is considered detected when:

```text
absolute(current_churn_rate - reference_churn_rate) >= 0.05
```

---

# 42. Normal Target Drift Result

For the normal current dataset:

```text
Reference churn rate: 0.2654
Current churn rate:   0.2654
Absolute delta:       0.0001
Threshold:            0.0500
Detected:             False
```

Therefore normal data does not trigger target drift.

---

# 43. Synthetic Target Drift Result

The controlled drifted dataset has:

```text
Reference churn rate: 0.2654
Drifted churn rate:   0.3648
Absolute delta:       0.0994
Threshold:            0.0500
Detected:             True
```

This demonstrates that the custom target monitoring logic correctly detects a meaningful target-distribution change.

---

# 44. Monitoring Reports

The monitoring system produces HTML reports including:

```text
artifacts/reports/telco_data_drift_report.html

artifacts/reports/telco_monitoring_current.html

artifacts/reports/telco_monitoring_drifted.html
```

The reports are generated locally and are intentionally excluded from Git because they are generated artifacts.

---

# 45. MLflow Monitoring

Monitoring runs are tracked using MLflow.

The monitoring experiment is:

```text
track-a-monitoring
```

The monitoring run records values such as:

```text
Drifted columns count
Drift share
Reference churn rate
Current churn rate
Target drift delta
Absolute target drift delta
Target drift detected
```

Both normal and synthetic monitoring scenarios have been executed.

---

# 46. Monitoring Run — Current Data

The normal monitoring run produced:

```text
Drifted columns count: 0
Drift share:           0.0000

Reference churn:       0.2654
Current churn:         0.2654

Absolute target delta: 0.0001
Target drift detected: False
```

---

# 47. Monitoring Run — Synthetic Drift

The synthetic monitoring run produced:

```text
Drifted columns count: 2
Drift share:           0.1053

Reference churn:       0.2654
Drifted churn:         0.3648

Absolute target delta: 0.0994
Target drift detected: True
```

---

# 48. Testing

Automated tests are implemented using:

```text
pytest
```

Current test modules:

```text
tests/test_evaluation.py
tests/test_monitoring.py
```

The test suite currently covers:

* Evaluation metric behavior
* Monitoring dataset availability
* Normal target-drift behavior
* Synthetic target-drift behavior
* Evidently drift metric extraction

---

# 49. Test Result

The complete automated test suite was executed using:

```powershell
uv run pytest -v
```

Verified result:

```text
6 passed
```

The test execution completed successfully.

Some dependency warnings are emitted by third-party dependencies such as Evidently/Litestar, but these do not represent test failures.

---

# 50. Project Structure

The main project structure is:

```text
track-a-data-science/
│
├── data/
│   ├── raw/
│   │   └── telco_churn.csv
│   │
│   ├── processed/
│   │   └── generated train/validation/test datasets
│   │
│   └── monitoring/
│       ├── reference.csv
│       ├── current.csv
│       └── drifted.csv
│
├── artifacts/
│   ├── models/
│   ├── plots/
│   │   ├── logistic_regression_confusion_matrix.png
│   │   ├── random_forest_confusion_matrix.png
│   │   └── gradient_boosting_confusion_matrix.png
│   │
│   └── reports/
│       ├── telco_data_drift_report.html
│       ├── telco_monitoring_current.html
│       └── telco_monitoring_drifted.html
│
├── src/
│   ├── data/
│   │   ├── download.py
│   │   ├── validate.py
│   │   ├── preprocess.py
│   │   └── split.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── models.py
│   │   ├── experiment.py
│   │   ├── registry.py
│   │   └── promote.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── evaluate.py
│   │   ├── plots.py
│   │   └── compare.py
│   │
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   └── schemas.py
│   │
│   └── monitoring/
│       ├── reference.py
│       ├── drift.py
│       └── report.py
│
├── tests/
│   ├── test_evaluation.py
│   └── test_monitoring.py
│
├── .env
├── .gitignore
├── README.md
├── pyproject.toml
└── uv.lock
```

---

# 51. Environment

The project uses:

```text
Python 3.12.10
```

The virtual environment is managed using:

```text
uv
```

The environment was recreated using Python 3.12 because the original Python 3.14 environment caused compatibility issues with native dependencies during MLflow/scikit-learn setup.

---

# 52. Core Dependencies

The verified environment contains:

```text
NumPy:        2.5.3
Pandas:       2.2.3
scikit-learn: 1.5.2
MLflow:       3.17.0
Evidently:    0.7.23
FastAPI:      0.142.4
ujson:        5.11.0
```

The environment is defined through:

```text
pyproject.toml
uv.lock
```

The lock file provides reproducible dependency resolution.

---

# 53. Environment Setup

From the project root:

```powershell
uv venv --python 3.12
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Synchronize dependencies:

```powershell
uv sync
```

Verify Python:

```powershell
uv run python --version
```

---

# 54. Dataset Commands

Download the dataset:

```powershell
uv run python src/data/download.py
```

Validate the raw dataset:

```powershell
uv run python src/data/validate.py
```

---

# 55. Training Commands

Train the candidate models:

```powershell
uv run python src/training/train.py
```

Run evaluation:

```powershell
uv run python src/evaluation/evaluate.py
```

Run MLflow experiments:

```powershell
uv run python src/training/experiment.py
```

Compare models:

```powershell
uv run python src/evaluation/compare.py
```

---

# 56. Registry Commands

Register the selected model:

```powershell
uv run python src/training/registry.py
```

Promote the model:

```powershell
uv run python src/training/promote.py
```

---

# 57. Serving Commands

Start the FastAPI application:

```powershell
uv run uvicorn src.serving.app:app --reload
```

The API exposes:

```text
GET /health
POST /predict
```

FastAPI also provides automatically generated API documentation.

---

# 58. Monitoring Commands

Generate monitoring datasets:

```powershell
uv run python src/monitoring/reference.py
```

Run feature and target drift analysis:

```powershell
uv run python src/monitoring/drift.py
```

Generate and log monitoring reports:

```powershell
uv run python src/monitoring/report.py
```

---

# 59. Testing Commands

Run the complete automated test suite:

```powershell
uv run pytest -v
```

Run only evaluation tests:

```powershell
uv run pytest tests/test_evaluation.py -v
```

Run only monitoring tests:

```powershell
uv run pytest tests/test_monitoring.py -v
```

---

# 60. Reproducibility

The project uses deterministic configuration wherever practical.

The primary random seed is:

```text
42
```

This seed is used for:

* Train/test splitting
* Validation splitting
* Logistic Regression
* Random Forest
* Gradient Boosting

The preprocessing pipeline is fitted only on training data.

Model artifacts are tracked by MLflow rather than manually copying model files between stages.

---

# 61. Git Hygiene

Generated runtime files are excluded from version control.

The `.gitignore` excludes:

```text
.venv/
.env
__pycache__/
.pytest_cache/

mlflow.db
mlruns/
mlartifacts/

artifacts/models/
artifacts/plots/
artifacts/reports/

data/processed/
data/monitoring/

.vscode/
.idea/
```

This separates:

```text
Source Code
Configuration
Documentation
```

from:

```text
Generated Runtime State
```

---

# 62. Important Engineering Decisions

## Training-only preprocessing

All learned preprocessing statistics come from the training dataset.

This prevents leakage from validation and test data.

---

## Unified preprocessing and model pipeline

The preprocessing transformer and classifier are packaged together.

This prevents training-time and inference-time preprocessing from diverging.

---

## Stratified data splitting

Stratification preserves class proportions across the datasets.

This is particularly important because churn is an imbalanced binary classification problem.

---

## F1-first model selection

F1 is used as the primary selection metric.

ROC-AUC is used as a secondary criterion.

The selection rule is explicitly encoded rather than manually choosing a model after looking at results.

---

## Controlled synthetic drift

Synthetic drift is deliberately introduced into the monitoring dataset.

This allows the monitoring implementation to be tested against a known distribution change.

The synthetic drift dataset is not presented as real production data.

---

## Separate feature and target monitoring

Feature drift and target drift are treated as separate monitoring concerns.

Evidently is used for feature distribution monitoring, while a custom target-rate calculation is used for target drift.

---

## MLflow aliases

Modern MLflow aliases are used for:

```text
staging
production
```

This avoids depending on legacy model-stage behavior.

---

# 63. Current End-to-End Status

The implemented lifecycle has been locally verified through:

```text
Environment Setup
        ✓
Dataset Loading
        ✓
Dataset Validation
        ✓
Preprocessing
        ✓
Train/Validation/Test Split
        ✓
Three Candidate Models
        ✓
Model Evaluation
        ✓
Confusion Matrix Generation
        ✓
MLflow Experiment Tracking
        ✓
Metric Logging
        ✓
Artifact Logging
        ✓
Model Comparison
        ✓
Best Model Selection
        ✓
MLflow Model Registry
        ✓
Staging Alias
        ✓
Production Alias
        ✓
FastAPI Serving
        ✓
Prediction Endpoint
        ✓
Reference Monitoring Dataset
        ✓
Current Monitoring Dataset
        ✓
Synthetic Drift Dataset
        ✓
Numerical Drift Monitoring
        ✓
Categorical Drift Monitoring
        ✓
Target Drift Monitoring
        ✓
Evidently HTML Reports
        ✓
MLflow Monitoring
        ✓
Automated Tests
        ✓
```

---

# 64. Final Verified Architecture

The final architecture can be summarized as:

```text
                         ┌───────────────────┐
                         │   Telco Dataset   │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ Data Validation   │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ Data Preprocessing│
                         │                   │
                         │ Numerical:        │
                         │ Impute + Scale    │
                         │                   │
                         │ Categorical:      │
                         │ Impute + OneHot   │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ Stratified Split  │
                         └─────────┬─────────┘
                                   │
                 ┌─────────────────┼─────────────────┐
                 │                 │                 │
                 ▼                 ▼                 ▼
          ┌────────────┐    ┌────────────┐    ┌──────────────┐
          │ Logistic   │    │ Random     │    │ Gradient     │
          │ Regression │    │ Forest     │    │ Boosting     │
          └─────┬──────┘    └─────┬──────┘    └──────┬───────┘
                │                 │                   │
                └─────────────────┼───────────────────┘
                                  │
                                  ▼
                         ┌───────────────────┐
                         │ Model Evaluation  │
                         │                   │
                         │ Accuracy          │
                         │ Precision         │
                         │ Recall            │
                         │ F1                │
                         │ ROC-AUC           │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │      MLflow       │
                         │ Experiment        │
                         │ Tracking          │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ Model Comparison  │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ Best Model        │
                         │ Logistic          │
                         │ Regression        │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ MLflow Registry   │
                         │                   │
                         │ TelcoChurn        │
                         │ Classifier        │
                         └─────────┬─────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
             ┌─────────────┐              ┌─────────────┐
             │   Staging   │              │ Production  │
             └─────────────┘              └──────┬──────┘
                                                 │
                                                 ▼
                                         ┌───────────────┐
                                         │    FastAPI    │
                                         │ Prediction API│
                                         └───────┬───────┘
                                                 │
                                                 ▼
                                         ┌───────────────┐
                                         │   Monitoring  │
                                         └───────┬───────┘
                                                 │
                               ┌─────────────────┴─────────────────┐
                               │                                   │
                               ▼                                   ▼
                       ┌───────────────┐                   ┌──────────────┐
                       │ Feature Drift │                   │ Target Drift │
                       │   Evidently   │                   │ Custom Logic │
                       └───────┬───────┘                   └──────┬───────┘
                               │                                  │
                               └────────────────┬─────────────────┘
                                                │
                                                ▼
                                        ┌───────────────┐
                                        │ HTML Reports  │
                                        │   + MLflow    │
                                        └───────────────┘
```

---

# 65. Conclusion

This project demonstrates a complete local Data Science and MLOps lifecycle rather than stopping at model training.

The system begins with raw customer data and progresses through:

```text
Validation
    ↓
Preprocessing
    ↓
Model Training
    ↓
Evaluation
    ↓
Experiment Tracking
    ↓
Model Selection
    ↓
Model Registry
    ↓
Staging
    ↓
Production
    ↓
API Serving
    ↓
Data Monitoring
    ↓
Target Monitoring
```

The selected model is currently:

```text
Logistic Regression
```

based on the predefined:

```text
F1-first
ROC-AUC-second
```

selection policy.

The model is registered as:

```text
TelcoChurnClassifier
Version 1
```

and the staging and production aliases have been verified.

The serving layer successfully exposes the production model through FastAPI, while the monitoring layer detects both controlled feature drift and controlled target drift.

The project therefore provides an executable demonstration of the transition from:

```text
Machine Learning Model
```

to:

```text
Managed, Served, and Monitored ML System
```

---

# 66. Technologies Used

```text
Python
uv
Pandas
NumPy
scikit-learn
MLflow
Evidently
FastAPI
Pydantic
pytest
SQLite
Git
GitHub
```

---

# 67. Author

**Rameshwor Poudel**

Computer Engineering
Institute of Engineering, Tribhuvan University
Nepal

GitHub:

```text
Rameshwor34
```

---

# 68. Project Philosophy

The central engineering principle of this project is:

```text
A machine learning project is not complete
when the model achieves a good score.

It becomes a deployable ML system when:

        Data
          ↓
      Validation
          ↓
      Preprocessing
          ↓
       Training
          ↓
      Evaluation
          ↓
       Tracking
          ↓
       Registry
          ↓
      Deployment
          ↓
       Serving
          ↓
      Monitoring
          ↓
      Feedback
```

This project is structured around that complete lifecycle.

````

