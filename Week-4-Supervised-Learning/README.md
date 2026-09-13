# Week 4 — Supervised Learning

## Breast Cancer Classification Using Machine Learning

This project applies **supervised machine learning** to the Wisconsin Diagnostic Breast Cancer (WDBC) dataset to develop a binary classification model for distinguishing **Benign (B)** and **Malignant (M)** breast cancer cases. The project follows a complete supervised learning workflow from dataset inspection and exploratory analysis to preprocessing, model training, evaluation, feature analysis, and prediction generation.

## Objectives

- Define a binary classification problem using a publicly available dataset.
- Inspect and validate the dataset before modeling.
- Perform exploratory data analysis to understand feature distributions and target classes.
- Prepare the data through target encoding, feature selection, and scaling.
- Split the dataset into training and testing subsets using stratification.
- Train a primary supervised learning classification model.
- Compare the primary model with an additional ensemble model.
- Evaluate model performance using multiple classification metrics.
- Analyze prediction errors and important features.
- Generate a reproducible dataset containing model predictions and probabilities.

## Dataset

The **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset contains numerical measurements computed from digitized images of breast mass cell nuclei. Each record is labelled as either benign or malignant.

The dataset contains **569 observations and 30 numerical diagnostic features**, along with an identifier and diagnosis label. The 30 features describe measurements such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

| Component | Description | Role |
|---|---|---|
| `ID` | Patient/sample identifier | Identifier; excluded from modeling |
| `Diagnosis` | B = Benign, M = Malignant | Target variable |
| 30 diagnostic features | Numerical cell-nuclei measurements | Predictive features |

### Data Quality

- Dataset size: **569 rows × 32 original WDBC fields**
- Missing values: **0**
- Duplicate rows: **0**
- Target classes: **357 Benign, 212 Malignant**
- Classification type: **Binary classification**

## Methodology

### 1. Data Inspection and Quality Assessment

The WDBC dataset and its supporting documentation were inspected to understand the structure, feature definitions, target labels, data types, missing values, duplicate records, and class distribution before model development.

### 2. Exploratory Data Analysis

EDA was performed to examine the target-class distribution, numerical feature behaviour, relationships among diagnostic measurements, and differences between benign and malignant observations. Visualizations were used to support interpretation before applying machine-learning models.

### 3. Target Encoding and Feature Selection

The diagnosis label was converted into a numerical target:

- **B (Benign) → 0**
- **M (Malignant) → 1**

The identifier column was removed from the predictive feature set because it is an identifier rather than a meaningful diagnostic measurement. The remaining 30 numerical features were used for classification.

### 4. Train-Test Split

The dataset was divided into:

- **80% training data**
- **20% testing data**

Stratified splitting with `random_state=42` was used so that the class proportions were preserved between the training and testing sets while maintaining reproducibility.

### 5. Feature Scaling

`StandardScaler` was applied to the numerical features for Logistic Regression. Scaling places features on comparable numerical ranges and prevents variables with larger raw magnitudes from disproportionately influencing the linear model.

### 6. Model Training

Two supervised learning algorithms were implemented:

**Logistic Regression — Primary Model**

Logistic Regression was selected as the primary model because the task is binary classification and the algorithm provides a relatively interpretable baseline for estimating class probabilities.

**Random Forest — Comparison Model**

Random Forest was used as a comparison model because its ensemble of decision trees can capture non-linear relationships and interactions among diagnostic features without requiring the same linear assumptions as Logistic Regression.

### 7. Model Evaluation

Both models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- ROC Curve
- 5-Fold Stratified Cross-Validation

Using multiple metrics provides a more complete evaluation than accuracy alone, particularly when distinguishing between false-positive and false-negative predictions.

## Final Results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | **96.49%** | **96.49%** |
| Precision | **97.50%** | **100.00%** |
| Recall | **92.86%** | **90.48%** |
| F1 Score | **95.12%** | **95.00%** |
| ROC-AUC | **0.9960** | **0.9970** |
| 5-Fold CV Accuracy | **97.37%** | **96.13%** |

### Confusion Matrix — Logistic Regression

| Actual \ Predicted | Benign | Malignant |
|---|---:|---:|
| Benign | 71 | 1 |
| Malignant | 3 | 39 |

The Logistic Regression model correctly classified **110 of 114 test observations**, with 4 incorrect predictions. Its recall of **92.86%** indicates that it identified most malignant cases in the held-out test set.

## Model Comparison and Interpretation

Logistic Regression and Random Forest achieved the same test accuracy of **96.49%**, but their error profiles were slightly different.

- Logistic Regression achieved higher **recall (92.86%)** than Random Forest (90.48%).
- Logistic Regression also achieved a slightly higher **F1 Score (95.12%)** and higher mean **5-Fold CV Accuracy (97.37%)**.
- Random Forest achieved **100% precision** and a slightly higher **ROC-AUC of 0.9970**.
- Because identifying malignant cases is particularly important for this classification task, recall and false-negative behaviour were considered important when selecting the preferred model.

Based on the combined evaluation, **Logistic Regression was selected as the preferred model** for this project, while Random Forest provided a strong comparative benchmark.

## Feature Analysis

Random Forest feature importance was examined to identify which diagnostic measurements contributed most strongly to the tree-based model's predictions. This analysis provides an additional perspective on the predictive structure of the dataset and supports interpretation of the classification results.

Feature importance should be interpreted as a model-specific measure of predictive contribution rather than as evidence of medical causation.

## Error Analysis

The Logistic Regression test-set confusion matrix contains:

- **71 true negatives**
- **39 true positives**
- **1 false positive**
- **3 false negatives**

The presence of false negatives demonstrates why accuracy alone is insufficient for evaluating this type of classification problem. Recall, precision, F1 Score, and confusion-matrix analysis provide additional information about the types of errors made by the model.

## Key Insights

1. The WDBC dataset provides a well-defined binary classification problem using 30 numerical diagnostic measurements.
2. The dataset contained **569 observations with no missing values or duplicate records**, reducing the need for extensive data cleaning before modeling.
3. Removing the identifier prevented a non-predictive ID value from influencing the classification models.
4. Standardization was important for the Logistic Regression workflow because the diagnostic variables have different numerical scales.
5. Both models achieved **96.49% test accuracy**, demonstrating strong predictive performance on the held-out test set.
6. Logistic Regression produced higher recall and F1 Score, while Random Forest produced higher precision and slightly higher ROC-AUC.
7. The **5-Fold CV Accuracy of 97.37% for Logistic Regression** provides additional evidence that its performance was consistent across multiple stratified validation folds.
8. The results demonstrate the importance of evaluating a classification model using several complementary metrics rather than relying on a single score.

## Technologies and Libraries

- **Python**
- **Pandas** — data loading, cleaning, transformation, and analysis
- **NumPy** — numerical operations
- **Matplotlib** — visualization
- **Seaborn** — statistical visualization
- **Scikit-learn** — preprocessing, model training, metrics, pipelines, and cross-validation
- **Google Colab / Jupyter Notebook** — development and execution environment

## Repository Files

```text
Week-4-Supervised-Learning/
├── README.md
├── wdbc.data
├── wdbc.names
├── breast_cancer_dataset.csv
├── breast_cancer_predictions.csv
├── Yuva_Week4_Breast_Cancer_Classification.ipynb
└── Yuva_Week4_Breast_Cancer_Classification_Report.docx
```

### Outputs

- `wdbc.data` — Original WDBC dataset used for the classification task.
- `wdbc.names` — Supporting dataset documentation and feature descriptions.
- `breast_cancer_dataset.csv` — Prepared dataset containing the diagnosis labels and numerical features used in the project.
- `breast_cancer_predictions.csv` — Test-set predictions, malignant probabilities, and prediction correctness.
- `Yuva_Week4_Breast_Cancer_Classification.ipynb` — Complete executed Python/Colab workflow including preprocessing, EDA, model training, evaluation, and analysis.
- `Yuva_Week4_Breast_Cancer_Classification_Report.docx` — Detailed report containing methodology, visualizations, model evaluation, interpretations, error analysis, and conclusion.

## Reproducibility

The notebook uses fixed random states for the train-test split and machine-learning models so that the workflow can be reproduced consistently. The project includes both the source dataset and generated prediction output to make the analysis easier to inspect and verify.

## Limitations

- The analysis is based on a single publicly available dataset and a single held-out test split.
- Strong test performance does not guarantee the same performance on external or unseen clinical datasets.
- The models are developed for educational and analytical purposes rather than clinical deployment.
- Feature importance and model predictions should not be interpreted as medical causation or clinical recommendations.
- Further work could include hyperparameter tuning, alternative classification algorithms, feature selection, calibration, external validation, and evaluation on independent datasets.

## Conclusion

The Week 4 project demonstrates a complete **supervised learning classification workflow** using the Wisconsin Diagnostic Breast Cancer dataset. The process covered data inspection, exploratory analysis, target encoding, feature selection, train-test splitting, feature scaling, Logistic Regression and Random Forest modeling, multi-metric evaluation, cross-validation, feature analysis, and error analysis.

Both models achieved **96.49% accuracy** on the test set. Logistic Regression was preferred because it achieved higher recall, F1 Score, and cross-validation accuracy, while Random Forest provided competitive performance with higher precision and ROC-AUC. Overall, the project demonstrates how a structured supervised-learning pipeline can be used to build, evaluate, compare, and interpret binary classification models using Python.

> **Disclaimer:** This project is intended for educational and data-science learning purposes only. It is not a clinical diagnostic system and should not be used for medical decision-making.

## Author

**Mohammed Aayan**

B.Tech — Computer Science and Information Technology
