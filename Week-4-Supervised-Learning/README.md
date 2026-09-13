# Week 4 — Supervised Learning

## Breast Cancer Classification Using Machine Learning

This project implements a supervised learning workflow for binary classification using the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset. The objective is to classify breast cancer cases as **Benign (B)** or **Malignant (M)** using numerical diagnostic measurements derived from cell nuclei.

### Project Workflow

- Loaded and inspected the WDBC dataset and supporting documentation.
- Performed data quality checks for missing values, duplicate records, data types, and target distribution.
- Conducted exploratory data analysis to study feature distributions and relationships.
- Encoded the diagnosis target as **B = 0** and **M = 1**.
- Removed the identifier field from the predictive feature set.
- Applied feature scaling using **StandardScaler** for Logistic Regression.
- Split the data into **80% training and 20% testing** using stratification.
- Implemented **Logistic Regression** as the primary classification model.
- Implemented **Random Forest** as a comparison model.
- Evaluated both models using Accuracy, Precision, Recall, F1 Score, ROC-AUC, confusion matrices, ROC curves, and 5-fold stratified cross-validation.
- Analyzed Random Forest feature importance and performed error analysis.

### Key Results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 96.49% | 96.49% |
| Precision | 97.50% | 100.00% |
| Recall | 92.86% | 90.48% |
| F1 Score | 95.12% | 95.00% |
| ROC-AUC | 0.9960 | 0.9970 |
| 5-Fold CV Accuracy | 97.37% | 96.13% |

Logistic Regression was selected as the preferred model based on its higher recall, F1 Score, and mean cross-validation accuracy, while Random Forest achieved slightly higher precision and ROC-AUC.

### Files

- `wdbc.data` — Original WDBC dataset
- `wdbc.names` — Dataset documentation and feature description
- `breast_cancer_dataset.csv` — Prepared dataset used for analysis
- `breast_cancer_predictions.csv` — Test-set predictions and probabilities
- `Yuva_Week4_Breast_Cancer_Classification.ipynb` — Complete Python/Colab implementation
- `Yuva_Week4_Breast_Cancer_Classification_Report.docx` — Detailed project report

### Technologies

**Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn**

> This project is developed for educational purposes and is not intended for clinical diagnosis or medical decision-making.