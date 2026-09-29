# Week 6 — Integrative Capstone Project

## Customer Churn Prediction and Customer Segmentation Using Data Science

This capstone integrates the major Data Science skills practiced throughout the internship into one end-to-end project. The project analyzes customer demographic, service, contract, and billing information to identify churn patterns, segment customers using unsupervised learning, predict churn using supervised machine learning, evaluate model performance, analyze errors, and derive evidence-based recommendations.

## Project Objectives

1. Analyze customer characteristics and identify patterns associated with churn.
2. Clean and preprocess a publicly available customer dataset.
3. Engineer meaningful customer lifecycle and service-adoption features.
4. Segment customers using K-Means clustering.
5. Predict customer churn using supervised classification models.
6. Validate and tune the models using stratified cross-validation.
7. Perform detailed error analysis and model interpretation.
8. Produce evidence-based insights, recommendations, limitations, and future improvement directions.

## Dataset

**Dataset:** Telco Customer Churn  
**Original size:** 7,043 customers × 21 variables  
**Target:** `Churn` (`Yes` / `No`)  
**Public source:** IBM sample Telco Customer Churn dataset

The data contains customer demographics, tenure, subscribed services, contract information, payment method, monthly charges, total charges, and churn status.

## Data Cleaning

The raw dataset was inspected before preprocessing.

Key validation findings:
- 7,043 original records and 21 variables.
- No duplicate rows were found.
- `TotalCharges` was stored as text rather than numeric.
- 11 blank `TotalCharges` values were identified.
- All 11 blank records had tenure = 0, so the cleaned numeric representation used 0.0 for these records.
- Final cleaned dataset: 7,043 × 21.
- Final missing values: 0.
- Final duplicate rows: 0.

## Exploratory Data Analysis

EDA included:
- Churn distribution and class balance analysis.
- Numerical distributions for tenure, monthly charges, and total charges.
- Churn rates across contract, internet service, payment method, support/security services, and other categorical variables.
- Tenure-group analysis.
- Service-adoption analysis.
- Correlation analysis.
- Chi-square association tests with Cramer's V.
- Mann-Whitney U tests for numerical group differences.

Important observed patterns:
- Overall churn rate: 26.54%.
- Month-to-month customers: 42.71% observed churn.
- One-year customers: 11.27%.
- Two-year customers: 2.83%.
- Fiber optic customers: 41.89%.
- DSL customers: 18.96%.
- Customers without internet service: 7.40%.
- Electronic-check customers: 45.29%.
- 0–6 month customers: 52.94%.
- Customers with more than 6 months tenure: 19.51%.

These are observed associations within the dataset and are not treated as causal effects.

## Feature Engineering

Additional analytical features include:
- `Num_Services`
- `SecuritySupport_Count`
- `Streaming_Count`
- `Tenure_Group`
- `Is_New_Customer`
- `Above_Median_MonthlyCharge`
- `Churn_Binary`

The customer identifier was excluded from modeling. Churn was excluded from unsupervised clustering to prevent target leakage.

## Customer Segmentation — K-Means

Seven numerical features were standardized before clustering:
- SeniorCitizen
- tenure
- MonthlyCharges
- TotalCharges
- Num_Services
- SecuritySupport_Count
- Streaming_Count

K values from 2 to 10 were evaluated using:
- Inertia / Elbow method
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score

The selected K=2 solution had:
- Silhouette Score: 0.3901
- Davies-Bouldin Index: 1.0622
- Calinski-Harabasz Score: 5586.3037

The clustering metrics did not completely agree: Silhouette and Calinski-Harabasz preferred K=2, while Davies-Bouldin preferred K=8. This trade-off is documented in the report.

### Segment Profiles

**Cluster 0**
- 2,873 customers (40.79%)
- Mean tenure: 49.45 months
- Mean MonthlyCharges: 89.31
- Mean TotalCharges: 4,410.58
- Mean services: 5.43

**Cluster 1**
- 4,170 customers (59.21%)
- Mean tenure: 20.60 months
- Mean MonthlyCharges: 47.85
- Mean TotalCharges: 811.65
- Mean services: 1.94

Observed churn rates after clustering:
- Cluster 0: 21.34%
- Cluster 1: 30.12%

Churn was not used to create the clusters.

## Churn Prediction

Three levels of classification were evaluated:
- Dummy majority-class baseline
- Logistic Regression
- Random Forest

An 80/20 stratified train-test split was used:
- Training: 5,634
- Testing: 1,409
- Churn rate preserved at 26.54% in both partitions

### Initial Models

**Logistic Regression**
- Accuracy: 80.27%
- Precision: 66.11%
- Recall: 52.67%
- F1: 58.63%
- ROC-AUC: 0.8458
- PR-AUC: 0.6525

**Random Forest**
- Accuracy: 77.86%
- Precision: 60.62%
- Recall: 47.33%
- F1: 53.15%
- ROC-AUC: 0.8188
- PR-AUC: 0.6078

The dummy baseline achieved 73.46% accuracy while detecting none of the churners, demonstrating why accuracy alone is insufficient for this problem.

## Model Validation and Improvement

Five-fold stratified cross-validation was performed.

Mean cross-validation results:
- Logistic Regression F1: 0.5979 ± 0.0335
- Logistic Regression ROC-AUC: 0.8481 ± 0.0113
- Random Forest F1: 0.5470 ± 0.0282
- Random Forest ROC-AUC: 0.8227 ± 0.0125

Hyperparameter tuning was performed with GridSearchCV.

Best Logistic Regression configuration:
- C = 0.01
- class_weight = balanced

Best Random Forest configuration:
- n_estimators = 250
- max_depth = None
- min_samples_leaf = 3
- class_weight = balanced

A probability-threshold analysis was also performed using out-of-fold training predictions.

## Error Analysis and Model Interpretation

The tuned Logistic Regression test predictions produced:
- True Positives: 295
- True Negatives: 764
- False Positives: 271
- False Negatives: 79

The analysis examined errors by:
- Tenure group
- Contract
- Internet service
- Payment method
- Customer cluster

Model interpretation included:
- Logistic Regression coefficients
- Random Forest feature importance

Important predictive signals included contract type, tenure, total charges, monthly charges, internet service, online security, technical support, and payment method.

## Insights and Recommendations

The project translates analytical evidence into practical recommendation areas such as:
- stronger early-tenure onboarding and engagement,
- focused analysis of month-to-month customers,
- investigation of payment/billing patterns,
- examination of support and security-service adoption,
- probability-based churn prioritization.

Recommendations are presented with supporting evidence, potential actions, and trade-offs.

## Limitations and Future Improvements

The project identifies limitations including:
- imperfect predictive performance,
- false positives and false negatives,
- threshold dependence,
- correlated predictors,
- limited granularity of the two-cluster segmentation,
- disagreement between clustering quality metrics.

Potential improvements include:
- probability calibration,
- cost-sensitive threshold selection,
- advanced nonlinear models,
- feature selection,
- explainable AI,
- interaction analysis,
- segment-specific modeling,
- alternative clustering algorithms.

## Project Workflow

Raw Data  
→ Data Inspection  
→ Data Cleaning  
→ Exploratory Data Analysis  
→ Feature Engineering  
→ Customer Segmentation  
→ Churn Prediction  
→ Cross-Validation & Tuning  
→ Error Analysis  
→ Insights & Recommendations  
→ Limitations & Improvements  
→ Final Report & Evidence

## Main Files

- `Yuva_Week6_Customer_Churn_Segmentation.ipynb`
- `Yuva_Week6_Customer_Churn_Integrative_Capstone_Report.docx`
- `Telco-Customer-Churn.csv`
- `Telco-Customer-Churn-Cleaned.csv`
- `Project-Evidence/` — organized datasets, analysis outputs, visualizations, validation results, error analysis, insights, and reflection materials.

## Technologies

Python, Pandas, NumPy, Matplotlib, Seaborn, SciPy, Scikit-learn, Google Colab, K-Means, Logistic Regression, Random Forest.

## Author

Mohammed Aayan
