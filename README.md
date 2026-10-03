# Virtual Data Science with Python

A collection of practical **Data Science, Machine Learning, and Deep Learning projects** completed as part of the **Yuva Virtual Data Science with Python Trainee Internship**. This repository documents the progression from data acquisition and preprocessing through exploratory analysis, unsupervised learning, supervised learning, deep learning, and an integrative capstone project.

The projects are implemented in **Python** using Jupyter/Google Colab and follow a structured workflow of data preparation, analysis, modeling, evaluation, interpretation, error analysis, and documentation.

---

## Internship Progress

| Week | Area | Project | Status |
|---|---|---|---|
| Week 1 | Data Acquisition, Cleaning & Preprocessing | Titanic Dataset Analysis | Completed |
| Week 2 | Exploratory Data Analysis & Visualization | Titanic Dataset EDA | Completed |
| Week 3 | Unsupervised Learning & Clustering | Customer Segmentation Using K-Means | Completed |
| Week 4 | Supervised Learning | Breast Cancer Classification | Completed |
| Week 5 | Deep Learning | Fashion-MNIST Image Classification Using CNN | Completed |
| Week 6 | Integrative Capstone Project | Customer Churn Prediction and Customer Segmentation | Completed |

---

# Week 1 — Data Acquisition, Cleaning & Preprocessing

## Titanic Dataset Analysis

The first project focused on preparing a raw dataset for reliable analysis and future machine-learning workflows.

### Main Work

- Acquired and inspected the Titanic dataset.
- Analyzed missing values and their proportions.
- Applied median and mode imputation where appropriate.
- Removed highly incomplete attributes.
- Detected and removed duplicate records.
- Standardized categorical values.
- Generated descriptive statistics.
- Detected numerical outliers using the IQR method.
- Applied IQR-based capping to selected numerical variables.
- Performed categorical encoding and numerical scaling.
- Created preliminary visualizations and final data-quality checks.

### Dataset Result

The original dataset contained **891 rows × 15 columns**. After cleaning and duplicate removal, the main working dataset contained **775 rows × 14 columns**.

### Output Files

- `Yuva_Week1_Titanic_Data_Cleaning.ipynb`
- `titanic_cleaned.csv`
- `titanic_preprocessed.csv`
- `Yuva_Week1_Titanic_Data_Cleaning_Report.docx`

---

# Week 2 — Exploratory Data Analysis & Visualization

## Exploratory Data Analysis of the Titanic Dataset

The second project continued with the Titanic dataset and focused on understanding patterns, relationships, distributions, and trends through exploratory data analysis and visualization.

### Main Work

- Performed dataset inspection and data-quality validation.
- Calculated descriptive statistics.
- Conducted univariate, bivariate, and multivariate analysis.
- Examined survival patterns by gender and passenger class.
- Analyzed age, fare, family size, embarkation port, and other passenger characteristics.
- Created derived features such as `family_size` and `age_group`.
- Performed correlation and anomaly analysis.
- Created multiple statistical visualizations using Matplotlib and Seaborn.
- Interpreted visual patterns and documented findings.

### Key Results

- Final analytical dataset: **765 rows × 16 columns**.
- Missing values: **0**.
- Duplicate rows: **0**.
- Overall survival rate: **41.29%**.
- Female survival rate: **73.97%**.
- Male survival rate: **21.53%**.
- First-class survival rate: **63.33%**.

### Output Files

- `Yuva_Week2_Titanic_EDA_Visualization.ipynb`
- `titanic_eda.csv`
- `Yuva_Week2_Titanic_EDA_Report.docx`

---

# Week 3 — Unsupervised Learning & Clustering

## Customer Segmentation Using K-Means Clustering

The third project introduced unsupervised machine learning by applying **K-Means clustering** to the Mall Customers Dataset. The objective was to identify customer groups with similar demographic and spending characteristics.

### Main Work

- Inspected dataset structure and quality.
- Performed exploratory analysis of age, income, spending score, and gender.
- Selected `Age`, `Annual Income (k$)`, and `Spending Score (1-100)` as clustering features.
- Excluded `CustomerID` because it is an identifier rather than a meaningful clustering feature.
- Retained gender for post-clustering descriptive analysis.
- Assessed potential outliers using the IQR method.
- Applied `StandardScaler` before distance-based clustering.
- Evaluated **K = 2 to 10** using the Elbow Method and Silhouette Score.
- Trained the final K-Means model.
- Profiled and interpreted the resulting customer segments.
- Generated cluster visualizations and business-oriented insights.

### Key Results

- Dataset size: **200 customers**.
- Clustering features: **3**.
- Candidate K values: **2–10**.
- Selected clusters: **K = 6**.
- Silhouette Score: **0.4284**.
- Final clustered dataset: **200 rows × 6 columns**.

### Identified Segments

1. **Mature Moderate Customers**
2. **Young Moderate Customers**
3. **High-Income Low-Spending Customers**
4. **High-Income High-Spending Customers**
5. **Young High-Spending Customers**
6. **Low-Income Low-Spending Customers**

### Output Files

- `Mall_Customers.csv`
- `Mall_Customers_Clustered.csv`
- `Yuva_Week3_Mall_Customer_Clustering.ipynb`
- `Yuva_Week3_Mall_Customer_Clustering_Report.docx`

---

# Week 4 — Supervised Learning

## Breast Cancer Classification Using Machine Learning

The fourth project focused on **supervised machine learning** and binary classification using the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset. The objective was to classify observations as **Benign (B)** or **Malignant (M)** using numerical diagnostic measurements.

### Main Work

- Loaded and inspected the WDBC dataset and supporting documentation.
- Performed missing-value, duplicate, data-type, and target-distribution checks.
- Conducted exploratory data analysis.
- Encoded the target as **B = 0** and **M = 1**.
- Removed the identifier from the predictive feature set.
- Split the dataset into **80% training and 20% testing** using stratification.
- Applied `StandardScaler` for Logistic Regression.
- Trained **Logistic Regression** as the primary model.
- Trained **Random Forest** as a comparison model.
- Evaluated the models using Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
- Used confusion matrices, ROC curves, and 5-fold stratified cross-validation.
- Performed feature-importance analysis and error analysis.

### Model Results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | **96.49%** | **96.49%** |
| Precision | **97.50%** | **100.00%** |
| Recall | **92.86%** | **90.48%** |
| F1 Score | **95.12%** | **95.00%** |
| ROC-AUC | **0.9960** | **0.9970** |
| 5-Fold CV Accuracy | **97.37%** | **96.13%** |

### Output Files

- `wdbc.data`
- `wdbc.names`
- `breast_cancer_dataset.csv`
- `breast_cancer_predictions.csv`
- `Yuva_Week4_Breast_Cancer_Classification.ipynb`
- `Yuva_Week4_Breast_Cancer_Classification_Report.docx`

> **Note:** This project is for educational machine-learning purposes and is not intended for clinical diagnosis or medical decision-making.

---

# Week 5 — Deep Learning

## Fashion-MNIST Image Classification Using Convolutional Neural Networks

The fifth project introduced **deep learning** for image classification using **TensorFlow/Keras** and the Fashion-MNIST dataset.

### Main Work

- Loaded and preprocessed 28×28 grayscale Fashion-MNIST images.
- Built a baseline **Multilayer Perceptron (MLP)**.
- Designed a **Convolutional Neural Network (CNN)** for image classification.
- Used convolution, Batch Normalization, ReLU activation, Max Pooling, Dropout, Global Average Pooling, and Dense layers.
- Trained and evaluated the models using appropriate classification metrics.
- Used Early Stopping and learning-rate scheduling during training.
- Compared CNN performance with the MLP baseline.
- Conducted controlled experiments using heavy and light image augmentation.
- Generated learning curves, confusion matrices, class-level accuracy, and error analysis.
- Investigated resource constraints and model generalization.

### Final CNN Results

The selected **CNN without augmentation** achieved:

| Metric | Result |
|---|---:|
| Test Accuracy | **91.91%** |
| Precision | **92.00%** |
| Recall | **91.91%** |
| F1-Score | **91.93%** |
| Test Loss | **0.2287** |
| Correct Predictions | **9,191 / 10,000** |

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Baseline MLP | 89.29% | 89.36% | 89.29% | 89.31% |
| CNN — No Augmentation | **91.91%** | **92.00%** | **91.91%** | **91.93%** |
| CNN — Heavy Augmentation | 87.03% | 88.23% | 87.03% | 87.30% |
| CNN — Light Augmentation | 82.84% | 83.44% | 82.84% | 82.42% |

### Key Findings

- The CNN without augmentation achieved higher test accuracy than the baseline MLP.
- The final model correctly classified **9,191 of 10,000** test images.
- **Bag, Sneaker, Trouser, and Ankle boot** had particularly strong class-level performance.
- **Shirt** was the most difficult class, with **78.8%** class accuracy.
- Major errors occurred between visually similar upper-body classes such as Shirt, T-shirt/top, Coat, and Pullover.
- The augmentation experiments showed lower performance than the non-augmented CNN in these controlled experiments.

### Output Files

- `Yuva_Week5_Fashion_MNIST_Deep_Learning.ipynb`
- `Yuva_Week5_Fashion_MNIST_Deep_Learning_Report.docx`
- `fashion_mnist_final_results.zip`

---

# Week 6 — Integrative Capstone Project

## Customer Churn Prediction and Customer Segmentation Using Data Science

Week 6 integrates the core skills practiced throughout the internship into a complete end-to-end Data Science project. The capstone uses the **IBM Telco Customer Churn** dataset to analyze churn patterns, engineer customer features, create customer segments with K-Means clustering, predict churn with supervised learning, validate and tune models, analyze errors, and translate the findings into evidence-based recommendations.

### Problem and Objective

The project investigates the following practical problem:

**How can customer data be analyzed and modeled to identify churn patterns, group customers with similar profiles, predict potential churn, and support evidence-based retention planning?**

The project covers:
- data inspection and cleaning,
- exploratory data analysis,
- feature engineering,
- unsupervised customer segmentation,
- supervised churn prediction,
- cross-validation and hyperparameter tuning,
- threshold analysis,
- error analysis and model interpretation,
- insights, recommendations, limitations, and future improvements.

### Dataset

- **Dataset:** Telco Customer Churn
- **Original size:** **7,043 rows × 21 columns**
- **Target:** `Churn` (`Yes` / `No`)
- **Identifier excluded from modeling:** `customerID`
- **Source:** IBM sample Telco Customer Churn dataset

### Data Cleaning & Validation

Key preprocessing findings:
- The raw dataset contained 7,043 records and 21 variables.
- No duplicate rows were identified.
- `TotalCharges` was stored as an object/text field even though it represents a numerical quantity.
- **11 blank `TotalCharges` values** were identified.
- All 11 blank records had **tenure = 0**; these were converted to numeric and represented as **0.0** after validation.
- Final cleaned dataset: **7,043 × 21**.
- Final missing values: **0**.
- Final duplicate rows: **0**.
- Final `TotalCharges` data type: **float64**.

### Exploratory Data Analysis

The EDA examined:
- overall churn distribution,
- numerical distributions,
- churn rates by contract, internet service, payment method, tenure group, and service adoption,
- correlations,
- categorical association tests using **Cramér's V**,
- numerical group comparisons using **Mann-Whitney U tests**.

Observed churn patterns in the dataset included:
- Overall churn: **26.54%**
- Month-to-month contract: **42.71%**
- One-year contract: **11.27%**
- Two-year contract: **2.83%**
- Fiber optic internet: **41.89%**
- Electronic check payment: **45.29%**
- 0–6 month tenure group: **52.94%**
- Customers with more than 6 months tenure: **19.51%**

These are descriptive associations in the dataset and are not treated as causal effects.

### Feature Engineering

Additional features were created to support deeper analysis and modeling:

- `Churn_Binary`
- `Num_Services`
- `SecuritySupport_Count`
- `Streaming_Count`
- `Tenure_Group`
- `Is_New_Customer`
- `Above_Median_MonthlyCharge`

For the segmentation task, churn was excluded to avoid target leakage.

### Customer Segmentation — K-Means

Seven standardized numerical features were used:
- `SeniorCitizen`
- `tenure`
- `MonthlyCharges`
- `TotalCharges`
- `Num_Services`
- `SecuritySupport_Count`
- `Streaming_Count`

Candidate values **K = 2 to 10** were evaluated using:
- Inertia / Elbow method
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score

The selected **K = 2** solution produced:
- Silhouette Score: **0.3901**
- Davies-Bouldin Index: **1.0622**
- Calinski-Harabasz Score: **5586.3037**

The clustering metrics were not fully aligned: Silhouette and Calinski-Harabasz supported K = 2, while Davies-Bouldin reached its best value at K = 8. This trade-off is documented in the project evidence and final report.

#### Cluster Profiles

| Cluster | Customers | Share | Mean Tenure | Mean Monthly Charges | Mean Total Charges | Mean Services | Observed Churn |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cluster 0 | 2,873 | 40.79% | 49.45 | 89.31 | 4,410.58 | 5.43 | 21.34% |
| Cluster 1 | 4,170 | 59.21% | 20.60 | 47.85 | 811.65 | 1.94 | 30.12% |

Churn was not used to create the clusters; churn rates were examined afterward for interpretation.

### Churn Prediction

An **80/20 stratified train-test split** was used:
- Training: **5,634**
- Testing: **1,409**
- Churn rate preserved at **26.54%** in both partitions

Models evaluated:
- Dummy majority-class baseline
- Logistic Regression
- Random Forest

#### Initial Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Dummy Baseline | 73.46% | 0.00% | 0.00% | 0.00% | 0.5000 | 0.2654 |
| Logistic Regression | 80.27% | 66.11% | 52.67% | 58.63% | 0.8458 | 0.6525 |
| Random Forest | 77.86% | 60.62% | 47.33% | 53.15% | 0.8188 | 0.6078 |

The dummy baseline demonstrates why accuracy alone is not sufficient for an imbalanced churn-classification problem.

### Model Validation, Tuning & Threshold Analysis

Five-fold stratified cross-validation produced:

| Model | CV Accuracy | CV Precision | CV Recall | CV F1 | CV ROC-AUC | CV PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8078 ± 0.0138 | 0.6713 ± 0.0308 | 0.5398 ± 0.0403 | 0.5979 ± 0.0335 | 0.8481 ± 0.0113 | 0.6635 ± 0.0154 |
| Random Forest | 0.7858 ± 0.0131 | 0.6235 ± 0.0338 | 0.4876 ± 0.0283 | 0.5470 ± 0.0282 | 0.8227 ± 0.0125 | 0.6143 ± 0.0289 |

GridSearchCV identified:
- Logistic Regression: **C = 0.01**, **class_weight = balanced**
- Random Forest: **n_estimators = 250**, **max_depth = None**, **min_samples_leaf = 3**, **class_weight = balanced**

Tuned test results:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Tuned Logistic Regression | 75.16% | 52.12% | 78.88% | 62.77% | 0.8451 | 0.6537 |
| Tuned Random Forest | 77.08% | 55.48% | 68.98% | 61.50% | 0.8367 | 0.6444 |

A threshold analysis on out-of-fold probabilities identified **0.52** as the best tested threshold for the defined OOF F1 objective. On the test set, the threshold-adjusted Logistic Regression recorded:
- Accuracy: **75.87%**
- Precision: **53.14%**
- Recall: **77.01%**
- F1: **62.88%**
- ROC-AUC: **0.8451**
- PR-AUC: **0.6537**

### Error Analysis & Model Interpretation

For the tuned Logistic Regression test predictions:
- True Positives: **295**
- True Negatives: **764**
- False Positives: **271**
- False Negatives: **79**
- Total errors: **350 / 1,409 (24.84%)**

Errors were further examined by:
- tenure group,
- contract type,
- internet service,
- payment method,
- customer cluster,
- prediction type.

Interpretation outputs included:
- Logistic Regression feature coefficients
- Random Forest feature importance

Important model signals included contract type, tenure, total charges, monthly charges, internet service, online security, technical support, and payment method.

### Key Insights & Recommendations

Evidence-based analytical areas identified from the dataset include:
- **Early-tenure risk:** 0–6 month customers showed 52.94% observed churn.
- **Contract pattern:** month-to-month customers showed 42.71% observed churn.
- **Payment pattern:** electronic-check customers showed 45.29% observed churn.
- **Service-support pattern:** customers without OnlineSecurity or TechSupport showed higher observed churn than customers with those services.
- **Segment-level differences:** the two K-Means clusters had observed churn rates of 21.34% and 30.12%.

Recommendation areas documented in the project include:
- stronger early-lifecycle onboarding and engagement,
- focused analysis of month-to-month customers,
- investigation of billing and payment experience,
- evaluation of support/security-service adoption,
- probability-based prioritization of churn-risk cases.

Recommendations are accompanied by evidence, possible actions, and trade-offs in the project evidence.

### Limitations & Further Improvements

The project documents:
- imperfect predictive performance,
- false positives and false negatives,
- threshold dependence,
- correlated predictors,
- limited clustering granularity at K = 2,
- disagreement between clustering quality metrics.

Concrete improvement areas include:
- probability calibration,
- cost-sensitive threshold selection,
- advanced nonlinear models,
- feature selection,
- explainable AI,
- interaction analysis,
- segment-specific modeling,
- alternative clustering algorithms.

### Week 6 Project Workflow

**Raw Data → Data Inspection → Data Cleaning → EDA → Feature Engineering → Customer Segmentation → Churn Prediction → Cross-Validation & Hyperparameter Tuning → Error Analysis & Model Interpretation → Insights & Recommendations → Limitations & Improvements → Final Report & Submission**

### Week 6 Files

- `Yuva_Week6_Customer_Churn_Segmentation.ipynb`
- `Yuva_Week6_Customer_Churn_Integrative_Capstone_Report.docx`
- `Telco-Customer-Churn.csv`
- `Telco-Customer-Churn-Cleaned.csv`
- `Project-Evidence/`
  - `01-Data-Cleaning/`
  - `02-EDA/`
  - `03-Feature-Engineering/`
  - `04-Customer-Segmentation/`
  - `05-Churn-Prediction/`
  - `06-Model-Validation/`
  - `07-Error-Analysis/`
  - `08-Insights-Recommendations/`
  - `09-Reflection-Improvements/`
  - `10-Project-Overview/`

The Project-Evidence directory contains the supporting datasets, CSV outputs, visualizations, statistical results, model-validation outputs, error-analysis files, recommendations, reflection files, and the end-to-end pipeline visual.

---

# Technology Stack

### Programming

- **Python**

### Data Processing & Analysis

- **Pandas**
- **NumPy**
- **SciPy**

### Data Visualization

- **Matplotlib**
- **Seaborn**

### Machine Learning

- **Scikit-learn**
  - StandardScaler
  - K-Means Clustering
  - Silhouette Score
  - Davies-Bouldin Index
  - Calinski-Harabasz Score
  - Logistic Regression
  - Random Forest
  - Classification metrics
  - Cross-validation
  - GridSearchCV

### Deep Learning

- **TensorFlow**
- **Keras**

### Development Environment

- **Google Colab**
- **Jupyter Notebook**

---

# Repository Structure

```text
Virtual-Data-Science-Python-Internship/
│
├── README.md
│
├── Week-1-Data-Acquisition-Cleaning-Preprocessing/
├── Week-2-Exploratory-Data-Analysis-Visualization/
├── Week-3-Unsupervised-Learning-Clustering/
├── Week-4-Supervised-Learning/
├── Week-5-Deep-Learning/
│
└── Week-6-Integrative-Capstone-Project/
    ├── README.md
    ├── Telco-Customer-Churn.csv
    ├── Telco-Customer-Churn-Cleaned.csv
    ├── Yuva_Week6_Customer_Churn_Segmentation.ipynb
    ├── Yuva_Week6_Customer_Churn_Integrative_Capstone_Report.docx
    │
    └── Project-Evidence/
        ├── 01-Data-Cleaning/
        ├── 02-EDA/
        ├── 03-Feature-Engineering/
        ├── 04-Customer-Segmentation/
        ├── 05-Churn-Prediction/
        ├── 06-Model-Validation/
        ├── 07-Error-Analysis/
        ├── 08-Insights-Recommendations/
        ├── 09-Reflection-Improvements/
        └── 10-Project-Overview/
            ├── 00_project_pipeline.png
            └── README.md
```

---

# Learning Progression

The repository demonstrates a progressive data-science workflow:

**Data → Cleaning → Exploration → Visualization → Unsupervised Learning → Supervised Learning → Deep Learning → Integrative Capstone**

Across the six completed weeks, the projects demonstrate practical experience in:

- Data acquisition, cleaning, validation, and preprocessing
- Exploratory data analysis and visualization
- Feature engineering
- Statistical testing and correlation analysis
- Unsupervised customer segmentation
- Supervised classification
- Model validation, hyperparameter tuning, and threshold analysis
- Neural-network and CNN development
- Image classification
- Error analysis and model interpretation
- Evidence-based recommendations
- Reflection, limitations, and technical improvement planning
- Technical reporting and documentation

---

# Current Status

**Weeks 1–6: Completed**

The repository now contains the completed internship work through the **Week 6 Integrative Capstone Project**, including the final notebook, cleaned/raw datasets, comprehensive report, organized project evidence, visualizations, model-validation outputs, error analysis, recommendations, and reflection materials.

---

# Author

**Mohammed Aayan**  
B.Tech — Computer Science & Information Technology

**Yuva Virtual Data Science with Python Trainee Internship**

---

## Author

**Mohammed Aayan**  
B.Tech — Computer Science & Information Technology
