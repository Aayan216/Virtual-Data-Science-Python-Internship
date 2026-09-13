# Virtual Data Science with Python

A collection of practical **Data Science and Machine Learning projects** completed as part of the **Yuva Virtual Data Science with Python Trainee Internship**. This repository documents the progression from data acquisition and preprocessing to exploratory analysis, unsupervised learning, and supervised machine learning.

The projects are implemented in **Python** using Jupyter/Google Colab and follow a structured workflow of data preparation, analysis, modeling, evaluation, interpretation, and documentation.

---

## Internship Progress

| Week | Area | Project | Status |
|---|---|---|---|
| Week 1 | Data Acquisition, Cleaning & Preprocessing | Titanic Dataset Analysis | Completed |
| Week 2 | Exploratory Data Analysis & Visualization | Titanic Dataset EDA | Completed |
| Week 3 | Unsupervised Learning & Clustering | Customer Segmentation Using K-Means | Completed |
| Week 4 | Supervised Learning | Breast Cancer Classification | Completed |
| Week 5 | Deep Learning | To be completed | Pending |
| Week 6 | Integrative Capstone Project | To be completed | Pending |

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

Logistic Regression was selected as the preferred model because it achieved higher recall, F1 Score, and mean cross-validation accuracy, while Random Forest achieved slightly higher precision and ROC-AUC.

### Output Files

- `wdbc.data`
- `wdbc.names`
- `breast_cancer_dataset.csv`
- `breast_cancer_predictions.csv`
- `Yuva_Week4_Breast_Cancer_Classification.ipynb`
- `Yuva_Week4_Breast_Cancer_Classification_Report.docx`

> **Note:** This project is for educational machine-learning purposes and is not intended for clinical diagnosis or medical decision-making.

---

# Technology Stack

### Programming

- **Python**

### Data Processing & Analysis

- **Pandas**
- **NumPy**

### Data Visualization

- **Matplotlib**
- **Seaborn**

### Machine Learning

- **Scikit-learn**
  - StandardScaler
  - K-Means Clustering
  - Silhouette Score
  - Logistic Regression
  - Random Forest
  - Classification metrics
  - Cross-validation

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
│   ├── README.md
│   ├── Yuva_Week1_Titanic_Data_Cleaning.ipynb
│   ├── titanic_cleaned.csv
│   ├── titanic_preprocessed.csv
│   └── Yuva_Week1_Titanic_Data_Cleaning_Report.docx
│
├── Week-2-Exploratory-Data-Analysis-Visualization/
│   ├── README.md
│   ├── Yuva_Week2_Titanic_EDA_Visualization.ipynb
│   ├── titanic_eda.csv
│   └── Yuva_Week2_Titanic_EDA_Report.docx
│
├── Week-3-Unsupervised-Learning-Clustering/
│   ├── README.md
│   ├── Mall_Customers.csv
│   ├── Mall_Customers_Clustered.csv
│   ├── Yuva_Week3_Mall_Customer_Clustering.ipynb
│   └── Yuva_Week3_Mall_Customer_Clustering_Report.docx
│
├── Week-4-Supervised-Learning/
│   ├── README.md
│   ├── wdbc.data
│   ├── wdbc.names
│   ├── breast_cancer_dataset.csv
│   ├── breast_cancer_predictions.csv
│   ├── Yuva_Week4_Breast_Cancer_Classification.ipynb
│   └── Yuva_Week4_Breast_Cancer_Classification_Report.docx
│
├── Week-5-Deep-Learning/
│   └── README.md
│
└── Week-6-Integrative-Capstone-Project/
    └── README.md
```

---

# Learning Progression

The repository demonstrates a progressive data-science workflow:

**Data → Cleaning → Exploration → Visualization → Unsupervised Learning → Supervised Learning → Deep Learning → Capstone**

Through the completed first four weeks, the projects demonstrate practical experience in preparing datasets, analyzing patterns, visualizing data, discovering customer segments, building classification models, evaluating machine-learning performance, and communicating results through documented reports.

---

# Future Work

The remaining internship stages will extend this foundation into:

- **Week 5:** Deep Learning application and neural-network workflows.
- **Week 6:** End-to-end integrative capstone project combining the skills developed throughout the internship.

---

# Author

**Mohammed Aayan**  
B.Tech — Computer Science & Information Technology

**Yuva Virtual Data Science with Python Trainee Internship**