# Week 1 — Data Acquisition, Cleaning & Preprocessing

## Titanic Dataset Analysis

This project was completed as part of the **Yuva Virtual Data Science with Python Trainee Internship — Week 1**.

The objective was to acquire and inspect a dataset, identify data-quality issues, clean and preprocess the data, detect and treat numerical outliers, and prepare reusable datasets for further analysis and machine-learning workflows.

## Dataset

**Titanic Dataset**

The dataset contains passenger information including age, sex, passenger class, fare, embarkation details, family-related variables, and survival status.

**Initial size:** 891 rows × 15 columns.

## Tools & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab / Jupyter Notebook

## Workflow

### 1. Data Acquisition & Initial Inspection

Loaded the Titanic dataset and examined its dimensions, columns, data types, and overall quality.

### 2. Missing-Value Analysis

Missing values were identified and their proportions were examined. The main issues were found in `age`, `embarked`, `deck`, and `embark_town`.

- `age` — median imputation
- `embarked` — mode imputation
- `deck` — removed because of excessive missingness

### 3. Duplicate Detection

Duplicate records were identified and removed to reduce redundancy and improve consistency.

### 4. Categorical Data Cleaning

Categorical values were inspected and standardized by removing unnecessary whitespace and normalizing text representation.

### 5. Statistical Analysis

Descriptive statistics were generated for numerical variables to understand their central tendency, spread, and value ranges.

### 6. Outlier Detection & Treatment

The **Interquartile Range (IQR)** method was used to identify potential outliers in `age`, `fare`, `sibsp`, and `parch`. IQR-based capping was applied to selected variables, particularly `age` and `fare`, to reduce the influence of extreme values while retaining observations.

### 7. Preprocessing

The cleaned data was prepared for downstream analysis and machine-learning workflows using categorical encoding and numerical feature scaling.

### 8. Visualization & Final Checks

Preliminary visualizations were created to inspect survival patterns and numerical distributions. Final data-quality checks were performed before exporting the datasets.

## Output Files

| File | Description |
|---|---|
| `Yuva_Week1_Titanic_Data_Cleaning.ipynb` | Complete Python notebook and workflow |
| `titanic_cleaned.csv` | Cleaned Titanic dataset |
| `titanic_preprocessed.csv` | Encoded and scaled dataset for further modeling |
| `Yuva_Week1_Titanic_Data_Cleaning_Report.docx` | Detailed report with methodology, results, and evidence |

## Key Learning Outcomes

This project provided hands-on experience with data acquisition, data-quality assessment, missing-value treatment, duplicate removal, categorical-data cleaning, descriptive statistics, IQR-based outlier detection and treatment, feature encoding, scaling, visualization, and dataset export.

## Result

The original Titanic dataset was systematically inspected, cleaned, and transformed into reusable datasets suitable for subsequent exploratory analysis and machine-learning tasks.

---

**Internship:** Yuva Virtual Data Science with Python Trainee Internship  
**Task:** Week 1 — Data Acquisition, Cleaning & Preprocessing  
**Project:** Titanic Dataset Analysis  
**Author:** Mohammed Aayan
