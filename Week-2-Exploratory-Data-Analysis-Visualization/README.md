# Week 2 — Exploratory Data Analysis and Visualization

## Project: Exploratory Data Analysis of the Titanic Dataset

This project was completed as part of Week 2 of the **Yuva Virtual Data Science with Python Trainee Internship**. The objective is to perform exploratory data analysis (EDA) on the Titanic dataset, identify important patterns and relationships, create clear visualizations, and communicate meaningful findings from the data.

## Task Objective

The Week 2 task focuses on:

- Performing initial exploratory analysis and basic statistical analysis.
- Identifying interesting features and patterns in the dataset.
- Creating multiple visualizations to explore distributions, relationships, correlations, and anomalies.
- Interpreting visualizations and explaining possible reasons behind observed patterns.
- Applying and documenting useful transformations and aggregations.
- Presenting the complete EDA process, visualizations, interpretations, and findings in a detailed report.

## Dataset

The **Titanic dataset** was used for this analysis. The dataset contains passenger-level information such as survival status, passenger class, gender, age, number of siblings/spouses, number of parents/children, fare, embarkation details, passenger type, and travelling-alone status.

The analysis begins with the cleaned dataset prepared during Week 1. After duplicate removal and final missing-value handling, the final analytical dataset contains **765 rows and 16 columns**, with **0 missing values** and **0 duplicate rows**.

## Tools and Technologies

- **Python**
- **Pandas** — data loading, manipulation, grouping, and aggregation
- **NumPy** — numerical operations
- **Matplotlib** — data visualization
- **Seaborn** — statistical visualization
- **Google Colab** — notebook development and execution

## EDA Workflow

### 1. Dataset Inspection

The dataset was examined using shape, column, data-type, head/tail, missing-value, duplicate, and unique-value checks to understand its structure and quality.

### 2. Descriptive Statistics

Numerical variables were analyzed using descriptive statistics to understand central tendency, spread, and ranges. Key results include:

- Average age: **29.36 years**
- Median age: **28 years**
- Average fare: **26.59**
- Median fare: **15.90**

### 3. Univariate Analysis

Individual variables were explored through count plots, histograms, and boxplots. The analysis covered:

- Overall survival distribution
- Passenger age distribution
- Fare distribution
- Passenger class distribution
- Gender distribution
- Port of embarkation distribution

### 4. Bivariate Analysis

Relationships between survival and important passenger characteristics were investigated, including:

- Gender vs. survival
- Passenger class vs. survival
- Age vs. survival
- Fare vs. survival
- Age vs. fare
- Family size vs. survival

### 5. Feature Engineering

Two derived features were created to support deeper analysis:

- **family_size** = `sibsp + parch + 1`
- **age_group** — passengers grouped into Child, Teenager, Young Adult, Adult, Senior Adult, and Senior categories.

### 6. Multivariate Analysis

Multiple variables were examined together, including passenger class and gender, age groups, embarkation port, family-related variables, and numerical correlations.

### 7. Correlation and Anomaly Analysis

A correlation heatmap was used to explore relationships among numerical variables. The IQR approach was also used to examine potential anomalies in numerical variables such as age and fare.

## Key Findings

- The final analytical dataset contains **765 passengers and 16 variables**.
- The overall survival rate is **41.29%**.
- Female passengers had a substantially higher survival rate (**73.97%**) than male passengers (**21.53%**).
- First-class passengers had the highest survival rate (**63.33%**), followed by second class (**50.61%**) and third class (**25.94%**).
- Survivors generally had higher fare values than non-survivors, highlighting the relationship between fare, passenger class, and survival.
- The combination of class and gender shows a strong difference in survival outcomes across groups.
- Among the displayed age groups, children had the highest survival rate at approximately **57.35%**.
- Cherbourg had the highest observed survival rate among the embarkation ports at approximately **58.06%**.
- Family-size analysis suggests that moderate travelling groups can have higher survival rates, while very large groups show lower observed survival rates.

## Visualizations

The project includes multiple visualizations with appropriate titles, labels, and legends where applicable. These visualizations are used to communicate distributions, group comparisons, relationships, correlations, and survival patterns.

## Project Files

```text
Week-2-Exploratory-Data-Analysis-Visualization/
├── README.md
├── Yuva_Week2_Titanic_EDA_Visualization.ipynb
├── titanic_eda.csv
└── Yuva_Week2_Titanic_EDA_Report_HD.docx
```

### Notebook

`Yuva_Week2_Titanic_EDA_Visualization.ipynb` contains the complete executed Python workflow for the Week 2 exploratory analysis and visualization.

### Dataset

`titanic_eda.csv` contains the final EDA dataset after the analysis and data-quality checks.

### Report

`Yuva_Week2_Titanic_EDA_Report_HD.docx` documents the EDA methodology, statistical analysis, visualizations, interpretations, transformations, key findings, and conclusion.

## Conclusion

This project demonstrates the use of Python-based exploratory data analysis to understand a real-world dataset and communicate insights visually. The analysis shows clear differences in survival across gender and passenger class while also exploring the roles of age, fare, family size, embarkation port, and other passenger characteristics. The completed EDA provides a structured foundation for further data science and machine-learning work.

## Author

**Mohammed Aayan**

B.Tech — Computer Science and Information Technology