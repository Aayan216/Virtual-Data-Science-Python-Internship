# Week 3 — Unsupervised Learning and Clustering

## Customer Segmentation Using K-Means Clustering

This project applies **unsupervised machine learning** to the Mall Customers Dataset to identify groups of customers with similar demographic and spending characteristics. The analysis follows a complete clustering workflow from data inspection and exploratory analysis to feature scaling, optimal cluster selection, K-Means modeling, cluster profiling, visualization, and interpretation.

## Objectives

- Inspect and validate the customer dataset.
- Perform exploratory data analysis on age, annual income, spending score, and gender.
- Select meaningful numerical features for clustering.
- Identify potential outliers using the IQR method without automatically removing valid observations.
- Standardize clustering features using `StandardScaler`.
- Evaluate candidate cluster counts from **K = 2 to 10** using the Elbow Method and Silhouette Score.
- Build and evaluate the final K-Means clustering model.
- Profile and interpret each customer segment using numerical summaries and visualizations.
- Translate the discovered segments into practical business insights.

## Dataset

The **Mall Customers Dataset** contains 200 customer records with the following original attributes:

| Column | Description | Role |
|---|---|---|
| `CustomerID` | Unique customer identifier | Identifier; excluded from clustering |
| `Gender` | Customer gender | Descriptive analysis only |
| `Age` | Customer age in years | Clustering feature |
| `Annual Income (k$)` | Annual income in thousands of dollars | Clustering feature |
| `Spending Score (1-100)` | Customer spending score | Clustering feature |

The final clustered dataset contains **200 rows and 6 columns**, with the `Cluster` label added after model training.

### Data Quality

- Missing values: **0**
- Duplicate rows: **0**
- Age range: **18–70 years**
- Annual income range: **$15k–$137k**
- Spending score range: **1–99**

## Methodology

### 1. Data Inspection and EDA

The dataset was inspected for structure, data types, missing values, duplicates, unique values, and gender distribution. Exploratory visualizations were used to understand the distributions and relationships among age, annual income, and spending score.

### 2. Feature Selection

The clustering model uses:

- **Age**
- **Annual Income (k$)**
- **Spending Score (1-100)**

`CustomerID` was excluded because it is only an identifier, while `Gender` was retained for descriptive post-clustering analysis rather than used as a numerical clustering feature.

### 3. Outlier Assessment

Potential outliers were evaluated using the **Interquartile Range (IQR)** method. The assessment identified 2 potential observations in annual income, while age and spending score had no IQR-based potential outliers. The observations were retained because they are valid customer records and removing them could discard meaningful customer segments.

### 4. Feature Scaling

`StandardScaler` was applied to the three clustering features so that age, income, and spending score contribute on comparable scales to the distance-based K-Means algorithm.

### 5. Selecting the Number of Clusters

Candidate values from **K = 2 through K = 10** were evaluated using:

- **Elbow Method** — to examine the reduction in within-cluster variation.
- **Silhouette Score** — to measure how well-separated and internally cohesive the clusters are.

The highest observed Silhouette Score was **0.4284 at K = 6**, so six clusters were selected for the final model.

### 6. K-Means Model

The final model was trained using:

- Algorithm: **K-Means Clustering**
- Number of clusters: **6**
- `random_state`: **42**
- `n_init`: **10**

## Final Results

| Metric | Result |
|---|---:|
| Customers | **200** |
| Final clusters | **6** |
| Silhouette Score | **0.4284** |
| Clustering features | **3** |

### Cluster Profiles

| Cluster | Customers | Avg Age | Avg Income (k$) | Avg Spending Score |
|---|---:|---:|---:|---:|
| 0 | 45 | 56.33 | 54.27 | 49.07 |
| 1 | 39 | 26.79 | 57.10 | 48.13 |
| 2 | 33 | 41.94 | 88.94 | 16.97 |
| 3 | 39 | 32.69 | 86.54 | 82.13 |
| 4 | 23 | 25.00 | 25.26 | 77.61 |
| 5 | 21 | 45.52 | 26.29 | 19.38 |

### Segment Interpretation

- **Cluster 0 — Mature Moderate Customers:** Older customers with moderate income and moderate spending behaviour.
- **Cluster 1 — Young Moderate Customers:** Younger customers with moderate income and moderate spending scores.
- **Cluster 2 — High-Income Low-Spending Customers:** Higher-income customers whose spending score is comparatively low, indicating potential scope for targeted engagement.
- **Cluster 3 — High-Income High-Spending Customers:** Younger, high-income customers with high spending scores, representing a strong-value customer segment.
- **Cluster 4 — Young High-Spending Customers:** Young customers with relatively low income but high spending scores.
- **Cluster 5 — Low-Income Low-Spending Customers:** Older, lower-income customers with low spending scores.

The overall dataset averages are **38.85 years** for age, **$60.56k** for annual income, and **50.20** for spending score. The dataset contains **112 female customers (56%)** and **88 male customers (44%)**.

## Key Insights

1. Customer behaviour is not explained by income alone; the clustering reveals distinct combinations of age, income, and spending behaviour.
2. The high-income/high-spending segment (Cluster 3) is clearly differentiated from the high-income/low-spending segment (Cluster 2), suggesting different engagement strategies may be appropriate.
3. Cluster 4 combines low average income with a high spending score, indicating that spending behaviour can differ substantially from income level.
4. The Silhouette Score of **0.4284** indicates useful but not perfectly separated customer groups, so the clusters should be treated as descriptive segments rather than absolute categories.

## Technologies and Libraries

- **Python**
- **Pandas** — data loading, cleaning, aggregation, and analysis
- **NumPy** — numerical operations
- **Matplotlib** — visualization
- **Seaborn** — statistical visualization
- **Scikit-learn** — StandardScaler, KMeans, and Silhouette Score
- **Google Colab / Jupyter Notebook** — development and execution environment

## Repository Files

```text
Week-3-Unsupervised-Learning-Clustering/
├── README.md
├── Mall_Customers.csv
├── Mall_Customers_Clustered.csv
├── Yuva_Week3_Mall_Customer_Clustering.ipynb
└── Yuva_Week3_Mall_Customer_Clustering_Report.docx
```

### Outputs

- `Mall_Customers.csv` — original dataset used for analysis.
- `Mall_Customers_Clustered.csv` — final dataset containing the assigned cluster for each customer.
- `Yuva_Week3_Mall_Customer_Clustering.ipynb` — complete executable analysis and clustering workflow.
- `Yuva_Week3_Mall_Customer_Clustering_Report.docx` — detailed project report containing methodology, visualizations, results, interpretations, and conclusions.

## Conclusion

The Week 3 project demonstrates a complete **K-Means customer segmentation workflow** using unsupervised learning. After data validation, feature selection, outlier assessment, and standardization, six customer segments were identified as the most suitable solution based on the evaluated clustering metrics. The resulting profiles provide interpretable groups based on age, annual income, and spending behaviour and demonstrate how clustering can support data-driven customer segmentation.
