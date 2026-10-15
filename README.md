# Data Science & Machine Learning Coursework Suite

A comprehensive collection of end-to-end data science projects, statistical analyses, unsupervised clustering, and supervised machine learning pipelines developed as part of the Data Science curriculum at Shahid Beheshti University (SBU).

Each module is self-contained with executable Jupyter notebooks, source datasets, technical documentation, and formal academic PDF reports.

---

## Repository Overview

```text
Data-Science-Assignments/
├── Assignment-1-House-Price-Prediction/
│   ├── house-prices-prediction-using-tfdf.ipynb
│   ├── Data analysis report.pdf
│   └── readme.md
├── Assignment-2-Airline-Delays-and-Statistical-Testing/
│   ├── EDA_Airline_Delays.ipynb
│   ├── EDA_Airline_Delays_Report.pdf
│   ├── Statistical_Testing_Ecommerce.ipynb
│   ├── Statistical_Testing_Ecommerce_Report.pdf
│   └── readme.md
├── Assignment-3-Country-Socioeconomic-Clustering/
│   ├── Country-data.csv
│   ├── country_clustering.ipynb
│   ├── Clustering_Analysis_Report.pdf
│   └── readme.md
├── Assignment-4-Customer-Churn-Prediction-Pipeline/
│   ├── train.csv
│   ├── test.csv
│   ├── predictions.csv
│   ├── customer_churn_prediction.ipynb
│   ├── Customer_Churn_Pipeline_Report.pdf
│   └── readme.md
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Project Modules Summary

| Module | Topic & Domain | Task Type | Core Algorithms & Methods | Key Datasets | Primary Deliverables |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Assignment 1** | **House Price Prediction** | Supervised Regression | TensorFlow Decision Forests (TF-DF), Random Forest, OOB Error, Feature Importance | Ames Housing Dataset | Notebook, Formal PDF Report |
| **Assignment 2 (Part A)** | **Airline Delays & Cancellations** | Exploratory Data Analysis (EDA) | Multivariate distributions, temporal aggregation, cancellation root-cause decomposition | US Multi-Year Flight Records | Notebook, Formal PDF Report |
| **Assignment 2 (Part B)** | **E-Commerce Statistical Testing** | Inferential Hypothesis Testing | One-Way ANOVA, Chi-Square ($\chi^2$), Tukey HSD post-hoc, Cohen's $f$, Cramér's $V$ | Retail E-Commerce Transactions | Notebook, Formal PDF Report |
| **Assignment 3** | **Country Socioeconomic Clustering** | Unsupervised Learning | PCA, K-Means Clustering, Agglomerative Hierarchical Clustering (Ward/Complete/Average), Hopkins Statistic | Global Socio-economic Indicators (167 countries) | Dataset, Notebook, Formal PDF Report |
| **Assignment 4** | **Telecom Customer Churn Pipeline** | Supervised Classification | Scikit-Learn Pipelines, SMOTE, Class Weighting, Logistic Regression, Random Forest, Gradient Boosting, ROC-AUC | Telecom Customer Churn Dataset | Datasets, Predictions, Notebook, Formal PDF Report |

---

## Detailed Module Descriptions

### 1. House Price Prediction Using TensorFlow Decision Forests
* **Location:** [`Assignment-1-House-Price-Prediction/`](Assignment-1-House-Price-Prediction/)
* **Objective:** Predict residential property sale prices from 79 explanatory features using tree-based ensemble models.
* **Key Methodology:**
  * Structural inspection, missing value imputation, and log-transformations on skewed target variables (`SalePrice`).
  * End-to-end regression modeling using `tensorflow_decision_forests`.
  * Tracking convergence of Out-of-Bag (OOB) Root Mean Squared Error (RMSE) across tree growth iterations.
  * Variable importance ranking via mean decrease in impurity and structural node analysis.

### 2. Exploratory Data Analysis & Inferential Statistical Testing
* **Location:** [`Assignment-2-Airline-Delays-and-Statistical-Testing/`](Assignment-2-Airline-Delays-and-Statistical-Testing/)
* **Objective:** Conduct rigorous exploratory analysis on large-scale transportation logs and validate business hypotheses on transaction records using statistical inference.
* **Key Methodology:**
  * **Airline Delays EDA:** Investigating departure vs. arrival delays, seasonality trends, carrier performance disparities, and airport network bottlenecks.
  * **E-Commerce Hypothesis Testing:**
    * Formulating null ($H_0$) and alternative ($H_1$) hypotheses.
    * Evaluating parametric assumptions (normality via Shapiro-Wilk/Q-Q plots, homoscedasticity via Levene's test).
    * One-Way ANOVA across customer tiers and product categories.
    * Contingency table analysis and Chi-Square test of independence ($\alpha = 0.05$).
    * Pairwise post-hoc comparisons using Tukey's HSD test with effect size metrics.

### 3. Socio-Economic Country Clustering (Aid Prioritization)
* **Location:** [`Assignment-3-Country-Socioeconomic-Clustering/`](Assignment-3-Country-Socioeconomic-Clustering/)
* **Objective:** Categorize 167 nations across socioeconomic and health factors (e.g., child mortality, GDP per capita, life expectancy, inflation) to objectively identify countries requiring humanitarian aid.
* **Key Methodology:**
  * Feature scaling and Hopkins statistic evaluation for cluster tendency validation ($H \approx 0.88$).
  * Principal Component Analysis (PCA) capturing >95% cumulative explained variance.
  * K-Means clustering optimization using the Elbow Method (Inertia/WCSS) and Silhouette score analysis ($k=3$).
  * Hierarchical agglomerative clustering using Euclidean distance and linkage criteria (Ward, Complete, Average), visualized via dendrograms.
  * Profiling clusters into Under-Developed, Developing, and Developed cohorts.

### 4. Telecom Customer Churn Machine Learning Pipeline
* **Location:** [`Assignment-4-Customer-Churn-Prediction-Pipeline/`](Assignment-4-Customer-Churn-Prediction-Pipeline/)
* **Objective:** Build an end-to-end, leak-free classification pipeline to predict customer churn in telecommunications.
* **Key Methodology:**
  * Data hygiene: handling whitespace anomalies in billing features and robust numerical imputation.
  * Preprocessing: `ColumnTransformer` integration with `StandardScaler` for numeric columns and `OneHotEncoder` for multi-category attributes.
  * Addressing extreme class imbalance using SMOTE and algorithmic cost-sensitive weighting.
  * Comparative model benchmarking:
    * Regularized Logistic Regression (L1/L2)
    * Random Forest Classifier
    * Gradient Boosted Decision Trees
  * Evaluation via Stratified K-Fold cross-validation, ROC-AUC curves, Precision-Recall AUC, and business profit-driven decision threshold optimization.
  * Model interpretability via permutation feature importance to identify primary churn drivers.

---

## Installation & Environment Setup

Clone the repository and install the verified dependencies in a virtual environment:

```bash
git clone https://github.com/MRL0R3/Data-Science-Assignments.git
cd Data-Science-Assignments

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required libraries
pip install -r requirements.txt
```

To run any notebook interactively:

```bash
jupyter notebook
```

---

## Author & Academic Context

* **Author:** Ahmadreza Jafari
* **Institution:** Shahid Beheshti University (SBU), Tehran, Iran
* **Department:** Department of Computer Science
* **GitHub:** [@MRL0R3](https://github.com/MRL0R3)
* **LinkedIn:** [Ahmadreza Jafari](https://www.linkedin.com/in/ahdjy/)