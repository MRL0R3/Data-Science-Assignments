# Assignment 3: Country Socio-Economic Clustering Using K-Means and Hierarchical Clustering

## Overview
This project classifies 167 countries based on socioeconomic and health factors (including GDP per capita, child mortality, life expectancy, inflation, and income) to identify nations in dire need of aid. The analysis leverages Principal Component Analysis (PCA) for dimensionality reduction alongside K-Means and Agglomerative Hierarchical Clustering.

## Files
- `country_clustering.ipynb` - Jupyter notebook containing end-to-end data preprocessing, PCA dimensionality reduction, Hopkins clusterability evaluation, K-Means, and Hierarchical clustering.
- `Country-data.csv` - The socioeconomic dataset consisting of 167 countries and 9 indicators.
- `Clustering_Analysis_Report.pdf` - Comprehensive technical report documenting cluster profiling, validation metrics, and socio-economic findings.

## Techniques & Methodologies Used
- **Data Preprocessing & Scaling:** Conversion of percentage features to absolute monetary quantities, outlier profiling, and `StandardScaler` normalization.
- **Clusterability Assessment:** Hopkins Statistic calculation and statistical distribution diagnostics.
- **Dimensionality Reduction:** Principal Component Analysis (PCA) with scree plot variance analysis (preserving >95% cumulative variance).
- **K-Means Clustering:** Optimal cluster determination via the Elbow Method (Inertia/WCSS) and Silhouette Coefficient analysis ($k=3$).
- **Hierarchical Clustering:** Agglomerative clustering evaluated across Complete, Ward, and Average linkage methods; Dendrogram visual verification.
- **Cluster Profiling & Interpretation:** Cross-cluster comparison of under-developed, developing, and developed nations to prioritize humanitarian aid allocation.

## Author
Ahmadreza Jafari
