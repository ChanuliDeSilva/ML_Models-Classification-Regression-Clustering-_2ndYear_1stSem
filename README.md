# ShopSphere: Customer Churn, Lifetime Value and Segmentation

Machine learning project for **ShopSphere**, an international online retailer growing quickly but
struggling with customer retention, unpredictable spending and unclear customer segments. The project
applies **classification, regression and clustering** to 50,000 customers and turns each task's best
model into a concrete business output. Developed following the **CRISP-DM** methodology.

*Module: Artificial Intelligence (COMP50059), University of Staffordshire / APIIT.*

## Business problems and tasks

| Task | Question | Output |
|---|---|---|
| **Classification** | Which customers are likely to churn? | A ranked list of high-risk, high-value customers |
| **Regression** | What will each customer's lifetime value be? | Bronze / Silver / Gold / Platinum value tiers |
| **Clustering** | What kinds of customers do we have? | Named customer segments, including one flagged for review |

## Dataset

[E-Commerce Customer Behavior dataset (Kaggle)](https://www.kaggle.com/datasets/dhairyajeetsingh/ecommerce-customer-behavior-dataset):
50,000 customers with demographic, engagement, purchase-behaviour, customer-service and financial
features, plus a `Churned` label. The CSV is not included in this repo. Download it and update the path
at the top of each notebook.

## Repository structure

```
├── Classification.ipynb                       # Task 2: churn prediction
├── Regression.ipynb                           # Task 3: lifetime value forecasting
├── Clustering.ipynb                           # Task 4: customer segmentation
├── ecommerce_customer_churn_dataset.csv       # Dataset
└── README.md
```

Each notebook is self-contained and follows the same flow: Business Understanding, Data
Understanding, Data Preparation, Modelling, Evaluation and Business Insight.

## Data preparation (shared)

- Removed duplicate rows and corrected logically invalid values (ages outside 18–100, negative purchase
  counts, rates above 100%) by clipping to valid bounds.
- Engineered features: engagement, purchase-activity, risk and friction scores.
- Median/mode imputation, fitted on the training set only to avoid data leakage.
- Outliers investigated with the IQR method and kept rather than removed. `RobustScaler` limits their
  influence on scaling.
- One-hot encoding of categorical features (`Gender`, plus `Country` for clustering).
- Investigated a decimal-entry error in `Average_Order_Value` (25 rows around 100× too large),
  found through a sharp gap in the sorted values.

## Models and results

### Classification: predicting `Churned` (29% churn rate)
Decision Tree, Random Forest, SVM, KNN, Logistic Regression, Artificial Neural Network.
`class_weight='balanced'` was used for most models. KNN used decision-threshold tuning instead.

| Model | F1 | ROC-AUC |
|---|---|---|
| ANN | 0.798 | 0.916 |
| Random Forest | 0.798 | 0.919 |
| SVM | 0.761 | 0.900 |
| Decision Tree | 0.771 | 0.894 |
| KNN | 0.660 | 0.826 |
| Logistic Regression | 0.596 | 0.786 |

The ANN and Random Forest are effectively tied. **Business insight:** combining the ANN's churn
probabilities with customer value identified **1,213 high-risk, high-value customers** (combined
Lifetime_Value ≈ $2.82M, 88.8% of whom churned in the test data).

### Regression: predicting `Lifetime_Value`
Linear, Polynomial, Ridge, Random Forest, Gradient Boosting, ANN.

| Model | RMSE | R² |
|---|---|---|
| **Gradient Boosting** | $220.56 | 0.9422 |
| Random Forest | $241.58 | 0.9306 |
| ANN | $415.69 | 0.7946 |
| Linear / Ridge / Polynomial (degree 1) | $650.88 | 0.4965 |

**Business insight:** customers were ranked into four value tiers by *predicted* value. The Platinum tier
(2,500 customers) holds about 46% of total forecasted value (≈ $6.68M).

### Clustering: customer segmentation (no target)
KMeans, Hierarchical (Ward), DBSCAN, Gaussian Mixture Model, Autoencoder + KMeans.

| Model | Silhouette | Davies-Bouldin |
|---|---|---|
| Autoencoder + KMeans | 0.2309 | 1.2376 |
| KMeans | 0.2182 | 1.2566 |
| Hierarchical | 0.2029 | 1.3140 |
| GMM | 0.1032 | 3.5389 |
| DBSCAN | n/a (found one cluster) | n/a |

**Business insight:** three named segments: Standard / Lower-Engagement (31,319 customers),
Loyal / High-Value (18,581) and a small High-Volume, High-Return group (100 customers, ~89% return rate)
flagged for fraud/returns review. KMeans, Hierarchical Clustering and the Autoencoder approach all
surfaced this group.

## Key findings

- Non-linear models clearly beat linear ones in both supervised tasks, so customer behaviour here depends on interacting factors.
- Silhouette scores of about 0.2 are typical for real behavioural data: segments overlap rather than forming sharp clusters. DBSCAN's single-cluster result agrees.
- Different clustering algorithms can disagree. GMM split customers by engagement level and missed the small anomalous group.
- Neural-network results vary slightly between runs even with a fixed seed.

## Known limitations

- The 25-row `Average_Order_Value` correction is applied in the Classification and Clustering notebooks.
  If it is not applied there, the linear-family results (R² ≈ 0.50) are affected by those rows, and the notebook notes this.
- Some tuning comparisons also looked at test-set scores. In each case the choice rests on
  cross-validation, as documented in the notebooks.

## Tech stack

Python · pandas · NumPy · scikit-learn · TensorFlow/Keras · SciPy · matplotlib · seaborn

## AI assistance

An AI assistant (Claude) was used to help debug code, explain concepts and interpret outputs.
All modelling decisions and written analysis were reviewed and are my own.

## Author

Chanuli De Silva · chanulisujanyadesilva@gmail.com
