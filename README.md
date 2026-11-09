# Customer Churn Prediction - Classical ML Capstone

## Overview

This project builds an end-to-end machine learning workflow for predicting customer churn using a synthetic telecom-style dataset.

The goal is to identify customers who are likely to leave so that a retention team can target them with proactive support or retention offers.

The dataset is generated directly inside the notebook, so no external files, internet connection, or APIs are required.

## Project Features

The notebook covers the complete classical machine learning workflow:

1. Problem framing and exploratory data analysis
2. Data leakage detection and removal
3. Train, validation, and test splitting
4. Numeric scaling and categorical one-hot encoding
5. Leakage-free preprocessing with `ColumnTransformer` and `Pipeline`
6. Comparison of four classification models
7. Hyperparameter tuning with `GridSearchCV`
8. Model evaluation using ROC-AUC, precision, recall, F1-score, and a confusion matrix
9. Decision-threshold comparison
10. Model interpretation using permutation importance and logistic regression coefficients
11. Business recommendation and reflection

## Models Compared

- Logistic Regression
- Random Forest
- Gradient Boosting
- RBF Support Vector Machine

The final model is a tuned `GradientBoostingClassifier`.

## Project Structure

```text
.
├── day16_capstone.ipynb
└── README.md
```

## Requirements

The notebook uses:

- Python
- NumPy
- pandas
- Matplotlib
- scikit-learn
- JupyterLab or Jupyter Notebook

## Running the Project

Open `day16_capstone.ipynb` in JupyterLab or Jupyter Notebook and run all cells from top to bottom.

In JupyterLab:

```text
Kernel -> Restart Kernel and Run All Cells
```

No dataset download is needed because the customer data is generated in memory with a fixed random seed.

## Machine Learning Approach

The target variable is:

```text
churn = 1 -> customer churns
churn = 0 -> customer stays
```

Because missing a real churner can be costly, churn recall is treated as an important business metric. ROC-AUC is used to compare and tune models independently of a particular classification threshold.

A deliberately leaking feature, `exit_survey_sent`, is identified and removed before training.

Numeric features are standardized with `StandardScaler`, while categorical features are encoded using `OneHotEncoder`. Both transformations are kept inside a scikit-learn pipeline so preprocessing is fitted only on training data.

## Evaluation

The final evaluation includes:

- ROC-AUC
- Precision
- Recall
- F1-score
- Confusion matrix
- Threshold comparison

The notebook also evaluates a lower decision threshold of `0.30` to increase churn recall when catching more potential churners is more important than avoiding false alarms.

## Interpretation

The project uses permutation importance to identify influential original features and logistic regression coefficients to understand which features increase or decrease predicted churn risk.

## Business Recommendation

The final notebook translates the model results into a short recommendation for a retention team. The general strategy is to identify high-risk customers early and use targeted retention actions while choosing the classification threshold based on the relative cost of a missed churner versus an unnecessary retention offer.

## Recommendation

We can predict churn reliably using account and plan data available today, with the tuned GradientBoosting model achieving about 0.95 ROC-AUC on the held-out test set. Using a 0.30 alert threshold catches roughly 84% of customers who will churn, while accepting extra false alarms, which is reasonable when losing a customer costs much more than making a retention offer. The retention team should prioritize higher-risk customers, especially those showing the strongest drivers identified by the model, and test proactive support or contract incentives for these groups. Before rollout, the business should attach real costs to missed churners and unnecessary offers so the operating threshold can be chosen from expected financial impact rather than model metrics alone.

## Reflection

### 1. Leakage

A feature such as `cancellation_date` could leak the outcome because it is normally known only after the customer has decided to leave. I would check when each feature becomes available, inspect suspiciously strong relationships with the target, and investigate any unrealistically high validation score before trusting the model.

### 2. Metric Choice

Precision matters more when acting on a churn alert is expensive. For example, if every flagged customer receives a large account credit, too many false positives waste substantial money, so the business may prefer fewer alerts with a higher fraction of true churners.

### 3. Trust

I would deploy the tuned GradientBoosting model because it performs strongly and its behavior can be investigated with permutation importance, while keeping logistic regression as a simple baseline. Since all four models score very closely, the next useful improvement is more likely to come from better behavioral features or fresher data than from a more complicated algorithm.

### 4. Honesty

If I used the test set to choose the threshold, the test set would become part of model development. Reporting that same test result afterward would be optimistic because I had already selected a decision rule that worked well on those examples. Threshold selection should use validation data, with test data reserved for the final unbiased estimate.
