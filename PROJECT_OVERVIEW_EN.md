# ChurnScope — customer churn analysis

ChurnScope is a small, reproducible analysis of the public IBM Telco Customer Churn sample. The question is practical: which customers appear more likely to leave, and what are the limits of a model used to prioritize retention work? The code and executed exploration notebook are in French; this page is a short English guide.

## Data checks

The file has 7,043 customers and 21 columns, including the customer ID and the `Churn` target. I checked every column's type, missing values, blank strings and categories. Eleven `TotalCharges` entries are blank; all belong to customers with zero months of tenure, so they are set to zero. There are no duplicate IDs. Twenty-two rows share the same profile if the ID is ignored; I kept them because they may represent different customers. The [full audit](reports/DATA_AUDIT.md) also checks non-negative charges and consistency between internet/phone subscriptions and related services.

## Modelling and evaluation

- The customer ID is excluded from the features. A stratified 80/20 split reserves 1,409 customers for the final test.
- Logistic regression, Random Forest and Gradient Boosting are compared with five-fold stratified cross-validation **on training data only**. Encoding and scaling stay inside each fold's pipeline.
- Gradient Boosting has the highest mean validation ROC-AUC (0.8481), narrowly ahead of logistic regression (0.8460). This small gap does not prove it will always be better.
- On the held-out test, Gradient Boosting reaches ROC-AUC **0.8433**, average precision **0.6600**, precision **0.6655** and recall **0.5160** at a threshold of 0.5. A majority-class baseline has 73.46% accuracy but detects no churners.

The confusion matrix contains 193 detected departures, 181 missed departures, 97 false alerts and 938 correctly identified stayers. The [error table](reports/error_analysis.csv) shows where the misses occur. The [threshold table](reports/threshold_analysis.csv) illustrates the precision–recall trade-off; it is descriptive, **not** a threshold selected on the test set.

## Explore and reproduce

The [executed notebook](notebooks/01_exploration.ipynb) covers data cleaning and exploratory charts. [Figures](reports/figures/) and [model metrics](reports/metrics.json) can be viewed without running the code. To reproduce the results with Python 3.11:

1. Download the [IBM sample CSV](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv) into `data/raw/Telco-Customer-Churn.csv`.
2. Install dependencies: `python -m pip install -r requirements.txt`.
3. Run `python src/train.py`, then `python app.py` for the local Dash dashboard.
4. Run tests with `python -m unittest discover -s tests -v`.

The raw data, test-level predictions and fitted model are deliberately not included in the archive. Training recreates the predictions and model locally. The dashboard shows held-out test results; it does not score new customers.

## Limits

This is a public, static sample dataset rather than a live retention system. There is one held-out test split, no time-based validation, no systematic hyperparameter search and no business-calibrated decision threshold. Observed differences between customer groups are associations, not evidence of causality.
