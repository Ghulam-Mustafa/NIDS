"""
v2: Logistic Regression with regularization strength/penalty tuned via
RandomizedSearchCV, trained on the SMOTE-balanced + correlation-pruned
v2 feature set.

Run directly:
    python -m ml.models_v2.train_logistic_regression_v2
"""

from sklearn.linear_model import LogisticRegression
from ml.models_v2.model_utils_v2 import load_processed_data_v2, tune_and_evaluate

MODEL_NAME = "logistic_regression_v2"

PARAM_DIST = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["lbfgs"],
}


def main():
    X_train, X_test, y_train, y_test = load_processed_data_v2()
    base = LogisticRegression(max_iter=500, class_weight="balanced")
    return tune_and_evaluate(base, PARAM_DIST, MODEL_NAME, X_train, y_train, X_test, y_test, n_iter=3, cv=2)


if __name__ == "__main__":
    main()
