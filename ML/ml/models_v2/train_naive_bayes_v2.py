"""
v2: Naive Bayes with var_smoothing tuned via RandomizedSearchCV, trained
on the SMOTE-balanced + correlation-pruned v2 feature set.

Run directly:
    python -m ml.models_v2.train_naive_bayes_v2
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from ml.models_v2.model_utils_v2 import load_processed_data_v2, tune_and_evaluate

MODEL_NAME = "naive_bayes_v2"

PARAM_DIST = {
    "var_smoothing": np.logspace(-11, -6, 20),
}


def main():
    X_train, X_test, y_train, y_test = load_processed_data_v2()
    return tune_and_evaluate(GaussianNB(), PARAM_DIST, MODEL_NAME, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
