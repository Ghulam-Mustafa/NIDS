"""
v2: Gradient Boosting classifier -- the 7th model, added because boosted
trees are typically the strongest classical option for tabular IDS
features (this is also what the GeeksforGeeks tutorial's own comparison
found).

Uses sklearn's HistGradientBoostingClassifier rather than the plain
GradientBoostingClassifier: it's a histogram-based implementation
(same family of algorithm as XGBoost/LightGBM) that trains dramatically
faster on datasets this size, and ships with scikit-learn directly so no
extra native dependency (xgboost/lightgbm) is required.

Run directly:
    python -m ml.models_v2.train_gradient_boosting_v2
"""

from sklearn.ensemble import HistGradientBoostingClassifier
from ml.models_v2.model_utils_v2 import load_processed_data_v2, tune_and_evaluate

MODEL_NAME = "gradient_boosting_v2"

PARAM_DIST = {
    "max_iter": [100, 150, 200],
    "max_depth": [None, 15, 20],
    "learning_rate": [0.1, 0.15, 0.2],
    "l2_regularization": [0.0, 0.5],
}


def main():
    X_train, X_test, y_train, y_test = load_processed_data_v2()
    base = HistGradientBoostingClassifier(random_state=42)
    return tune_and_evaluate(base, PARAM_DIST, MODEL_NAME, X_train, y_train, X_test, y_test, n_iter=3, cv=2)


if __name__ == "__main__":
    main()
