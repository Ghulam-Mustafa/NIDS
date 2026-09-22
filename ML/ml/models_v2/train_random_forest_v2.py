"""
v2: Random Forest with tree count/depth/leaf-size tuned via
RandomizedSearchCV, trained on the SMOTE-balanced + correlation-pruned
v2 feature set.

Run directly:
    python -m ml.models_v2.train_random_forest_v2
"""

from sklearn.ensemble import RandomForestClassifier
from ml.models_v2.model_utils_v2 import load_processed_data_v2, tune_and_evaluate

MODEL_NAME = "random_forest_v2"

PARAM_DIST = {
    "n_estimators": [100, 150, 200],
    "max_depth": [15, 20, 25],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}


def main():
    X_train, X_test, y_train, y_test = load_processed_data_v2()
    base = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1)
    return tune_and_evaluate(base, PARAM_DIST, MODEL_NAME, X_train, y_train, X_test, y_test, n_iter=4, cv=2)


if __name__ == "__main__":
    main()
