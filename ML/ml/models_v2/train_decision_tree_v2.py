"""
v2: Decision Tree with depth/leaf-size/criterion tuned via
RandomizedSearchCV, trained on the SMOTE-balanced + correlation-pruned
v2 feature set.

Run directly:
    python -m ml.models_v2.train_decision_tree_v2
"""

from sklearn.tree import DecisionTreeClassifier
from ml.models_v2.model_utils_v2 import load_processed_data_v2, tune_and_evaluate

MODEL_NAME = "decision_tree_v2"

PARAM_DIST = {
    "max_depth": [10, 15, 20, 25, 30, None],
    "min_samples_leaf": [1, 2, 5, 10],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"],
}


def main():
    X_train, X_test, y_train, y_test = load_processed_data_v2()
    base = DecisionTreeClassifier(class_weight="balanced", random_state=42)
    return tune_and_evaluate(base, PARAM_DIST, MODEL_NAME, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
