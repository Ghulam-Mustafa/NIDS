"""
Decision Tree classifier for NSL-KDD attack categories.
Interpretable, handles non-linear feature splits well (mirrors how a
human would write firewall-style rules). max_depth is capped to reduce
overfitting on the noisy minority attack classes.

Run directly:
    python -m ml.models.train_decision_tree
"""

from sklearn.tree import DecisionTreeClassifier
from ml.models.model_utils import load_processed_data, train_and_evaluate

MODEL_NAME = "decision_tree"


def build_model():
    return DecisionTreeClassifier(
        max_depth=20,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )


def main():
    X_train, X_test, y_train, y_test = load_processed_data()
    model = build_model()
    return train_and_evaluate(model, MODEL_NAME, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
