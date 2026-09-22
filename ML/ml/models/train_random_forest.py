"""
Random Forest classifier for NSL-KDD attack categories.
Usually the best all-rounder for this problem: handles non-linearity,
gives feature importances, and copes reasonably with class imbalance.
This is the model earmarked for the real-time (dynamic) pipeline later,
since it's fast at inference and consistently strong on flow-based
tabular features.

Run directly:
    python -m ml.models.train_random_forest
"""

from sklearn.ensemble import RandomForestClassifier
from ml.models.model_utils import load_processed_data, train_and_evaluate

MODEL_NAME = "random_forest"


def build_model():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def main():
    X_train, X_test, y_train, y_test = load_processed_data()
    model = build_model()
    return train_and_evaluate(model, MODEL_NAME, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
