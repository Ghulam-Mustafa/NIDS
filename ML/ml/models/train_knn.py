"""
K-Nearest Neighbors classifier for NSL-KDD attack categories.
Simple, works decently on well-separated classes. Note: inference cost
scales with training-set size (distance computed against all stored
points), which makes KNN a poor fit for the *dynamic* real-time mode
later — it's included here for the comparison table, not for deployment.

Run directly:
    python -m ml.models.train_knn
"""

from sklearn.neighbors import KNeighborsClassifier
from ml.models.model_utils import load_processed_data, train_and_evaluate

MODEL_NAME = "knn"


def build_model():
    return KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        n_jobs=-1,
    )


def main():
    X_train, X_test, y_train, y_test = load_processed_data()
    model = build_model()
    return train_and_evaluate(model, MODEL_NAME, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
