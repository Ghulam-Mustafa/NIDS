"""
Support Vector Machine (RBF kernel) classifier for NSL-KDD attack categories.

Note: SVM training cost scales poorly (roughly quadratic+) with the
number of training rows, and the full NSL-KDD training set has ~126k
rows. To keep training time practical, we fit on a stratified subsample
of the training set (SVM_TRAIN_SAMPLE_SIZE) while still evaluating on
the full, untouched test set. This is a documented trade-off, not a
bug — it's part of why SVM is a poor fit for the real-time pipeline
later (slow train AND slow inference relative to tree-based models).

Run directly:
    python -m ml.models.train_svm
"""

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from ml.models.model_utils import load_processed_data, train_and_evaluate

MODEL_NAME = "svm"
SVM_TRAIN_SAMPLE_SIZE = 15000  # stratified subsample for tractable training time


def build_model():
    return SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")


def main():
    X_train, X_test, y_train, y_test = load_processed_data()

    if len(X_train) > SVM_TRAIN_SAMPLE_SIZE:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train,
            train_size=SVM_TRAIN_SAMPLE_SIZE,
            stratify=y_train,
            random_state=42,
        )
        print(f"[svm] Subsampled training set to {SVM_TRAIN_SAMPLE_SIZE} rows "
              f"(stratified) for tractable training time.")

    model = build_model()
    return train_and_evaluate(model, MODEL_NAME, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
