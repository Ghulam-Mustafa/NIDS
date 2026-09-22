"""
v2: SVM (RBF kernel) with C/gamma tuned via RandomizedSearchCV.

Same rationale as v1's SVM script: training cost scales poorly with row
count, and here we ALSO have a CV search loop multiplying that cost
further, so we tune and fit on a stratified subsample of the (already
SMOTE-balanced) training set.

Run directly:
    python -m ml.models_v2.train_svm_v2
"""

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from ml.models_v2.model_utils_v2 import load_processed_data_v2, tune_and_evaluate

MODEL_NAME = "svm_v2"
SVM_TRAIN_SAMPLE_SIZE = 12000

PARAM_DIST = {
    "C": [0.1, 1, 5, 10, 50],
    "gamma": ["scale", "auto", 0.01, 0.1],
}


def main():
    X_train, X_test, y_train, y_test = load_processed_data_v2()

    if len(X_train) > SVM_TRAIN_SAMPLE_SIZE:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=SVM_TRAIN_SAMPLE_SIZE,
            stratify=y_train, random_state=42,
        )
        print(f"[svm_v2] Subsampled training set to {SVM_TRAIN_SAMPLE_SIZE} rows (stratified).")

    base = SVC(kernel="rbf", class_weight="balanced")
    return tune_and_evaluate(base, PARAM_DIST, MODEL_NAME, X_train, y_train, X_test, y_test, n_iter=8, cv=3)


if __name__ == "__main__":
    main()
