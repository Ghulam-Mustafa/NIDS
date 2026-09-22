"""
v2: KNN with n_neighbors/weights tuned via RandomizedSearchCV.

KNN's training-time "fit" just stores the data, but both hyperparameter
search (cv folds x candidates) and inference cost scale with the size of
the stored training set. After SMOTE the v2 training set is ~131k rows;
tuning KNN directly on that is impractically slow in a CV loop, so -- as
in v1's SVM script -- we tune and fit on a stratified subsample instead.
This is a documented trade-off, consistent with why KNN was already
flagged in Phase 1 as a poor fit for real-time inference regardless.

Run directly:
    python -m ml.models_v2.train_knn_v2
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from ml.models_v2.model_utils_v2 import load_processed_data_v2, tune_and_evaluate

MODEL_NAME = "knn_v2"
KNN_TRAIN_SAMPLE_SIZE = 20000

PARAM_DIST = {
    "n_neighbors": [3, 5, 7, 9, 11, 15],
    "weights": ["uniform", "distance"],
    "p": [1, 2],  # 1 = Manhattan, 2 = Euclidean
}


def main():
    X_train, X_test, y_train, y_test = load_processed_data_v2()

    if len(X_train) > KNN_TRAIN_SAMPLE_SIZE:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=KNN_TRAIN_SAMPLE_SIZE,
            stratify=y_train, random_state=42,
        )
        print(f"[knn_v2] Subsampled training set to {KNN_TRAIN_SAMPLE_SIZE} rows (stratified).")

    return tune_and_evaluate(KNeighborsClassifier(n_jobs=1), PARAM_DIST, MODEL_NAME,
                              X_train, y_train, X_test, y_test, n_iter=10, cv=3)


if __name__ == "__main__":
    main()
