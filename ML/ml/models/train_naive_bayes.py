"""
Gaussian Naive Bayes classifier for NSL-KDD attack categories.
Fastest model here, useful as a lightweight baseline — but its feature
independence assumption is unrealistic for correlated network features
(e.g. src_bytes/dst_bytes/duration), so expect it to be the weakest
performer in the comparison table.

Run directly:
    python -m ml.models.train_naive_bayes
"""

from sklearn.naive_bayes import GaussianNB
from ml.models.model_utils import load_processed_data, train_and_evaluate

MODEL_NAME = "naive_bayes"


def build_model():
    return GaussianNB()


def main():
    X_train, X_test, y_train, y_test = load_processed_data()
    model = build_model()
    return train_and_evaluate(model, MODEL_NAME, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
