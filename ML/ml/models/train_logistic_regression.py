"""
Logistic Regression classifier for NSL-KDD attack categories.
Fast, interpretable baseline. Uses class_weight='balanced' to compensate
for the heavy imbalance toward 'normal'/'dos' vs the tiny 'u2r' class.

Run directly:
    python -m ml.models.train_logistic_regression
"""

from sklearn.linear_model import LogisticRegression
from ml.models.model_utils import load_processed_data, train_and_evaluate

MODEL_NAME = "logistic_regression"


def build_model():
    # Note: newer scikit-learn versions dropped the multi_class= argument;
    # the 'lbfgs' solver already handles multinomial (softmax) classification
    # natively for multi-class targets, so no explicit flag is needed.
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )


def main():
    X_train, X_test, y_train, y_test = load_processed_data()
    model = build_model()
    return train_and_evaluate(model, MODEL_NAME, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
