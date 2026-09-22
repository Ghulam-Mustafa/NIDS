"""
Shared utilities used by every train_<model>.py script:
    - loading the preprocessed NSL-KDD arrays
    - saving/loading a trained model consistently
    - computing a standard metrics dict (accuracy, precision, recall, f1)
    - printing a readable classification report

Keeping this logic in one place means each train_<model>.py file only
contains the code specific to *that* model.
"""

import os
import time
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models_saved")

CATEGORY_NAMES = ["normal", "dos", "probe", "r2l", "u2r"]  # fixed order, matches label_encoder


def load_processed_data():
    """Returns X_train, X_test, y_train, y_test as numpy arrays."""
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    return X_train, X_test, y_train, y_test


def save_model(model, name: str) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(model, path)
    return path


def load_model(name: str):
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    return joblib.load(path)


def train_and_evaluate(model, model_name: str, X_train, y_train, X_test, y_test) -> dict:
    """Fits `model`, times training + inference, evaluates on the test
    set, saves the model to disk, and returns a metrics dict suitable
    for the model-comparison table."""
    print(f"\n=== Training {model_name} ===")

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - t0

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "train_time_sec": round(train_time, 3),
        "inference_time_sec": round(inference_time, 4),
    }

    print(f"Accuracy: {metrics['accuracy']:.4f} | F1 (macro): {metrics['f1_macro']:.4f} "
          f"| train: {train_time:.2f}s | inference: {inference_time:.4f}s")
    print(classification_report(y_test, y_pred, target_names=CATEGORY_NAMES, zero_division=0))

    save_path = save_model(model, model_name)
    print(f"Saved model -> {save_path}")

    return metrics


def get_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)
