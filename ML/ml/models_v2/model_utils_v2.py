"""
Shared utilities for the v2 (tuned) training scripts. Adds hyperparameter
search on top of the plain fit/evaluate flow from model_utils.py.

Kept as a separate module (not editing model_utils.py) so the v1 training
scripts are completely unaffected by this work.
"""

import os
import time
import joblib
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
PROCESSED_V2_DIR = os.path.join(BASE_DIR, "data", "processed_v2")
MODELS_V2_DIR = os.path.join(BASE_DIR, "models_saved_v2")

CATEGORY_NAMES = ["normal", "dos", "probe", "r2l", "u2r"]


def load_processed_data_v2():
    X_train = np.load(os.path.join(PROCESSED_V2_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_V2_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(PROCESSED_V2_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(PROCESSED_V2_DIR, "y_test.npy"))
    return X_train, X_test, y_train, y_test


def save_model_v2(model, name: str) -> str:
    os.makedirs(MODELS_V2_DIR, exist_ok=True)
    path = os.path.join(MODELS_V2_DIR, f"{name}.joblib")
    joblib.dump(model, path)
    return path


def load_model_v2(name: str):
    return joblib.load(os.path.join(MODELS_V2_DIR, f"{name}.joblib"))


def tune_and_evaluate(
    base_model, param_distributions, model_name: str,
    X_train, y_train, X_test, y_test,
    n_iter: int = 12, cv: int = 3,
) -> dict:
    """RandomizedSearchCV over param_distributions (scored on macro F1,
    since that's the metric v1 was weak on), then evaluate the best
    estimator on the held-out test set and save it."""
    print(f"\n=== Tuning {model_name} (RandomizedSearchCV, {n_iter} candidates x {cv}-fold) ===")

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="f1_macro",
        random_state=42,
        n_jobs=-1,
    )

    t0 = time.time()
    search.fit(X_train, y_train)
    train_time = time.time() - t0

    print(f"Best params: {search.best_params_}")
    print(f"Best CV macro-F1: {search.best_score_:.4f}")

    best_model = search.best_estimator_

    t0 = time.time()
    y_pred = best_model.predict(X_test)
    inference_time = time.time() - t0

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "train_time_sec": round(train_time, 2),
        "inference_time_sec": round(inference_time, 4),
        "best_params": str(search.best_params_),
    }

    print(f"Test accuracy: {metrics['accuracy']:.4f} | Test F1 (macro): {metrics['f1_macro']:.4f}")
    print(classification_report(y_test, y_pred, target_names=CATEGORY_NAMES, zero_division=0))

    save_path = save_model_v2(best_model, model_name)
    print(f"Saved tuned model -> {save_path}")

    return metrics
