"""
Preprocessing pipeline for NSL-KDD.

Loads the raw KDDTrain+.txt / KDDTest+.txt files, applies:
    1. Column naming (from feature_columns.py)
    2. Categorical encoding (protocol_type, service, flag) via OneHotEncoder
    3. Coarse label mapping (41 raw attack types -> 5 categories)
    4. Label encoding of the target (normal/dos/probe/r2l/u2r -> 0..4)
    5. Numeric feature scaling via StandardScaler

Saves processed train/test arrays (.npy) and the fitted encoder/scaler
objects (.joblib) under data/processed/, so every training script and
the live-capture feature extractor can reuse the exact same transforms.

Run directly:
    python -m ml.data.preprocess
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from ml.data.feature_columns import (
    RAW_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS,
    ATTACK_CATEGORIES, map_label_to_category,
)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def _load_raw(filename: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, filename)
    df = pd.read_csv(path, header=None, names=RAW_COLUMNS)
    return df


def preprocess_and_save() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train_df = _load_raw("KDDTrain+.txt")
    test_df = _load_raw("KDDTest+.txt")
    print(f"Loaded train: {train_df.shape}, test: {test_df.shape}")

    # Map raw attack labels -> 5 coarse categories
    train_df["category"] = train_df["label"].apply(map_label_to_category)
    test_df["category"] = test_df["label"].apply(map_label_to_category)

    # --- Categorical encoding (fit on train only, transform both) ---
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(train_df[CATEGORICAL_COLUMNS])

    train_cat = ohe.transform(train_df[CATEGORICAL_COLUMNS])
    test_cat = ohe.transform(test_df[CATEGORICAL_COLUMNS])
    cat_feature_names = ohe.get_feature_names_out(CATEGORICAL_COLUMNS)

    # --- Numeric scaling (fit on train only, transform both) ---
    scaler = StandardScaler()
    train_num = scaler.fit_transform(train_df[NUMERIC_COLUMNS])
    test_num = scaler.transform(test_df[NUMERIC_COLUMNS])

    # --- Combine numeric + categorical into final feature matrix ---
    X_train = np.hstack([train_num, train_cat])
    X_test = np.hstack([test_num, test_cat])
    feature_names = list(NUMERIC_COLUMNS) + list(cat_feature_names)

    # --- Label encoding of target category ---
    label_encoder = LabelEncoder()
    label_encoder.fit(ATTACK_CATEGORIES)  # fixed order: normal,dos,probe,r2l,u2r
    y_train = label_encoder.transform(train_df["category"])
    y_test = label_encoder.transform(test_df["category"])

    # --- Save everything ---
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

    joblib.dump(ohe, os.path.join(PROCESSED_DIR, "onehot_encoder.joblib"))
    joblib.dump(scaler, os.path.join(PROCESSED_DIR, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(PROCESSED_DIR, "label_encoder.joblib"))
    joblib.dump(feature_names, os.path.join(PROCESSED_DIR, "feature_names.joblib"))

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print("Class distribution (train):")
    print(train_df["category"].value_counts())
    print("Class distribution (test):")
    print(test_df["category"].value_counts())
    print(f"Saved processed data + encoders to {PROCESSED_DIR}")


if __name__ == "__main__":
    preprocess_and_save()
