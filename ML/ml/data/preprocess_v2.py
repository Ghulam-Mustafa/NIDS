"""
v2 preprocessing pipeline for NSL-KDD. Builds on the same raw files as
preprocess.py but adds two improvements aimed directly at the weak spot
in the v1 results (low recall/F1 on the tiny r2l/u2r classes):

    1. Correlation-based feature pruning: drop numeric features that are
       highly correlated (>0.9) with another feature already kept, since
       redundant features add noise without adding signal (mirrors the
       manual correlation-analysis step common in IDS tutorials, but
       computed automatically from the data rather than hardcoded).
    2. SMOTE oversampling of the minority classes (r2l, u2r) on the
       TRAINING set only. The test set is left completely untouched and
       unbalanced, so evaluation still reflects real-world class
       frequencies -- only the model's training signal changes.

Saves to data/processed_v2/ so this never overwrites the v1 outputs in
data/processed/.

Run directly:
    python -m ml.data.preprocess_v2
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

from ml.data.feature_columns import (
    RAW_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS,
    ATTACK_CATEGORIES, map_label_to_category,
)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_V2_DIR = os.path.join(BASE_DIR, "data", "processed_v2")

CORRELATION_THRESHOLD = 0.90


def _load_raw(filename: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, filename)
    return pd.read_csv(path, header=None, names=RAW_COLUMNS)


def _drop_correlated_features(train_num_df: pd.DataFrame, threshold: float) -> list:
    """Returns the list of numeric columns to KEEP, after dropping one
    column from every pair with |correlation| > threshold. Correlation
    is computed on the training data only, to avoid any test-set leakage
    into a preprocessing decision."""
    corr = train_num_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    kept = [c for c in train_num_df.columns if c not in to_drop]
    print(f"[feature pruning] Dropping {len(to_drop)} highly-correlated columns "
          f"(threshold {threshold}): {to_drop}")
    print(f"[feature pruning] Keeping {len(kept)} of {len(train_num_df.columns)} numeric columns")
    return kept


def preprocess_and_save() -> None:
    os.makedirs(PROCESSED_V2_DIR, exist_ok=True)

    train_df = _load_raw("KDDTrain+.txt")
    test_df = _load_raw("KDDTest+.txt")
    print(f"Loaded train: {train_df.shape}, test: {test_df.shape}")

    train_df["category"] = train_df["label"].apply(map_label_to_category)
    test_df["category"] = test_df["label"].apply(map_label_to_category)

    # --- Step 1: correlation-based numeric feature pruning (fit on train) ---
    kept_numeric_cols = _drop_correlated_features(train_df[NUMERIC_COLUMNS], CORRELATION_THRESHOLD)

    # --- Step 2: categorical encoding (fit on train only) ---
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(train_df[CATEGORICAL_COLUMNS])
    train_cat = ohe.transform(train_df[CATEGORICAL_COLUMNS])
    test_cat = ohe.transform(test_df[CATEGORICAL_COLUMNS])
    cat_feature_names = ohe.get_feature_names_out(CATEGORICAL_COLUMNS)

    # --- Step 3: numeric scaling on the pruned columns only (fit on train) ---
    scaler = StandardScaler()
    train_num = scaler.fit_transform(train_df[kept_numeric_cols])
    test_num = scaler.transform(test_df[kept_numeric_cols])

    X_train = np.hstack([train_num, train_cat])
    X_test = np.hstack([test_num, test_cat])
    feature_names = list(kept_numeric_cols) + list(cat_feature_names)

    # --- Step 4: label encoding ---
    label_encoder = LabelEncoder()
    label_encoder.fit(ATTACK_CATEGORIES)
    y_train = label_encoder.transform(train_df["category"])
    y_test = label_encoder.transform(test_df["category"])

    print("\nClass distribution BEFORE SMOTE (train):")
    print(train_df["category"].value_counts())

    # --- Step 5: SMOTE oversampling of minority classes (train only) ---
    # Boost r2l and u2r specifically -- these are the classes v1 struggled
    # with. Targets are capped well below the majority class size so we
    # don't manufacture an unrealistic amount of synthetic data.
    counts = pd.Series(y_train).value_counts().to_dict()
    idx = {name: i for i, name in enumerate(label_encoder.classes_)}

    sampling_strategy = {}
    if counts.get(idx["r2l"], 0) < 5000:
        sampling_strategy[idx["r2l"]] = 5000
    if counts.get(idx["u2r"], 0) < 1000:
        sampling_strategy[idx["u2r"]] = 1000

    smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=5, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("\nClass distribution AFTER SMOTE (train):")
    print(pd.Series(y_train_res).map(lambda i: label_encoder.classes_[i]).value_counts())

    # --- Save everything (separate from v1) ---
    np.save(os.path.join(PROCESSED_V2_DIR, "X_train.npy"), X_train_res)
    np.save(os.path.join(PROCESSED_V2_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_V2_DIR, "y_train.npy"), y_train_res)
    np.save(os.path.join(PROCESSED_V2_DIR, "y_test.npy"), y_test)

    joblib.dump(ohe, os.path.join(PROCESSED_V2_DIR, "onehot_encoder.joblib"))
    joblib.dump(scaler, os.path.join(PROCESSED_V2_DIR, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(PROCESSED_V2_DIR, "label_encoder.joblib"))
    joblib.dump(feature_names, os.path.join(PROCESSED_V2_DIR, "feature_names.joblib"))
    joblib.dump(kept_numeric_cols, os.path.join(PROCESSED_V2_DIR, "kept_numeric_cols.joblib"))

    print(f"\nX_train (post-SMOTE): {X_train_res.shape}, X_test: {X_test.shape}")
    print(f"Saved processed v2 data + encoders to {PROCESSED_V2_DIR}")


if __name__ == "__main__":
    preprocess_and_save()
