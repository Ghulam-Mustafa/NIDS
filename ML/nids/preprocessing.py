"""Cleaning and feature selection for CICFlowMeter flow records."""
import numpy as np
import pandas as pd
from .columns import normalize_columns
from .labels import LABEL_COL, map_label

_TIME_COLUMNS = (
    "Flow Duration",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
)
_MICROSECONDS_PER_SECOND = 1_000_000


def basic_clean(df: pd.DataFrame, drop_cols):
    """Row-preserving cleaning shared by training and inference.

    Returns (X float32 with NaN for bad values, y or None). Never drops rows, so
    predictions stay aligned with the input records.
    """
    drop = {c.lower() for c in drop_cols}
    df = normalize_columns(df)

    y = None
    if LABEL_COL in df.columns:
        y = df[LABEL_COL].map(map_label)
        df = df.drop(columns=LABEL_COL)
    df = df.drop(columns=[c for c in df.columns if c.lower() in drop])

    X = df.apply(pd.to_numeric, errors="coerce").astype("float32")
    X = X.replace([np.inf, -np.inf], np.nan)          # Flow Bytes/s, Flow Packets/s
    for c in X.columns:                               # negatives are CICFlowMeter artefacts...
        if "init" not in c.lower():                   # ...except Init_Win_bytes_* where -1 is meaningful
            X[c] = X[c].clip(lower=0)
    return X, y


def build_training_frame(df: pd.DataFrame, drop_cols, benign_cap=0, seed=42):
    """Training-only cleaning: drop NaN rows, drop exact duplicates (before splitting!)."""
    X, y = basic_clean(df, drop_cols)
    n0 = len(X)
    keep = X.notna().all(axis=1).to_numpy()
    X, y = X[keep], y[keep]

    codes = y.astype("category").cat.codes.to_numpy()
    dup = X.assign(_y=codes).duplicated().to_numpy()
    X, y = X[~dup], y[~dup]
    print(f"  cleaning: {n0:,} -> {len(X):,} rows (NaN/inf rows and exact duplicates removed)")

    if benign_cap and (y == "BENIGN").sum() > benign_cap:
        rng = np.random.RandomState(seed)
        b_idx = np.flatnonzero((y == "BENIGN").to_numpy())
        drop_idx = rng.choice(b_idx, len(b_idx) - benign_cap, replace=False)
        mask = np.ones(len(X), dtype=bool)
        mask[drop_idx] = False
        X, y = X[mask], y[mask]
    return X.reset_index(drop=True), y.reset_index(drop=True)


class FlowPreprocessor:
    """Learns the feature set + fill values on the TRAIN split, then applies them at inference."""

    def __init__(self, drop_cols, corr_threshold=0.98, corr_sample=200_000, seed=42):
        self.drop_cols = [c.lower() for c in drop_cols]
        self.corr_threshold = corr_threshold
        self.corr_sample = corr_sample
        self.seed = seed
        self.columns_ = None
        self.medians_ = None
        self.dropped_ = {}

    def fit(self, X: pd.DataFrame):
        constant = [c for c in X.columns if X[c].nunique() <= 1]
        X = X.drop(columns=constant)
        sample = X.sample(min(self.corr_sample, len(X)), random_state=self.seed)
        corr = sample.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        redundant = [c for c in upper.columns if (upper[c] > self.corr_threshold).any()]
        self.columns_ = [c for c in X.columns if c not in redundant]
        self.medians_ = X[self.columns_].median()
        self.dropped_ = {"constant": constant, "correlated": redundant}
        print(f"  features: kept {len(self.columns_)} "
              f"(dropped {len(constant)} constant, {len(redundant)} correlated)")
        return self

    def transform_clean(self, X: pd.DataFrame) -> pd.DataFrame:
        """For frames already passed through basic_clean (training path)."""
        return X[self.columns_]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """For raw CICFlowMeter records (inference path)."""
        X, _ = basic_clean(df, self.drop_cols)
        missing = [c for c in self.columns_ if c not in X.columns]
        warnings = []
        train_dur = self.medians_.get("Flow Duration")
        if "Flow Duration" in X.columns and train_dur and train_dur > 0:
            m = X["Flow Duration"].median()
            if m == m and m / train_dur < 1e-4:
                converted = [c for c in _TIME_COLUMNS if c in X.columns]
                X.loc[:, converted] *= _MICROSECONDS_PER_SECOND
                warnings.append(
                    "Timing features look like seconds; converted to microseconds to match "
                    f"training data ({len(converted)} columns).")
        if missing:
            warnings.append(f"{len(missing)} model feature(s) missing from the input and filled with training medians.")
        self.last_missing_, self.last_warnings_ = missing, warnings
        if missing:
            print(f"  warning: {len(missing)} expected feature(s) missing, filled with medians: {missing[:5]}...")
        X = X.reindex(columns=self.columns_)
        return X.fillna(self.medians_).astype("float32")
