"""Stage 1 of the workflow: raw CSVs -> cleaned, split, feature-selected parquet files."""
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ensure_dir, path
from .data import load_raw
from .preprocessing import FlowPreprocessor, build_training_frame


def run(cfg: dict):
    d, p = cfg["data"], cfg["preprocessing"]
    processed = ensure_dir(path(cfg, d["processed_dir"]))
    models_dir = ensure_dir(path(cfg, cfg["paths"]["models_dir"]))
    reports = ensure_dir(path(cfg, cfg["paths"]["reports_dir"]))

    print("[prepare] loading raw CSVs")
    df = load_raw(str(path(cfg, d["raw_glob"])), p["drop_columns"])
    n_raw = len(df)

    print("[prepare] cleaning")
    X, y = build_training_frame(df, p["drop_columns"], d["benign_cap"], d["random_state"])
    del df

    counts = y.value_counts()
    rare = counts[counts < 2].index.tolist()
    if rare:
        print(f"  warning: dropping classes with <2 samples (cannot stratify): {rare}")
        keep = ~y.isin(rare)
        X, y = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)

    print("[prepare] stratified train/test split")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=d["test_size"], stratify=y, random_state=d["random_state"])

    print("[prepare] fitting feature selection on the train split only")
    pre = FlowPreprocessor(p["drop_columns"], p["corr_threshold"], p["corr_sample"],
                           d["random_state"]).fit(X_tr)

    for name, Xs, ys in (("train", X_tr, y_tr), ("test", X_te, y_te)):
        out = pre.transform_clean(Xs).reset_index(drop=True)
        out["label"] = ys.to_numpy()
        out.to_parquet(processed / f"{name}.parquet", index=False)
        print(f"  wrote {name}.parquet: {out.shape}")

    joblib.dump(pre, models_dir / "preprocessor.joblib")
    dist = pd.DataFrame({"total": y.value_counts(), "train": y_tr.value_counts(),
                         "test": y_te.value_counts()}).fillna(0).astype(int)
    dist.to_csv(reports / "class_distribution.csv")
    (reports / "preprocessing_summary.json").write_text(json.dumps({
        "raw_rows": n_raw, "clean_rows": int(len(X)),
        "n_features": len(pre.columns_), "features": pre.columns_,
        "dropped_constant": pre.dropped_["constant"],
        "dropped_correlated": pre.dropped_["correlated"],
    }, indent=2))
    print("[prepare] done\n", dist)
