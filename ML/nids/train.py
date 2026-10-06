"""Stage 2 of the workflow: train the detector (stage 1) and attack classifier (stage 2)."""
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ensure_dir, path
from .models import build_model, fit_model


def run(cfg: dict):
    processed = path(cfg, cfg["data"]["processed_dir"])
    models_dir = ensure_dir(path(cfg, cfg["paths"]["models_dir"]))
    benign = cfg["inference"]["benign_label"]

    train = pd.read_parquet(processed / "train.parquet")
    y = train.pop("label")
    X = train
    print(f"[train] {len(X):,} training flows, {X.shape[1]} features")

    # ---- Stage 1: detector (benign vs attack) --------------------------------------
    dcfg = cfg["detector"]
    y_bin = (y != benign).astype(int)
    Xd, yd = X, y_bin
    cap = dcfg.get("max_train_rows", 0)
    if cap and cap < len(X):
        Xd, _, yd, _ = train_test_split(X, y_bin, train_size=cap, stratify=y_bin,
                                        random_state=cfg["data"]["random_state"])
    t0 = time.time()
    detector = fit_model(build_model(dcfg["type"], dcfg["params"]), Xd, yd)
    print(f"[train] detector ({dcfg['type']}) fitted on {len(Xd):,} rows in {time.time()-t0:.0f}s")

    # ---- Stage 2: attack-type classifier (attack flows only) -----------------------
    ccfg = cfg["classifier"]
    is_attack = (y != benign).to_numpy()
    Xa, ya = X[is_attack], y[is_attack]
    t0 = time.time()
    classifier = fit_model(build_model(ccfg["type"], ccfg["params"]), Xa, ya)
    print(f"[train] classifier ({ccfg['type']}) fitted on {len(Xa):,} attack rows in {time.time()-t0:.0f}s")

    # ---- Stage 3: anomaly detector (benign flows only) -----------------------------
    acfg = cfg.get("anomaly", {})
    anomaly_meta = {"enabled": False}
    if acfg.get("enabled", False):
        Xb = X[~is_attack]
        perm = np.random.RandomState(cfg["data"]["random_state"]).permutation(len(Xb))
        n_fit = min(acfg.get("max_train_rows", 500_000), int(0.8 * len(Xb)))
        fit_idx, calib_idx = perm[:n_fit], perm[n_fit:n_fit + 200_000]
        t0 = time.time()
        anomaly = build_model(acfg["type"], acfg["params"]).fit(Xb.iloc[fit_idx])
        # calibrate on benign flows the model has not seen: flag the lowest `target_fpr` share
        scores = anomaly.score_samples(Xb.iloc[calib_idx])
        threshold = float(np.quantile(scores, acfg["target_fpr"]))
        joblib.dump(anomaly, models_dir / "anomaly.joblib")
        anomaly_meta = {"enabled": True, "type": acfg["type"], "threshold": threshold,
                        "target_fpr": acfg["target_fpr"], "label": acfg["label"],
                        "min_class_confidence": acfg.get("min_class_confidence", 0.0)}
        print(f"[train] anomaly detector ({acfg['type']}) fitted on {len(fit_idx):,} benign rows "
              f"in {time.time()-t0:.0f}s, threshold={threshold:.4f}")

    joblib.dump(detector, models_dir / "detector.joblib")
    joblib.dump(classifier, models_dir / "classifier.joblib")
    (models_dir / "metadata.json").write_text(json.dumps({
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "detector": dcfg["type"], "classifier": ccfg["type"],
        "attack_classes": [str(c) for c in classifier.classes_],
        "attack_threshold": cfg["inference"]["attack_threshold"],
        "benign_label": benign,
        "anomaly": anomaly_meta,
        "n_features": X.shape[1],
        "n_train_rows": int(len(X)),
    }, indent=2))
    print(f"[train] saved models to {models_dir}")
