"""Smoke test: synthetic CICFlowMeter-style data through prepare -> train -> evaluate -> predict."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nids import evaluate, predict, prepare, train          # noqa: E402
from nids.config import load_config                           # noqa: E402
from nids.preprocessing import FlowPreprocessor              # noqa: E402

FEATS = [" Destination Port", " Flow Duration", " Total Fwd Packets", " Flow Bytes/s",
         " Init_Win_bytes_forward", " Fwd Header Length", " Fwd Header Length.1",
         " Packet Length Mean", " Flow IAT Mean", " Constant Col"]
CLASSES = {"BENIGN": 1500, "DoS Hulk": 400, "DDoS": 400, "PortScan": 300,
           "Web Attack \u2013 Brute Force": 120, "Heartbleed": 12}


def make_csv(path, seed=0):
    rng = np.random.RandomState(seed)
    frames = []
    for i, (name, n) in enumerate(CLASSES.items()):
        base = rng.rand(len(FEATS)) * 50
        data = rng.normal(loc=base + i * 30, scale=5 + i, size=(n, len(FEATS)))
        df = pd.DataFrame(data, columns=FEATS)
        df[" Fwd Header Length.1"] = df[" Fwd Header Length"]      # duplicate column, like 2017
        df[" Constant Col"] = 1.0
        df["Flow ID"] = "1.1.1.1-2.2.2.2"
        df[" Source IP"] = "1.1.1.1"
        df[" Label"] = name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df.loc[3, " Flow Bytes/s"] = np.inf                             # inf, NaN and negatives
    df.loc[4, " Flow Duration"] = np.nan
    df.loc[5, " Flow IAT Mean"] = -7.0
    df.to_csv(path, index=False)


@pytest.fixture()
def cfg(tmp_path):
    (tmp_path / "data" / "raw").mkdir(parents=True)
    make_csv(tmp_path / "data" / "raw" / "day1.csv")
    cfg = load_config()
    cfg["_root"] = str(tmp_path)
    cfg["detector"]["params"].update(max_iter=40)
    cfg["classifier"]["params"].update(n_estimators=30)
    return cfg


def test_full_workflow(cfg, tmp_path):
    prepare.run(cfg)
    assert (tmp_path / "data/processed/train.parquet").exists()
    train.run(cfg)
    metrics = evaluate.run(cfg)
    assert metrics["macro_f1"] > 0.8
    assert metrics["detector"]["recall"] > 0.9
    assert (tmp_path / "reports/confusion_matrix.png").exists()
    assert (tmp_path / "models/anomaly.joblib").exists()
    assert metrics["alerting"]["attack_recall_with_anomaly"] >= \
        metrics["alerting"]["attack_recall_known_stages_only"]
    assert metrics["alerting"]["benign_alert_rate_with_anomaly"] < 0.10
    meta = json.loads((tmp_path / "models/metadata.json").read_text())
    assert "DoS" in meta["attack_classes"]

    out = tmp_path / "pred.csv"
    predict.run(cfg, str(tmp_path / "data/raw/day1.csv"), str(out), chunksize=1000)
    pred = pd.read_csv(out)
    assert len(pred) == sum(CLASSES.values())
    assert {"prediction", "attack_probability", "anomaly_score"} <= set(pred.columns)


def test_inference_converts_second_timing_units():
    pre = FlowPreprocessor(["flow id"])
    pre.columns_ = ["Flow Duration", "Flow IAT Mean", "Total Fwd Packets"]
    pre.medians_ = pd.Series({
        "Flow Duration": 50_000.0,
        "Flow IAT Mean": 10_000.0,
        "Total Fwd Packets": 2.0,
    })

    transformed = pre.transform(pd.DataFrame({
        "flow_duration": [1.0, 2.0],
        "flow_iat_mean": [0.5, 1.5],
        "tot_fwd_pkts": [2.0, 3.0],
    }))

    assert transformed["Flow Duration"].tolist() == [1_000_000.0, 2_000_000.0]
    assert transformed["Flow IAT Mean"].tolist() == [500_000.0, 1_500_000.0]
    assert transformed["Total Fwd Packets"].tolist() == [2.0, 3.0]
    assert any("converted to microseconds" in warning for warning in pre.last_warnings_)
