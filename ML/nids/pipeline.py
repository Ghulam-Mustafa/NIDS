"""Three-stage inference pipeline.

  1. detector   : attack vs benign                     (all flows)
  2. classifier : which known attack                   (flows the detector flags;
                  low-confidence predictions are labelled ANOMALY instead)
  3. anomaly    : does a "benign" flow look unusual?   (flows the detector passes)
                  -> labelled ANOMALY = possible unseen attack
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


class NIDSPipeline:
    def __init__(self, preprocessor, detector, classifier, threshold=0.5, benign_label="BENIGN",
                 anomaly=None, anomaly_threshold=None, anomaly_label="ANOMALY",
                 min_class_confidence=0.0):
        self.preprocessor = preprocessor
        self.detector = detector
        self.classifier = classifier
        self.threshold = threshold
        self.benign_label = benign_label
        self.anomaly = anomaly
        self.anomaly_threshold = anomaly_threshold
        self.anomaly_label = anomaly_label
        self.min_class_confidence = min_class_confidence

    @classmethod
    def load(cls, models_dir, threshold=None, benign_label=None):
        d = Path(models_dir)
        meta = json.loads((d / "metadata.json").read_text())
        am = meta.get("anomaly", {"enabled": False})
        return cls(
            preprocessor=joblib.load(d / "preprocessor.joblib"),
            detector=joblib.load(d / "detector.joblib"),
            classifier=joblib.load(d / "classifier.joblib"),
            threshold=meta["attack_threshold"] if threshold is None else threshold,
            benign_label=benign_label or meta["benign_label"],
            anomaly=joblib.load(d / "anomaly.joblib") if am.get("enabled") else None,
            anomaly_threshold=am.get("threshold"),
            anomaly_label=am.get("label", "ANOMALY"),
            min_class_confidence=am.get("min_class_confidence", 0.0),
        )

    def predict_clean(self, X: pd.DataFrame) -> pd.DataFrame:
        """X: already-preprocessed features (columns == preprocessor.columns_)."""
        prob = self.detector.predict_proba(X)[:, 1]
        labels = np.full(len(X), self.benign_label, dtype=object)
        flagged = prob >= self.threshold
        if flagged.any():
            proba = self.classifier.predict_proba(X[flagged])
            pred = self.classifier.classes_[proba.argmax(axis=1)].astype(object)
            if self.anomaly is not None and self.min_class_confidence > 0:
                pred[proba.max(axis=1) < self.min_class_confidence] = self.anomaly_label
            labels[flagged] = pred
        out = pd.DataFrame({"prediction": labels, "attack_probability": prob}, index=X.index)

        if self.anomaly is not None:
            score = self.anomaly.score_samples(X)          # lower = more unusual
            odd = (~flagged) & (score < self.anomaly_threshold)
            labels[odd] = self.anomaly_label
            out["prediction"] = labels
            out["anomaly_score"] = score
        return out

    def predict_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """df: raw CICFlowMeter records."""
        return self.predict_clean(self.preprocessor.transform(df))
