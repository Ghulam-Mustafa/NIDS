"""Stage 3 of the workflow: end-to-end evaluation on the held-out test split."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, roc_auc_score)

from .config import ensure_dir, path
from .pipeline import NIDSPipeline


def _div(a, b):
    return float(a / b) if b else 0.0


def _plot_confusion(cm, labels, out):
    norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    size = max(5, 0.9 * len(labels) + 2)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalised)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_importance(model, names, out, top=20):
    imp = pd.Series(model.feature_importances_, index=names).sort_values(ascending=False)
    imp.to_csv(str(out).replace(".png", ".csv"), header=["importance"])
    fig, ax = plt.subplots(figsize=(7, 6))
    imp.head(top)[::-1].plot.barh(ax=ax)
    ax.set_title(f"Top {top} features (attack-type classifier)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run(cfg: dict):
    processed = path(cfg, cfg["data"]["processed_dir"])
    reports = ensure_dir(path(cfg, cfg["paths"]["reports_dir"]))
    pipe = NIDSPipeline.load(path(cfg, cfg["paths"]["models_dir"]),
                             threshold=cfg["inference"]["attack_threshold"],
                             benign_label=cfg["inference"]["benign_label"])
    benign, anom = pipe.benign_label, pipe.anomaly_label

    test = pd.read_parquet(processed / "test.parquet")
    y_true = test.pop("label")
    out = pipe.predict_clean(test[pipe.preprocessor.columns_])
    y_pred = out["prediction"]
    print(f"[evaluate] {len(test):,} held-out flows")

    # Classification quality is judged on the TRUE classes; ANOMALY is not a true class.
    true_labels = sorted(set(y_true), key=lambda c: (c != benign, c))
    all_labels = true_labels + ([anom] if (y_pred == anom).any() else [])
    report = classification_report(y_true, y_pred, labels=true_labels, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(reports / "classification_report.csv")
    print(classification_report(y_true, y_pred, labels=true_labels, digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    pd.DataFrame(cm, index=all_labels, columns=all_labels).to_csv(reports / "confusion_matrix.csv")
    _plot_confusion(cm, all_labels, reports / "confusion_matrix.png")

    # ---- binary detector quality (stage 1 only) ----
    is_atk = (y_true != benign).to_numpy()
    prob = out["attack_probability"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(is_atk.astype(int), (prob >= pipe.threshold).astype(int),
                                      labels=[0, 1]).ravel()
    precision, recall = _div(tp, tp + fp), _div(tp, tp + fn)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=true_labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=true_labels, average="weighted", zero_division=0),
        "detector": {
            "threshold": pipe.threshold,
            "precision": precision, "recall": recall,
            "f1": _div(2 * precision * recall, precision + recall),
            "false_positive_rate": _div(fp, fp + tn),
            "roc_auc": float(roc_auc_score(is_atk, prob)) if 0 < is_atk.sum() < len(is_atk) else None,
        },
    }

    # ---- what the anomaly stage adds: attacks caught / benign flows wrongly flagged ----
    known = ((y_pred != benign) & (y_pred != anom)).to_numpy()
    any_alert = (y_pred != benign).to_numpy()
    metrics["alerting"] = {
        "attack_recall_known_stages_only": float(known[is_atk].mean()) if is_atk.any() else None,
        "attack_recall_with_anomaly": float(any_alert[is_atk].mean()) if is_atk.any() else None,
        "benign_alert_rate_known_stages_only": float(known[~is_atk].mean()) if (~is_atk).any() else None,
        "benign_alert_rate_with_anomaly": float(any_alert[~is_atk].mean()) if (~is_atk).any() else None,
    }
    if pipe.anomaly is not None:
        flagged = (y_pred == anom)
        metrics["anomaly_stage"] = {
            "threshold": pipe.anomaly_threshold,
            "flagged_fraction_by_true_class": {
                c: float(flagged[y_true == c].mean()) for c in true_labels},
        }
        try:
            metrics["anomaly_stage"]["roc_auc_attack_vs_benign"] = float(
                roc_auc_score(is_atk, -out["anomaly_score"].to_numpy()))
        except ValueError:
            pass

    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    if hasattr(pipe.classifier, "feature_importances_"):
        _plot_importance(pipe.classifier, pipe.preprocessor.columns_, reports / "feature_importance.png")
    print(f"[evaluate] reports written to {reports}")
    return metrics
