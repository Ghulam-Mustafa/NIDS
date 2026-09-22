"""
Loads the seven already-tuned v2 models from models_saved_v2/ and
regenerates the comparison table + chart WITHOUT retraining anything.

Run directly:
    python -m ml.compare_models_v2
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ml.models_v2.model_utils_v2 import load_processed_data_v2, load_model_v2, CATEGORY_NAMES

REPORTS_V2_DIR = os.path.join(os.path.dirname(__file__), "..", "reports_v2")

MODEL_NAMES = [
    "naive_bayes_v2", "decision_tree_v2", "random_forest_v2",
    "knn_v2", "svm_v2", "logistic_regression_v2", "gradient_boosting_v2",
]


def main():
    _, X_test, _, y_test = load_processed_data_v2()

    rows = []
    for name in MODEL_NAMES:
        model = load_model_v2(name)
        y_pred = model.predict(X_test)
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        })

    df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)

    os.makedirs(REPORTS_V2_DIR, exist_ok=True)
    df.to_csv(os.path.join(REPORTS_V2_DIR, "model_comparison_v2.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df["model"], df["f1_macro"], color="#3b6ea5")
    ax.set_ylabel("Macro F1 score")
    ax.set_title("IDS Model Comparison v2 (NSL-KDD, SMOTE + tuned, 5-class)")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    chart_path = os.path.join(REPORTS_V2_DIR, "model_comparison_v2.png")
    plt.savefig(chart_path, dpi=150)

    print(df.to_string(index=False))
    print(f"\nSaved table -> {os.path.join(REPORTS_V2_DIR, 'model_comparison_v2.csv')}")
    print(f"Saved chart -> {chart_path}")


if __name__ == "__main__":
    main()
