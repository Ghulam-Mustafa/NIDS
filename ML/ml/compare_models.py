"""
Loads the six already-trained models from models_saved/ and regenerates
the comparison table (and a bar chart) WITHOUT retraining anything.
Useful once train_all.py has been run once — re-run this any time you
just want the report again.

Run directly:
    python -m ml.compare_models
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ml.models.model_utils import load_processed_data, load_model, CATEGORY_NAMES

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

MODEL_NAMES = [
    "naive_bayes", "decision_tree", "random_forest",
    "knn", "svm", "logistic_regression",
]


def main():
    _, X_test, _, y_test = load_processed_data()

    rows = []
    for name in MODEL_NAMES:
        model = load_model(name)
        y_pred = model.predict(X_test)
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        })

    df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)

    # Bar chart of macro F1 across models
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["model"], df["f1_macro"], color="#3b6ea5")
    ax.set_ylabel("Macro F1 score")
    ax.set_title("IDS Model Comparison (NSL-KDD, 5-class)")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    chart_path = os.path.join(REPORTS_DIR, "model_comparison.png")
    plt.savefig(chart_path, dpi=150)

    print(df.to_string(index=False))
    print(f"\nSaved table -> {os.path.join(REPORTS_DIR, 'model_comparison.csv')}")
    print(f"Saved chart -> {chart_path}")


if __name__ == "__main__":
    main()
