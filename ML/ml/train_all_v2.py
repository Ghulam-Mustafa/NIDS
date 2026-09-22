"""
Runs all 7 v2 classifiers end-to-end (tune + evaluate + save), then
writes the combined comparison table to reports_v2/model_comparison_v2.csv.

Run directly:
    python -m ml.train_all_v2
"""

import os
import pandas as pd

from ml.models_v2 import (
    train_naive_bayes_v2,
    train_decision_tree_v2,
    train_random_forest_v2,
    train_knn_v2,
    train_svm_v2,
    train_logistic_regression_v2,
    train_gradient_boosting_v2,
)

REPORTS_V2_DIR = os.path.join(os.path.dirname(__file__), "..", "reports_v2")


def main():
    results = []
    results.append(train_naive_bayes_v2.main())
    results.append(train_decision_tree_v2.main())
    results.append(train_random_forest_v2.main())
    results.append(train_knn_v2.main())
    results.append(train_svm_v2.main())
    results.append(train_logistic_regression_v2.main())
    results.append(train_gradient_boosting_v2.main())

    df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)

    os.makedirs(REPORTS_V2_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_V2_DIR, "model_comparison_v2.csv")
    df.to_csv(out_path, index=False)

    print("\n\n=== FINAL v2 COMPARISON (sorted by macro F1) ===")
    print(df.drop(columns=["best_params"]).to_string(index=False))
    print(f"\nSaved comparison table -> {out_path}")


if __name__ == "__main__":
    main()
