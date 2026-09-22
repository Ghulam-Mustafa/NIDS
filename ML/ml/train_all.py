"""
Runs all six classifiers end-to-end (train + evaluate + save), then
writes the combined comparison table to reports/model_comparison.csv.

Run directly:
    python -m ml.train_all
"""

import os
import pandas as pd

from ml.models import (
    train_naive_bayes,
    train_decision_tree,
    train_random_forest,
    train_knn,
    train_svm,
    train_logistic_regression,
)

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")


def main():
    results = []
    results.append(train_naive_bayes.main())
    results.append(train_decision_tree.main())
    results.append(train_random_forest.main())
    results.append(train_knn.main())
    results.append(train_svm.main())
    results.append(train_logistic_regression.main())

    df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    df.to_csv(out_path, index=False)

    print("\n\n=== FINAL COMPARISON (sorted by macro F1) ===")
    print(df.to_string(index=False))
    print(f"\nSaved comparison table -> {out_path}")


if __name__ == "__main__":
    main()
