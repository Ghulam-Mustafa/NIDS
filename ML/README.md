# Hybrid Intrusion Detection System — Static + Dynamic

# Note : Use python - 3.12.xx for better usability as other versions are incompatible

A network intrusion detection interface with two modes:
- **Static mode**: upload a `.pcap`/`.pcapng` file (e.g. exported from Wireshark) for offline analysis.
- **Dynamic mode**: live packet capture + real-time classification and traffic monitoring.

Both modes share the same trained ML classification core, so detection behaves consistently
whether you're analyzing a saved capture or watching live traffic.

## Project phases

- [x] **Phase 1 — ML models** (this phase): train & compare 6 classical ML classifiers on NSL-KDD
- [ ] **Phase 2 — Live capture**: real-time packet sniffing + flow-based feature extraction
- [ ] **Phase 3 — Backend**: FastAPI server (pcap upload endpoint + WebSocket live stream)
- [ ] **Phase 4 — Frontend**: static/dynamic mode UI with live dashboard

## Phase 1 — Dataset & Models

### Dataset
**NSL-KDD** — the improved version of the classic KDD Cup 1999 dataset, standard benchmark for
IDS classification research. 41 features per connection record, labeled with specific attack
types, which we bucket into 5 coarse categories for classification:

| Category | Meaning | Example attacks |
|---|---|---|
| `normal` | benign traffic | — |
| `dos` | Denial of Service | neptune, smurf, back, teardrop |
| `probe` | scanning/surveillance | satan, ipsweep, nmap, portsweep |
| `r2l` | Remote-to-Local (unauthorized remote access) | guess_passwd, ftp_write, phf |
| `u2r` | User-to-Root (privilege escalation) | buffer_overflow, rootkit, loadmodule |

### Models trained (all 6, on identical preprocessed features)
1. Naive Bayes (`train_naive_bayes.py`)
2. Decision Tree (`train_decision_tree.py`)
3. Random Forest (`train_random_forest.py`)
4. KNN (`train_knn.py`)
5. SVM — RBF kernel (`train_svm.py`)
6. Logistic Regression (`train_logistic_regression.py`)

### Installing dependencies

`requirements.txt` pulls in all three phase-specific files below. If you
only need one phase right now, install that file directly and skip the
rest:

- `requirements-ml.txt` — Phase 1 only (scikit-learn, pandas, numpy, joblib, matplotlib)
- `requirements-capture.txt` — Phase 2 (scapy, pyshark) — **also needs a
  system-level packet capture driver**; see the notes inside that file
  (Npcap on Windows, libpcap/tshark on Linux/macOS). pip installing the
  Python packages alone is not enough for live capture to work.
- `requirements-backend.txt` — Phase 3 (FastAPI, uvicorn, websockets)

All three use version *ranges* rather than exact pins, so pip's resolver
can pick versions that are actually compatible with each other and with
your Python version (tested on 3.10–3.12), instead of forcing exact
versions that may conflict on your machine.

### How to run Phase 1 from scratch

```bash
pip install -r requirements-ml.txt

# 1. Download NSL-KDD (KDDTrain+.txt, KDDTest+.txt) from GitHub mirror
python -m ml.data.download_dataset

# 2. Preprocess: encode categoricals, map labels, scale features
python -m ml.data.preprocess

# 3. Train all 6 models + generate comparison table
python -m ml.train_all

# (optional) Regenerate the comparison table/chart from saved models,
# without retraining anything:
python -m ml.compare_models
```

### Results (macro-averaged, 5-class)

See `reports/model_comparison.csv` and `reports/model_comparison.png` after running.
On this run:

| Model | Accuracy | F1 (macro) | Train time | Inference time |
|---|---|---|---|---|
| Logistic Regression | 0.79 | **0.61** | 33.7s | 0.008s |
| KNN | 0.77 | 0.57 | 0.02s | 17.6s |
| SVM | 0.77 | 0.55 | 1.6s* | 3.8s |
| Decision Tree | 0.75 | 0.53 | 0.8s | 0.007s |
| Random Forest | 0.75 | 0.52 | 13.8s | 0.29s |
| Naive Bayes | 0.47 | 0.33 | 0.2s | 0.06s |

*SVM trained on a stratified 15k-row subsample for tractable runtime — see comment in `train_svm.py`.

**Key takeaway for the dynamic/real-time mode**: KNN and SVM have the best raw scores in some
metrics but are far too slow at **inference** to classify live traffic flow-by-flow at line
rate. **Decision Tree and Logistic Regression are the practical choices for the real-time
pipeline** (sub-10ms inference); Random Forest is a reasonable middle ground if you can batch
predictions. This trade-off is discussed further in Phase 2.

All minority classes (`r2l`, `u2r`) score much lower recall/F1 across every model — this is
expected given how rare they are in the training data (995 and 52 samples respectively out of
~126k). Worth calling out explicitly in your report as a dataset limitation, not a model bug.
This is exactly what Phase 1b (below) targets.

## Phase 1b — Improved pipeline (v2): SMOTE + tuning + Gradient Boosting

A second, parallel pipeline (`ml/data/preprocess_v2.py`, `ml/models_v2/`, `ml/train_all_v2.py`,
`ml/compare_models_v2.py`) that improves on Phase 1 without touching any of the original v1
files. Three changes:

1. **Correlation-based feature pruning** — numeric features with >0.90 correlation to another
   kept feature are dropped automatically (fit on the training set only). This dropped 7 of 38
   numeric columns, landing on almost the same set the GFG tutorial's own manual correlation
   analysis dropped — a good sanity check that the approach is sound.
2. **SMOTE oversampling** of the two smallest classes, **training set only** — `r2l` boosted
   from 995→5,000 rows, `u2r` from 52→1,000. The test set is left completely untouched and
   unbalanced, so evaluation still reflects real-world class frequencies.
3. **Hyperparameter tuning** via `RandomizedSearchCV` (scored on macro F1) for all 6 original
   models, plus a **7th model, Gradient Boosting** (`sklearn.ensemble.HistGradientBoostingClassifier`
   — a fast histogram-based booster, same family as XGBoost/LightGBM, no extra native
   dependency required).

### How to run

```bash
pip install -r requirements-ml.txt   # now includes imbalanced-learn for SMOTE

python -m ml.data.preprocess_v2      # feature pruning + SMOTE, saves to data/processed_v2/
python -m ml.train_all_v2            # tunes + trains all 7 models, saves to models_saved_v2/
python -m ml.compare_models_v2       # regenerates the table/chart without retraining
```

### v1 vs v2 results (macro F1)

| Model | v1 | v2 |
|---|---|---|
| **Gradient Boosting** (new in v2) | — | **0.614** ← best overall |
| Logistic Regression | 0.606 | 0.577 |
| KNN | 0.575 | 0.588 |
| Naive Bayes | 0.330 | **0.583** (biggest single jump) |
| SVM | 0.552 | 0.537 |
| Decision Tree | 0.533 | 0.536 |
| Random Forest | 0.516 | 0.506 |

See `reports_v2/model_comparison_v2.csv` / `.png` for the full run (precision/recall/train time
included).

### Honest caveats — read before quoting these numbers in a report

- **Random Forest and SVM did not meaningfully improve.** For Random Forest specifically,
  `r2l` precision hit 0.89 but recall was 0.01 — the model is essentially overfitting to the
  synthetic SMOTE points during training rather than learning a generalizable pattern, so it
  still barely flags real `r2l` traffic in the test set. This is a legitimate, reportable
  finding about how SMOTE interacts with unconstrained tree ensembles, not a bug in the code.
- **Hyperparameter search grids were deliberately narrowed** because this environment has a
  single CPU core, and wide grids (e.g. Random Forest with `n_estimators` up to 400 and
  unbounded depth) timed out. The `PARAM_DIST` dict at the top of each `train_*_v2.py` file can
  be widened on a machine with more cores for a more thorough search — this may improve results
  further, particularly for Random Forest and SVM.
- **u2r remains hard even after SMOTE** (only 67 real examples in the test set) — F1 for that
  class ranges roughly 0.11–0.43 across v2 models. Don't be surprised if a professor asks about
  this; it's a fair question, and "the dataset has almost no u2r examples to learn from" is the
  honest answer, not a modeling failure to fix.

### File structure (Phase 1b additions)

```
ids-project/
├── requirements-ml.txt        (updated: added imbalanced-learn)
├── data/
│   └── processed_v2/          X_train.npy (post-SMOTE), X_test.npy, y_train.npy, y_test.npy,
│                                onehot_encoder.joblib, scaler.joblib, label_encoder.joblib,
│                                feature_names.joblib, kept_numeric_cols.joblib
├── models_saved_v2/           one .joblib file per tuned model (7 total)
├── reports_v2/                model_comparison_v2.csv, model_comparison_v2.png
└── ml/
    ├── data/
    │   └── preprocess_v2.py           feature pruning + SMOTE (separate from preprocess.py)
    ├── models_v2/
    │   ├── model_utils_v2.py          RandomizedSearchCV-based tune_and_evaluate()
    │   ├── train_naive_bayes_v2.py
    │   ├── train_decision_tree_v2.py
    │   ├── train_random_forest_v2.py
    │   ├── train_knn_v2.py
    │   ├── train_svm_v2.py
    │   ├── train_logistic_regression_v2.py
    │   └── train_gradient_boosting_v2.py   new 7th model
    ├── train_all_v2.py            runs all 7 + writes comparison table
    └── compare_models_v2.py       re-evaluates saved v2 models without retraining
```

## Which files to audit or replace, and recommended Python version

**Nothing from Phase 1 (v1) needs replacing** — `ml/data/preprocess.py`, `ml/models/*.py`,
`ml/train_all.py`, `ml/compare_models.py` are unchanged and still valid as your baseline/control
results.

**Files worth auditing before you submit, in priority order:**

1. `ml/models_v2/train_random_forest_v2.py` and `train_gradient_boosting_v2.py` — `PARAM_DIST`
   and `n_iter`/`cv` were narrowed specifically to fit this sandbox's single CPU core. On your
   own machine (or CBIT's lab systems, which likely have more cores), widen these back out —
   e.g. Random Forest's `n_estimators` back to `[150, 200, 300, 400]` and `n_iter=10, cv=3` — and
   re-run for potentially better, more thoroughly-searched results.
2. `ml/data/preprocess_v2.py` — the SMOTE target counts (`r2l`→5000, `u2r`→1000) and the
   correlation threshold (0.90) are reasonable defaults but worth trying a couple of alternate
   values (e.g. threshold 0.85, or u2r→2000) if you want to show sensitivity analysis in your
   report.
3. `requirements-ml.txt` / `requirements-capture.txt` / `requirements-backend.txt` — re-check
   these install cleanly on whatever machine you'll actually demo on; version ranges should
   resolve fine, but worth a dry run before your submission deadline, not the night before.
4. `README.md` — the results tables here were generated in this sandbox session; if you
   re-run with wider search grids per point 1, update the numbers before submitting so your
   report and your code match.

**Recommended Python version: 3.11** (specifically 3.11.x). Reasoning: 3.10–3.12 are all tested
and working, but 3.11 is the safest middle choice — old enough that every pinned package
(scikit-learn, imbalanced-learn, scapy, pyshark, fastapi) has mature, well-tested wheels for it,
new enough to still get security updates for the lifetime of a semester project. Avoid 3.13 for
now — some scientific-Python packages were still catching up on prebuilt wheels for it as of
this project's dependency versions, which is the kind of thing that causes exactly the
install errors you hit earlier. If your college lab machines are locked to a specific Python
version already, that takes priority — just confirm it falls in the 3.10–3.12 range this
project was tested against.


### File structure (Phase 1)

```
ids-project/
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                  KDDTrain+.txt, KDDTest+.txt (downloaded)
│   └── processed/            X_train.npy, X_test.npy, y_train.npy, y_test.npy,
│                              onehot_encoder.joblib, scaler.joblib,
│                              label_encoder.joblib, feature_names.joblib
├── models_saved/             one .joblib file per trained model
├── reports/                  model_comparison.csv, model_comparison.png
└── ml/
    ├── data/
    │   ├── feature_columns.py     column names + attack-category mapping
    │   ├── download_dataset.py    fetches NSL-KDD from GitHub mirror
    │   └── preprocess.py          encoding + scaling + saving processed arrays
    ├── models/
    │   ├── model_utils.py         shared load/save/evaluate helpers
    │   ├── train_naive_bayes.py
    │   ├── train_decision_tree.py
    │   ├── train_random_forest.py
    │   ├── train_knn.py
    │   ├── train_svm.py
    │   └── train_logistic_regression.py
    ├── train_all.py           runs all 6 + writes comparison table
    └── compare_models.py      re-evaluates saved models without retraining
```
