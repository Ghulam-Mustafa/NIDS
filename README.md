# CICIDS Network Intrusion Detection Pipeline

A three-stage ML pipeline for flow-based network intrusion detection on **CICIDS2017**
(~2.8M flows, the `MachineLearningCVE` CSVs). CSE-CIC-IDS2018 CSVs load with the same code.

```
raw CSVs ──► prepare ──► train ──► evaluate ──► predict (new flow CSVs)
             clean        stage 1: detector      metrics, confusion
             dedupe       (benign vs attack)     matrix, importances,
             split        stage 2: classifier    anomaly-stage impact
             select       (which known attack)
             features     stage 3: anomaly detector
                          (unseen attacks)

flow ──► detector ──┬─ attack ──► classifier ──► DoS / DDoS / PortScan / ...
                    │                 └─ low confidence ──► ANOMALY
                    └─ benign ──► anomaly detector ──┬─ normal ──► BENIGN
                                                     └─ unusual ──► ANOMALY
```

**Why three stages?** ~80% of flows are benign. Stage 1 answers "is this an attack?" using
all the data; stage 2 is trained only on attack flows, so rare attacks (Bot, WebAttack,
Infiltration, Heartbleed) are not drowned out by benign traffic. Final classes:
`BENIGN, DoS, DDoS, PortScan, BruteForce, WebAttack, Bot, Infiltration, Heartbleed`, plus
`ANOMALY` for flows that do not fit known traffic (possible unseen attacks).

Stage 3 is an Isolation Forest trained on **benign flows only**. Its threshold is calibrated on
held-out benign flows so that only `target_fpr` (default 1%) of normal traffic is flagged.
It scores flows the detector passed as benign. In addition, flows the detector flagged whose best
attack-class probability is below `min_class_confidence` (default 0.5) are labelled `ANOMALY`
instead of being forced into a known class. Set `anomaly.enabled: false` to run without it.

## Folder layout

```
cicids-nids/
├── config.yaml            # every setting: paths, thresholds, model types + hyperparameters
├── requirements.txt
├── Makefile               # make prepare | train | evaluate | all | test
├── data/
│   ├── raw/               # <- put the CICIDS CSVs here
│   └── processed/         # train.parquet, test.parquet (generated)
├── models/                # preprocessor, detector, classifier, anomaly, metadata.json (generated)
├── reports/               # metrics.json, classification report, confusion matrix,
│                          # feature importance, class distribution (generated)
├── backend/               # Node/Express API (upload, convert, classify)
├── frontend/              # React (Vite) dashboard
├── nids/                  # the Python package
│   ├── labels.py          # label harmonisation (2017 + 2018 names -> 9 classes)
│   ├── data.py            # memory-friendly CSV loading
│   ├── preprocessing.py   # cleaning + FlowPreprocessor (feature selection)
│   ├── models.py          # model factory
│   ├── prepare.py  train.py  evaluate.py  predict.py   # one module per workflow stage
│   ├── pipeline.py        # NIDSPipeline: the deployable two-stage predictor
│   └── cli.py
└── tests/test_pipeline.py # end-to-end smoke test on synthetic data
```

## Quick start

```bash
pip install -r requirements.txt
# copy the 8 CICIDS2017 CSVs into data/raw/
python -m nids run-all            # prepare -> train -> evaluate
python -m nids predict --input new_flows.csv --output predictions.csv
```

Or run stages individually: `python -m nids prepare`, `train`, `evaluate`.
Use `--config other.yaml` (before the command) to run with different settings.
Needs roughly 8-16 GB RAM for the full dataset; set `data.benign_cap` in `config.yaml`
(e.g. 400000) for quick experiments.

## Preprocessing (prepare stage)

- Column names stripped; repeated header rows (2018) and duplicated columns removed.
- Identifier columns (Flow ID, IPs, Src Port, Timestamp) dropped so the model learns behaviour, not the capture setup.
- `inf`/NaN rows (mainly `Flow Bytes/s`, `Flow Packets/s`) removed; exact duplicates removed **before** splitting to prevent train/test leakage.
- Negative values clipped to 0, except `Init_Win_bytes_*` where -1 is meaningful.
- Stratified 80/20 split; constant and highly correlated (>0.98) features are chosen **on the train split only** and saved with median fill values, so inference applies identical preprocessing.

## Outputs (reports/)

`metrics.json` (accuracy, macro/weighted F1 over the true classes, detector precision/recall/FPR/ROC-AUC,
and an `alerting` block showing attack recall and benign alert rate with and without the anomaly stage),
`classification_report.csv`, `confusion_matrix.png/.csv`, `feature_importance.png/.csv`,
`class_distribution.csv`, `preprocessing_summary.json`.
Judge the model by **macro F1** and per-class recall, not accuracy: rare classes are where it is hardest.

## Deployment notes and limitations

- The model consumes **CICFlowMeter flow records**, not raw packets. To classify live traffic or a pcap,
  convert it to flows with CICFlowMeter first, then run `predict`.
- `attack_threshold` (stage 1) and `anomaly.target_fpr` / `min_class_confidence` (stage 3) in `config.yaml`
  trade detection rate against false alarms. `ANOMALY` alerts need analyst review: they mean "unusual",
  not "malicious".
- The anomaly stage is only a safety net. Tree models extrapolate: traffic far outside anything seen in
  training can still be assigned a known class with high confidence (seen in synthetic tests). CICIDS2017 has
  no unseen-attack test built in, so measure it by holding one attack class out of training.
- CICIDS2017 is a lab-generated dataset with known label quirks; expect strong in-dataset scores
  and weaker performance on other networks. Validate on other data before real deployment.

---

# Web app (React + Node)

```
browser (React, :5173) ──► Node/Express (:5000) ──► CICFlowMeter (pcap -> flows.csv)
                                                └─► python -m nids predict  (3-stage model)
```

- `frontend/` - React (Vite). Drag-and-drop upload, progress steps, class breakdown, filterable flow table, CSV download.
- `backend/` - Express. `POST /api/analyze` stores the upload and starts a job; `GET /api/jobs/:id` is polled for status;
  `GET /api/jobs/:id/flows` pages through results; `GET /api/jobs/:id/download` returns `predictions.csv`.

```bash
# 0. train the model first (python -m nids run-all) so models/ is populated
# 1. backend
cd backend && cp .env.example .env     # edit CONVERTER / CFM_PATH / PYTHON_BIN, then set the vars in your shell
npm install && npm start
# 2. frontend (second terminal)
cd frontend && npm install && npm run dev      # open http://localhost:5173
```

Environment variables are read from the process environment (`.env` is a template); on Windows PowerShell:
`$env:CONVERTER="cfm"; $env:CFM_PATH="C:\tools\CICFlowMeter-4.0\bin\cfm.bat"; npm start`.
Jobs are kept in memory and files in `backend/jobs/`; this is a single-user demo, not a hardened service
(no auth, no cleanup). Add both before exposing it beyond localhost.

# Converting pcap files with CICFlowMeter

The model reads **flows**, so a pcap must be converted first. The web app does this automatically; to do it by hand:

**Option A - official CICFlowMeter (Java), the tool CICIDS2017 was built with**
1. Install libpcap (Linux: `sudo apt-get install libpcap-dev`) or WinPcap/Npcap (Windows), and a JDK.
2. Download/build CICFlowMeter-4.0 from github.com/ahlashkari/CICFlowMeter.
3. Run: `cfm.bat "C:\pcaps" "C:\flows"` (Linux: `./cfm /pcaps /flows`). Each `x.pcap` produces a flow CSV in the output folder.
4. Classify: `python -m nids predict --input C:\flows\x.pcap_Flow.csv --output predictions.csv`.
   Set `CONVERTER=cfm` and `CFM_PATH` in the backend. The tool also has a Docker packaging if the Java/jnetpcap setup fights you.

**Option B - Python port, easiest to install**
`pip install cicflowmeter` then `cicflowmeter -f capture.pcap -c flows.csv`. Set `CONVERTER=python`.
This is a re-implementation; its feature values can differ from the original tool. The
prediction preprocessor detects when `Flow Duration` is in seconds instead of the
microseconds used by CICIDS2017 and converts all duration, IAT, active, and idle timing
features to microseconds before inference. It reports the conversion in the prediction
summary. Other feature-definition differences can still cause distribution shift.

**Check before trusting results**
- Column names differ between tool versions. `nids/columns.py` maps the CICIDS2017, 2018 and snake_case spellings to one
  schema; any model feature still missing is filled with a training median and shown as a warning in the UI.
- Units must match: CICIDS2017 durations/IATs are microseconds. `predict` automatically converts
  seconds-based timing features to microseconds when detected and reports this in its summary.
- Sanity test: convert one of the original CICIDS2017 pcaps (e.g. Monday) with your converter and confirm the output
  mostly classifies as BENIGN, and a Friday DDoS/PortScan pcap shows those attacks.
- Flows from a different tool version or network than CICIDS2017 will shift the feature distributions, so expect
  more false alarms on your own captures than on the dataset's test split.
