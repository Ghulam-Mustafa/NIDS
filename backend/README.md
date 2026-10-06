# NIDS backend (Node + Express)

Receives a pcap or flow CSV, converts pcaps to flows with CICFlowMeter, runs your ML model, and serves the results.

## Folder layout (put this next to your ML folder)

```
project/
├── ML/         <- your ML project (nids/, config.yaml, models/ ...)
├── backend/    <- this folder
└── frontend/
```

## Setup

1. In `ML/`, make sure the models are trained (`python -m nids run-all`) so `ML/models/metadata.json` exists.
   The ML folder needs the latest `nids/` package (it must include `nids/columns.py` and the `--summary`
   option of `python -m nids predict`).
2. `cp .env.example .env` (Windows: copy it) and edit:
   - `NIDS_DIR` - path to the ML folder (default `../ML`)
   - `PYTHON_BIN` - the Python that has the ML requirements installed
   - `CONVERTER` / `CFM_PATH` - see below
3. `npm install`, then `npm start` (needs Node 18+). It prints warnings if the ML folder or models are missing.
   Check `http://localhost:5000/api/health`.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/analyze` | multipart field `file` (.pcap/.pcapng/.cap/.csv) -> `{ id }` |
| GET | `/api/jobs/:id` | status (`queued`/`converting`/`classifying`/`done`/`error`) + summary |
| GET | `/api/jobs/:id/flows?cls=alerts\|all\|<class>&page=1&size=15` | paged predictions |
| GET | `/api/jobs/:id/download` | full `predictions.csv` |
| GET | `/api/health` | converter, ML folder, models ready |

For each upload the backend runs, inside `NIDS_DIR`:
`python -m nids predict --input <flows.csv> --output predictions.csv --summary summary.json`

## Converting pcaps with CICFlowMeter

- **Official Java tool** (`CONVERTER=cfm`): install libpcap (Linux) or WinPcap/Npcap (Windows) and a JDK, get
  CICFlowMeter-4.0 from github.com/ahlashkari/CICFlowMeter, and point `CFM_PATH` at `bin/cfm.bat` (or `bin/cfm`).
  The backend runs `cfm "<input folder>" "<output folder>"`.
- **Python port** (`CONVERTER=python`): `pip install cicflowmeter`; the backend runs `cicflowmeter -f in.pcap -c flows.csv`.
  It is a re-implementation, so values can differ from the original tool.
- Column names differ between tool versions; the ML package maps known spellings, and the UI warns when model
  features are missing or Flow Duration looks like seconds (CICIDS2017 uses microseconds).
- Sanity test: convert an original CICIDS2017 pcap (e.g. Monday) and confirm it classifies as mostly BENIGN.

Notes: jobs are kept in memory and files in `backend/jobs/`. There is no authentication or cleanup, so keep it on localhost.
