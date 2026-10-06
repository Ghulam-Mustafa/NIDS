"""Stage 4 (deployment): classify new CICFlowMeter flow CSVs with the trained pipeline."""
import json

import pandas as pd

from .columns import normalize_columns

from .config import path
from .pipeline import NIDSPipeline

ID_COLS = {"flow id", "src ip", "dst ip", "src port", "destination port", "protocol", "timestamp"}


def run(cfg: dict, input_path: str, output_path: str, chunksize: int = 200_000, summary_path=None):
    pipe = NIDSPipeline.load(path(cfg, cfg["paths"]["models_dir"]),
                             threshold=cfg["inference"]["attack_threshold"],
                             benign_label=cfg["inference"]["benign_label"])
    first, counts, rows, missing, warns = True, pd.Series(dtype="int64"), 0, set(), []
    for chunk in pd.read_csv(input_path, encoding="latin1", low_memory=False, chunksize=chunksize):
        chunk = normalize_columns(chunk)
        ids = chunk[[c for c in chunk.columns if c.lower() in ID_COLS]].reset_index(drop=True)
        res = pipe.predict_raw(chunk).reset_index(drop=True)
        pd.concat([ids, res], axis=1).to_csv(
            output_path, mode="w" if first else "a", header=first, index=False)
        counts = counts.add(res["prediction"].value_counts(), fill_value=0)
        rows += len(res)
        missing |= set(pipe.preprocessor.last_missing_)
        warns += [w for w in pipe.preprocessor.last_warnings_ if w not in warns]
        first = False
    if summary_path:
        with open(summary_path, "w") as f:
            json.dump({"rows": rows, "counts": {k: int(v) for k, v in counts.items()},
                       "missing_features": sorted(missing), "warnings": warns}, f)
    print(f"[predict] wrote {output_path}\n{counts.astype(int).sort_values(ascending=False)}")
