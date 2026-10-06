"""Raw CSV loading (memory-friendly)."""
import glob
import os
import pandas as pd
from .labels import LABEL_COL


def load_raw(pattern: str, drop_cols) -> pd.DataFrame:
    """Load every CSV matching `pattern` into one float32 frame plus a Label column."""
    drop = {c.lower() for c in drop_cols}
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files match '{pattern}'. Put the CICIDS CSVs in data/raw/.")
    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="latin1", low_memory=False)
        df.columns = df.columns.str.strip()                       # 2017 headers have leading spaces
        label = next((c for c in df.columns if c.lower() == "label"), None)
        if label is None:
            raise ValueError(f"{f}: no Label column found")
        df = df.rename(columns={label: LABEL_COL})
        df = df[df[LABEL_COL].astype(str).str.lower() != "label"]  # 2018: repeated header rows
        df = df.drop(columns=[c for c in df.columns if c.lower() in drop])
        feats = [c for c in df.columns if c != LABEL_COL]
        df[feats] = df[feats].apply(pd.to_numeric, errors="coerce").astype("float32")
        frames.append(df)
        print(f"  loaded {os.path.basename(f)}: {df.shape[0]:,} rows")
    return pd.concat(frames, ignore_index=True)
