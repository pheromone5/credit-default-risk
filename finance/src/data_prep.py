"""Shared loading / encoding helpers so every model sees identical data and folds."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import ARTIFACTS, DATA_PROC, ID, TARGET
from validation import load_splits

DROP_COLS = {ID, TARGET, "split", "fold"}
RESULTS_PATH = ARTIFACTS / "results.json"


def load_features(app_only: bool = False) -> pd.DataFrame:
    feats = pd.read_parquet(DATA_PROC / "features.parquet")
    if app_only:
        keep = [c for c in feats.columns
                if not c.startswith(("BUR_", "PREV_", "INS_", "CC_", "POS_"))]
        feats = feats[keep]
    splits = load_splits()
    return feats.merge(splits[[ID, "split", "fold"]], on=ID, how="left")


def split_frame(df: pd.DataFrame):
    dev = df[df["split"] == "dev"].reset_index(drop=True)
    hold = df[df["split"] == "holdout"].reset_index(drop=True)
    return dev, hold


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in DROP_COLS]


def as_categorical(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """LightGBM handles pandas `category` dtype natively - no one-hot needed."""
    out = df.copy()
    for c in cols:
        out[c] = out[c].astype("category")
    return out


def object_columns(df: pd.DataFrame) -> list[str]:
    """Non-numeric feature columns (pandas 3 stores strings as `str`, not `object`)."""
    return [c for c in feature_columns(df)
            if not pd.api.types.is_numeric_dtype(df[c])]


def save_predictions(name: str, dev_ids, oof, hold_ids, hold_pred) -> None:
    np.savez(ARTIFACTS / f"pred_{name}.npz",
             dev_ids=np.asarray(dev_ids), oof=np.asarray(oof, dtype=np.float64),
             hold_ids=np.asarray(hold_ids), hold=np.asarray(hold_pred, dtype=np.float64))


def load_predictions(name: str):
    z = np.load(ARTIFACTS / f"pred_{name}.npz")
    return z["dev_ids"], z["oof"], z["hold_ids"], z["hold"]


def record_result(row: dict) -> None:
    """Append/replace a row in artifacts/results.json, keyed by model name."""
    rows = []
    if RESULTS_PATH.exists():
        rows = json.loads(RESULTS_PATH.read_text())
    rows = [r for r in rows if r["model"] != row["model"]]
    rows.append(row)
    RESULTS_PATH.write_text(json.dumps(rows, indent=2))


def all_results() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        return pd.DataFrame()
    return pd.DataFrame(json.loads(RESULTS_PATH.read_text()))
