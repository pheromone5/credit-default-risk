"""
Validation strategy - written and frozen BEFORE any model is fitted.

Rules of this project:
  1. One holdout slice (20%, stratified) is cut once and touched exactly once,
     at the very end, to produce the numbers that go on the CV.
  2. The remaining 80% is split into N_FOLDS stratified folds. Every model -
     logistic, LightGBM, GRU - trains on the SAME fold assignment, so their
     out-of-fold predictions are directly comparable and can be blended without
     leakage.
  3. The assignment is written to disk. Nothing regenerates it on the fly, and
     nothing reshuffles it after seeing a score.

This file is the "test strategy and systematic test procedures" deliverable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import ARTIFACTS, DATA_RAW, ID, N_FOLDS, RAW_FILES, SEED, TARGET

SPLIT_PATH = ARTIFACTS / "splits.csv"
HOLDOUT_FRAC = 0.20


def build_splits(force: bool = False) -> pd.DataFrame:
    if SPLIT_PATH.exists() and not force:
        return pd.read_csv(SPLIT_PATH)

    app = pd.read_csv(DATA_RAW / RAW_FILES["application"], usecols=[ID, TARGET])
    rng = np.random.default_rng(SEED)

    # stratified holdout
    holdout = np.zeros(len(app), dtype=bool)
    for cls in (0, 1):
        idx = np.flatnonzero(app[TARGET].to_numpy() == cls)
        chosen = rng.choice(idx, size=int(round(HOLDOUT_FRAC * len(idx))), replace=False)
        holdout[chosen] = True

    fold = np.full(len(app), -1, dtype=int)
    dev_idx = np.flatnonzero(~holdout)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, val) in enumerate(skf.split(dev_idx, app[TARGET].to_numpy()[dev_idx])):
        fold[dev_idx[val]] = k

    splits = pd.DataFrame({
        ID: app[ID],
        TARGET: app[TARGET],
        "split": np.where(holdout, "holdout", "dev"),
        "fold": fold,
    })
    splits.to_csv(SPLIT_PATH, index=False)
    return splits


def load_splits() -> pd.DataFrame:
    if not SPLIT_PATH.exists():
        return build_splits()
    return pd.read_csv(SPLIT_PATH)


if __name__ == "__main__":
    s = build_splits(force=True)
    print(f"wrote {SPLIT_PATH}")
    print(s.groupby(["split", "fold"])[TARGET].agg(["size", "mean"]).to_string())
