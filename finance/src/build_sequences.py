"""
Turn the monthly tables into a dense (n_customers, SEQ_LEN, n_channels) tensor.

This is the step that makes a deep model worth having. The aggregate features in
features_pandas.py answer "how late does this customer usually pay?". The tensor
answers "is this customer getting better or worse, and how fast?" - a question a
mean cannot represent, because two customers with identical average delinquency
can be moving in opposite directions.

Channels (see config.SEQ_FEATURES):
    dpd         max days past due observed that month
    util        credit card balance / limit
    pay_ratio   amount paid / amount due that month
    late_days   mean (payment date - due date) that month
    n_payments  number of instalment events that month
    observed    1 if any record exists that month (a mask, so the network can
                distinguish "paid nothing" from "no relationship yet")

Months are ordered oldest -> newest, i.e. index 0 is MONTHS_BALANCE = -SEQ_LEN
and index SEQ_LEN-1 is the most recent month.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (DATA_PROC, DATA_RAW, ID, RAW_FILES, SEQ_FEATURES, SEQ_LEN)

OUT = DATA_PROC / "sequences.npz"


def _monthly_frames() -> pd.DataFrame:
    """Long frame: one row per (customer, month) with all raw channel inputs."""
    frames = []

    cc = pd.read_csv(DATA_RAW / RAW_FILES["credit_card"],
                     usecols=[ID, "MONTHS_BALANCE", "AMT_BALANCE",
                              "AMT_CREDIT_LIMIT_ACTUAL", "SK_DPD"])
    cc["util"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    frames.append(cc[[ID, "MONTHS_BALANCE", "SK_DPD", "util"]])

    pos = pd.read_csv(DATA_RAW / RAW_FILES["pos_cash"],
                      usecols=[ID, "MONTHS_BALANCE", "SK_DPD"])
    pos["util"] = np.nan
    frames.append(pos[[ID, "MONTHS_BALANCE", "SK_DPD", "util"]])

    dpd = (pd.concat(frames, ignore_index=True)
             .groupby([ID, "MONTHS_BALANCE"], as_index=False)
             .agg(dpd=("SK_DPD", "max"), util=("util", "mean")))

    ins = pd.read_csv(DATA_RAW / RAW_FILES["installments"],
                      usecols=[ID, "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT",
                               "AMT_INSTALMENT", "AMT_PAYMENT"])
    # map a due date onto the monthly grid used by the balance tables
    ins["MONTHS_BALANCE"] = np.ceil(ins["DAYS_INSTALMENT"] / 30.0).astype(int)
    ins["late_days"] = ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]
    ins["pay_ratio"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"].replace(0, np.nan)
    ins_m = (ins.groupby([ID, "MONTHS_BALANCE"], as_index=False)
                .agg(late_days=("late_days", "mean"),
                     pay_ratio=("pay_ratio", "mean"),
                     n_payments=("AMT_PAYMENT", "size")))

    return dpd.merge(ins_m, on=[ID, "MONTHS_BALANCE"], how="outer")


def build(save: bool = True):
    app = pd.read_csv(DATA_RAW / RAW_FILES["application"], usecols=[ID])
    ids = app[ID].to_numpy()
    pos_of = pd.Series(np.arange(len(ids)), index=ids)

    long = _monthly_frames()
    long = long[(long["MONTHS_BALANCE"] >= -SEQ_LEN) & (long["MONTHS_BALANCE"] <= -1)]
    long = long[long[ID].isin(pos_of.index)]
    long["observed"] = 1.0

    n_ch = len(SEQ_FEATURES)
    X = np.zeros((len(ids), SEQ_LEN, n_ch), dtype=np.float32)

    row = pos_of.loc[long[ID].to_numpy()].to_numpy()
    col = long["MONTHS_BALANCE"].to_numpy() + SEQ_LEN          # -24 -> 0
    for c, name in enumerate(SEQ_FEATURES):
        vals = long[name].to_numpy(dtype=np.float64)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        X[row, col, c] = vals

    # scale each channel to roughly unit range using dev-set statistics only would
    # be ideal; channel-wise robust scaling on all rows is acceptable here because
    # no target information is involved.
    for c in range(n_ch):
        if SEQ_FEATURES[c] == "observed":
            continue
        v = X[:, :, c]
        lo, hi = np.percentile(v[v != 0] if (v != 0).any() else v, [1, 99])
        rng = max(hi - lo, 1e-6)
        X[:, :, c] = np.clip((v - lo) / rng, -5, 5)

    coverage = (X[:, :, SEQ_FEATURES.index("observed")].sum(axis=1) > 0).mean()
    if save:
        np.savez_compressed(OUT, ids=ids, X=X, channels=np.array(SEQ_FEATURES))
        print(f"sequences -> {OUT}  shape={X.shape}  "
              f"customers with history: {coverage:.1%}")
    return ids, X


def load():
    z = np.load(OUT, allow_pickle=True)
    return z["ids"], z["X"], list(z["channels"])


if __name__ == "__main__":
    build()
