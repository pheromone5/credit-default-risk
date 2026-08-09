"""
Feature engineering, pandas implementation (the reference).

features_spark.py mirrors this logic on the same column names so the two can be
diffed against each other. Run both and compare - that comparison is the point
of having a Spark path at all, and it is what you talk about in an interview.

Design notes worth defending out loud:
  * Ratios beat raw amounts. A 500k loan means nothing without the income it
    sits against, so CREDIT/INCOME and ANNUITY/INCOME carry more signal than
    either component alone.
  * DAYS_EMPLOYED contains the sentinel 365243 for "not employed". Left alone it
    poisons every mean. It is nulled and flagged.
  * days_late = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT is the single most useful
    engineered field in the whole dataset: positive means paid late.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATA_PROC, DATA_RAW, ID, RAW_FILES

EMPLOYED_SENTINEL = 365243


def _flatten(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df.columns = [f"{prefix}_{a}_{b}".upper() if b else f"{prefix}_{a}".upper()
                  for a, b in df.columns]
    return df.reset_index()


def application_features() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / RAW_FILES["application"])

    df["FLAG_NOT_EMPLOYED"] = (df["DAYS_EMPLOYED"] == EMPLOYED_SENTINEL).astype(int)
    df.loc[df["DAYS_EMPLOYED"] == EMPLOYED_SENTINEL, "DAYS_EMPLOYED"] = np.nan

    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    df["EMPLOYED_BIRTH_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].clip(lower=1)
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25

    ext = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    df["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
    df["EXT_SOURCE_MIN"] = ext.min(axis=1)
    df["EXT_SOURCE_STD"] = ext.std(axis=1)
    df["EXT_SOURCE_NA_COUNT"] = ext.isna().sum(axis=1)
    return df


def bureau_features() -> pd.DataFrame:
    b = pd.read_csv(DATA_RAW / RAW_FILES["bureau"])
    b["DEBT_CREDIT_RATIO"] = b["AMT_CREDIT_SUM_DEBT"] / b["AMT_CREDIT_SUM"].replace(0, np.nan)
    b["IS_ACTIVE"] = (b["CREDIT_ACTIVE"] == "Active").astype(int)
    b["HAS_OVERDUE"] = (b["AMT_CREDIT_SUM_OVERDUE"] > 0).astype(int)

    agg = b.groupby(ID).agg({
        "SK_ID_BUREAU": "count",
        "IS_ACTIVE": ["sum", "mean"],
        "HAS_OVERDUE": ["sum", "mean"],
        "AMT_CREDIT_SUM": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_DEBT": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_OVERDUE": ["sum", "max"],
        "CREDIT_DAY_OVERDUE": ["mean", "max"],
        "DEBT_CREDIT_RATIO": ["mean", "max"],
        "DAYS_CREDIT": ["min", "max", "mean"],
    })
    return _flatten(agg, "BUR")


def previous_features() -> pd.DataFrame:
    p = pd.read_csv(DATA_RAW / RAW_FILES["previous"])
    p["APP_CREDIT_RATIO"] = p["AMT_APPLICATION"] / p["AMT_CREDIT"].replace(0, np.nan)
    p["IS_REFUSED"] = (p["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    p["IS_APPROVED"] = (p["NAME_CONTRACT_STATUS"] == "Approved").astype(int)

    agg = p.groupby(ID).agg({
        "SK_ID_PREV": "count",
        "IS_REFUSED": ["sum", "mean"],
        "IS_APPROVED": ["sum", "mean"],
        "AMT_APPLICATION": ["mean", "max"],
        "AMT_CREDIT": ["mean", "max", "sum"],
        "AMT_ANNUITY": ["mean", "max"],
        "APP_CREDIT_RATIO": ["mean", "max"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "CNT_PAYMENT": ["mean", "max"],
    })
    return _flatten(agg, "PREV")


def installment_features() -> pd.DataFrame:
    """The behavioural core: did this customer pay on time, and in full?"""
    i = pd.read_csv(DATA_RAW / RAW_FILES["installments"])
    i["DAYS_LATE"] = i["DAYS_ENTRY_PAYMENT"] - i["DAYS_INSTALMENT"]
    i["IS_LATE"] = (i["DAYS_LATE"] > 0).astype(int)
    i["IS_LATE_30"] = (i["DAYS_LATE"] > 30).astype(int)
    i["PAYMENT_RATIO"] = i["AMT_PAYMENT"] / i["AMT_INSTALMENT"].replace(0, np.nan)
    i["UNDERPAID"] = (i["AMT_PAYMENT"] < i["AMT_INSTALMENT"] - 1).astype(int)
    i["SHORTFALL"] = (i["AMT_INSTALMENT"] - i["AMT_PAYMENT"]).clip(lower=0)

    agg = i.groupby(ID).agg({
        "NUM_INSTALMENT_NUMBER": "count",
        "DAYS_LATE": ["mean", "max", "std", "sum"],
        "IS_LATE": ["mean", "sum"],
        "IS_LATE_30": ["mean", "sum"],
        "PAYMENT_RATIO": ["mean", "min", "std"],
        "UNDERPAID": ["mean", "sum"],
        "SHORTFALL": ["sum", "max"],
        "AMT_PAYMENT": ["sum", "mean"],
        "AMT_INSTALMENT": ["sum", "mean"],
    })
    out = _flatten(agg, "INS")

    # recency split: behaviour in the last 12 months vs everything before it
    recent = i[i["DAYS_INSTALMENT"] >= -365]
    r = recent.groupby(ID).agg({"DAYS_LATE": ["mean", "max"], "IS_LATE": "mean"})
    out = out.merge(_flatten(r, "INS_RECENT"), on=ID, how="left")
    return out


def credit_card_features() -> pd.DataFrame:
    c = pd.read_csv(DATA_RAW / RAW_FILES["credit_card"])
    c["UTILISATION"] = c["AMT_BALANCE"] / c["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    c["HAS_DPD"] = (c["SK_DPD"] > 0).astype(int)
    agg = c.groupby(ID).agg({
        "MONTHS_BALANCE": ["count", "min"],
        "UTILISATION": ["mean", "max", "std"],
        "AMT_BALANCE": ["mean", "max"],
        "SK_DPD": ["mean", "max", "sum"],
        "HAS_DPD": ["mean", "sum"],
        "CNT_DRAWINGS_CURRENT": ["mean", "max"],
    })
    return _flatten(agg, "CC")


def pos_cash_features() -> pd.DataFrame:
    p = pd.read_csv(DATA_RAW / RAW_FILES["pos_cash"])
    p["HAS_DPD"] = (p["SK_DPD"] > 0).astype(int)
    agg = p.groupby(ID).agg({
        "MONTHS_BALANCE": ["count", "min"],
        "CNT_INSTALMENT": ["mean", "max"],
        "CNT_INSTALMENT_FUTURE": ["mean", "min"],
        "SK_DPD": ["mean", "max", "sum"],
        "HAS_DPD": ["mean", "sum"],
    })
    return _flatten(agg, "POS")


def build(save: bool = True) -> pd.DataFrame:
    app = application_features()
    for fn in (bureau_features, previous_features, installment_features,
               credit_card_features, pos_cash_features):
        app = app.merge(fn(), on=ID, how="left")

    # counts are genuinely zero when a customer has no rows in that table
    for col in [c for c in app.columns if c.endswith(("_COUNT", "_SUM"))]:
        app[col] = app[col].fillna(0)

    if save:
        out = DATA_PROC / "features.parquet"
        app.to_parquet(out, index=False)
        print(f"pandas features -> {out}  shape={app.shape}")
    return app


if __name__ == "__main__":
    import time
    t0 = time.time()
    df = build()
    print(f"elapsed {time.time() - t0:.1f}s   {df.shape[1]} columns")
