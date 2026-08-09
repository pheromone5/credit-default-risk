"""
Generate synthetic data that mirrors the Home Credit Default Risk schema.

WHY THIS EXISTS
---------------
The real dataset lives behind a Kaggle competition login and is ~2.5 GB. This
module produces files with the *same names, same columns and same dtypes*, so
every downstream script runs unchanged the moment you drop the real CSVs into
data/raw/. Use it to develop and debug the pipeline, never to report results.

HONEST NOTE ON THE GENERATIVE PROCESS
-------------------------------------
The latent risk score has two parts:
  * z_static  -> leaks into EXT_SOURCE_*, income, employment, education
  * z_traj    -> leaks ONLY into the *slope* of monthly delinquency

The monthly level is offset so that the mean DPD over the window is independent
of the slope. That means simple mean/max/sum aggregation cannot see z_traj, but
a sequence model can. This is deliberately favourable to the GRU. On the real
Home Credit data no such guarantee exists and the sequence model typically does
*not* beat LightGBM standalone. Do not quote synthetic numbers anywhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATA_RAW, SEED, SEQ_LEN

RNG = np.random.default_rng(SEED)

N_CUSTOMERS = 20_000
BASE_DEFAULT_RATE = 0.08

INCOME_TYPES = ["Working", "State servant", "Commercial associate", "Pensioner", "Student"]
EDUCATION = ["Lower secondary", "Secondary / secondary special", "Incomplete higher",
             "Higher education", "Academic degree"]
FAMILY = ["Single / not married", "Married", "Civil marriage", "Widow", "Separated"]
HOUSING = ["House / apartment", "With parents", "Municipal apartment", "Rented apartment",
           "Office apartment", "Co-op apartment"]
OCCUPATION = ["Laborers", "Sales staff", "Core staff", "Managers", "Drivers",
              "High skill tech staff", "Accountants", "Medicine staff", "Security staff",
              "Cooking staff", "Cleaning staff", "Private service staff", None]
ORGANIZATION = ["Business Entity Type 3", "Self-employed", "Other", "Medicine",
                "Government", "School", "Trade: type 7", "Kindergarten", "Construction",
                "Business Entity Type 2", "Transport: type 4", "XNA"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _pick(options, n, p=None):
    idx = RNG.choice(len(options), size=n, p=p)
    return np.array([options[i] for i in idx], dtype=object)


def build_application() -> tuple[pd.DataFrame, np.ndarray]:
    n = N_CUSTOMERS
    ids = np.arange(100_001, 100_001 + n)

    z_static = RNG.normal(size=n)
    z_traj = RNG.normal(size=n)

    # target driven by both components; intercept solved for the base rate
    logit = -0.9 * z_static - 0.55 * z_traj
    lo, hi = -10.0, 10.0                      # bisect for the exact base rate
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(mid + logit).mean() < BASE_DEFAULT_RATE:
            lo = mid
        else:
            hi = mid
    target = RNG.binomial(1, _sigmoid((lo + hi) / 2 + logit))

    # EXT_SOURCE_* are the dominant features in the real data: noisy views of risk
    ext1 = np.clip(_sigmoid(1.1 * z_static + RNG.normal(0, 0.9, n)), 0.001, 0.999)
    ext2 = np.clip(_sigmoid(1.3 * z_static + RNG.normal(0, 0.7, n)), 0.001, 0.999)
    ext3 = np.clip(_sigmoid(0.9 * z_static + RNG.normal(0, 1.1, n)), 0.001, 0.999)
    # ~55% missingness on EXT_SOURCE_1, as in the real file
    ext1[RNG.random(n) < 0.55] = np.nan
    ext3[RNG.random(n) < 0.20] = np.nan

    income = np.round(np.exp(RNG.normal(11.9 + 0.18 * z_static, 0.45, n)) / 1000) * 1000
    credit = np.round(income * np.exp(RNG.normal(0.75 - 0.10 * z_static, 0.55, n)) / 1000) * 1000
    annuity = np.round(credit * RNG.uniform(0.04, 0.14, n) / 100) * 100
    goods = np.round(credit * np.clip(RNG.normal(0.92, 0.10, n), 0.4, 1.0) / 1000) * 1000

    days_birth = -np.round(RNG.uniform(21, 68, n) * 365.25 + 40 * z_static).astype(int)
    employed = -np.round(np.abs(RNG.gamma(2.0, 900 + 260 * z_static.clip(-2, 3), n))).astype(int)
    # the real file encodes "not employed" as the sentinel 365243
    employed[RNG.random(n) < 0.18] = 365243

    df = pd.DataFrame({
        "SK_ID_CURR": ids,
        "TARGET": target,
        "NAME_CONTRACT_TYPE": _pick(["Cash loans", "Revolving loans"], n, p=[0.9, 0.1]),
        "CODE_GENDER": _pick(["M", "F", "XNA"], n, p=[0.34, 0.6599, 0.0001]),
        "FLAG_OWN_CAR": _pick(["Y", "N"], n, p=[0.34, 0.66]),
        "FLAG_OWN_REALTY": _pick(["Y", "N"], n, p=[0.69, 0.31]),
        "CNT_CHILDREN": RNG.poisson(0.42, n),
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods,
        "NAME_INCOME_TYPE": _pick(INCOME_TYPES, n, p=[0.52, 0.07, 0.23, 0.17, 0.01]),
        "NAME_EDUCATION_TYPE": _pick(EDUCATION, n, p=[0.04, 0.71, 0.10, 0.145, 0.005]),
        "NAME_FAMILY_STATUS": _pick(FAMILY, n, p=[0.15, 0.64, 0.10, 0.06, 0.05]),
        "NAME_HOUSING_TYPE": _pick(HOUSING, n, p=[0.89, 0.05, 0.03, 0.02, 0.005, 0.005]),
        "OCCUPATION_TYPE": _pick(OCCUPATION, n),
        "ORGANIZATION_TYPE": _pick(ORGANIZATION, n),
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": employed,
        "DAYS_REGISTRATION": -np.round(RNG.uniform(0, 20000, n)).astype(int),
        "DAYS_ID_PUBLISH": -np.round(RNG.uniform(0, 6000, n)).astype(int),
        "REGION_POPULATION_RELATIVE": np.round(RNG.uniform(0.0005, 0.073, n), 5),
        "REGION_RATING_CLIENT": RNG.choice([1, 2, 3], n, p=[0.13, 0.74, 0.13]),
        "CNT_FAM_MEMBERS": 1 + RNG.poisson(1.1, n),
        "EXT_SOURCE_1": np.round(ext1, 6),
        "EXT_SOURCE_2": np.round(ext2, 6),
        "EXT_SOURCE_3": np.round(ext3, 6),
    })
    return df, z_traj


def build_bureau(app: pd.DataFrame) -> pd.DataFrame:
    n_rows_per = RNG.poisson(3.2, len(app))
    ids = np.repeat(app["SK_ID_CURR"].to_numpy(), n_rows_per)
    m = len(ids)
    risk = np.repeat((1 - app["EXT_SOURCE_2"].to_numpy()), n_rows_per)
    credit_sum = np.round(np.exp(RNG.normal(11.5, 1.2, m)) / 100) * 100
    debt = credit_sum * RNG.beta(1.5, 4 - 2 * risk, m)
    overdue = np.where(RNG.random(m) < 0.05 + 0.15 * risk,
                       np.round(debt * RNG.uniform(0.01, 0.3, m)), 0.0)
    return pd.DataFrame({
        "SK_ID_CURR": ids,
        "SK_ID_BUREAU": np.arange(5_000_001, 5_000_001 + m),
        "CREDIT_ACTIVE": _pick(["Closed", "Active", "Sold"], m, p=[0.53, 0.45, 0.02]),
        "CREDIT_CURRENCY": "currency 1",
        "DAYS_CREDIT": -np.round(RNG.uniform(1, 2900, m)).astype(int),
        "CREDIT_DAY_OVERDUE": np.where(overdue > 0, RNG.integers(1, 120, m), 0),
        "AMT_CREDIT_SUM": credit_sum,
        "AMT_CREDIT_SUM_DEBT": np.round(debt, 2),
        "AMT_CREDIT_SUM_OVERDUE": np.round(overdue, 2),
        "CREDIT_TYPE": _pick(["Consumer credit", "Credit card", "Car loan", "Mortgage"],
                             m, p=[0.63, 0.3, 0.05, 0.02]),
    })


def build_previous(app: pd.DataFrame) -> pd.DataFrame:
    n_prev = 1 + RNG.poisson(1.4, len(app))
    ids = np.repeat(app["SK_ID_CURR"].to_numpy(), n_prev)
    m = len(ids)
    amt_app = np.round(np.exp(RNG.normal(11.8, 0.9, m)) / 1000) * 1000
    return pd.DataFrame({
        "SK_ID_PREV": np.arange(1_000_001, 1_000_001 + m),
        "SK_ID_CURR": ids,
        "NAME_CONTRACT_TYPE": _pick(["Cash loans", "Consumer loans", "Revolving loans"],
                                    m, p=[0.44, 0.44, 0.12]),
        "AMT_APPLICATION": amt_app,
        "AMT_CREDIT": np.round(amt_app * RNG.uniform(0.8, 1.15, m) / 1000) * 1000,
        "AMT_ANNUITY": np.round(amt_app * RNG.uniform(0.04, 0.15, m) / 100) * 100,
        "NAME_CONTRACT_STATUS": _pick(["Approved", "Refused", "Canceled", "Unused offer"],
                                      m, p=[0.62, 0.17, 0.19, 0.02]),
        "DAYS_DECISION": -np.round(RNG.uniform(1, 2900, m)).astype(int),
        "CNT_PAYMENT": RNG.choice([0, 6, 12, 18, 24, 36, 48], m,
                                  p=[0.05, 0.16, 0.29, 0.13, 0.2, 0.12, 0.05]),
    })


def _monthly_dpd(app: pd.DataFrame, z_traj: np.ndarray) -> np.ndarray:
    """(n_customers, SEQ_LEN) delinquency panel, months ordered -SEQ_LEN .. -1.

    slope carries z_traj; the level is offset so the row mean does not.
    """
    n = len(app)
    t = np.arange(SEQ_LEN)                       # 0 = oldest month
    centred = t - t.mean()
    risk = (1 - app["EXT_SOURCE_2"].to_numpy())
    level = 2.5 + 9.0 * np.clip(risk, 0, 1)      # driven by z_static only
    slope = 0.55 * z_traj + RNG.normal(0, 0.12, n)

    panel = level[:, None] + slope[:, None] * centred[None, :]
    panel = panel + RNG.normal(0, 1.6, size=panel.shape)
    panel = np.clip(np.round(panel), 0, None)
    # most customer-months are perfectly clean, as in the real data
    panel[RNG.random(panel.shape) < 0.55] = 0
    return panel.astype(int)


def build_monthly(app: pd.DataFrame, prev: pd.DataFrame, z_traj: np.ndarray):
    """POS_CASH_balance, credit_card_balance and installments_payments."""
    panel = _monthly_dpd(app, z_traj)
    id_to_row = {sk: i for i, sk in enumerate(app["SK_ID_CURR"].to_numpy())}

    # one representative previous contract per customer carries the monthly history
    anchor = prev.groupby("SK_ID_CURR", as_index=False)["SK_ID_PREV"].min()
    anchor["is_card"] = RNG.random(len(anchor)) < 0.45

    months = np.arange(-SEQ_LEN, 0)
    rows_curr = np.repeat(anchor["SK_ID_CURR"].to_numpy(), SEQ_LEN)
    rows_prev = np.repeat(anchor["SK_ID_PREV"].to_numpy(), SEQ_LEN)
    is_card = np.repeat(anchor["is_card"].to_numpy(), SEQ_LEN)
    mb = np.tile(months, len(anchor))
    dpd = np.array([panel[id_to_row[c], m + SEQ_LEN] for c, m in zip(rows_curr, mb)])

    limit = np.repeat(np.round(app.set_index("SK_ID_CURR")
                               .loc[anchor["SK_ID_CURR"], "AMT_CREDIT"].to_numpy() * 0.3
                               / 1000) * 1000, SEQ_LEN)
    util = np.clip(RNG.beta(2, 3, len(mb)) + 0.02 * dpd, 0, 1.4)

    card = pd.DataFrame({
        "SK_ID_PREV": rows_prev, "SK_ID_CURR": rows_curr, "MONTHS_BALANCE": mb,
        "AMT_BALANCE": np.round(limit * util, 2),
        "AMT_CREDIT_LIMIT_ACTUAL": limit,
        "AMT_PAYMENT_TOTAL_CURRENT": np.round(limit * util * RNG.uniform(0.02, 0.4, len(mb)), 2),
        "CNT_DRAWINGS_CURRENT": RNG.poisson(1.2, len(mb)),
        "SK_DPD": dpd,
        "NAME_CONTRACT_STATUS": "Active",
    })[is_card]

    cnt_inst = np.repeat(RNG.choice([12, 18, 24, 36], len(anchor)), SEQ_LEN)
    pos = pd.DataFrame({
        "SK_ID_PREV": rows_prev, "SK_ID_CURR": rows_curr, "MONTHS_BALANCE": mb,
        "CNT_INSTALMENT": cnt_inst,
        "CNT_INSTALMENT_FUTURE": np.clip(cnt_inst + mb, 0, None),
        "NAME_CONTRACT_STATUS": "Active",
        "SK_DPD": dpd,
    })[~is_card]

    # instalment ledger: one row per scheduled payment, late by roughly the DPD
    keep = RNG.random(len(mb)) < 0.85
    due_amt = np.round(np.repeat(
        app.set_index("SK_ID_CURR").loc[anchor["SK_ID_CURR"], "AMT_ANNUITY"].to_numpy(),
        SEQ_LEN) * RNG.uniform(0.8, 1.2, len(mb)), 2)
    days_inst = mb * 30 - RNG.integers(0, 12, len(mb))
    late = np.clip(dpd + RNG.normal(0, 2, len(mb)), 0, None).round().astype(int)
    paid_ratio = np.clip(RNG.normal(1.0, 0.05, len(mb)) - 0.01 * dpd, 0, 1.2)
    inst = pd.DataFrame({
        "SK_ID_PREV": rows_prev, "SK_ID_CURR": rows_curr,
        "NUM_INSTALMENT_VERSION": 1.0,
        "NUM_INSTALMENT_NUMBER": np.tile(np.arange(1, SEQ_LEN + 1), len(anchor)),
        "DAYS_INSTALMENT": days_inst,
        "DAYS_ENTRY_PAYMENT": days_inst + late,
        "AMT_INSTALMENT": due_amt,
        "AMT_PAYMENT": np.round(due_amt * paid_ratio, 2),
    })[keep]

    return pos, card, inst


def main():
    print(f"generating {N_CUSTOMERS:,} synthetic customers -> {DATA_RAW}")
    app, z_traj = build_application()
    bureau = build_bureau(app)
    prev = build_previous(app)
    pos, card, inst = build_monthly(app, prev, z_traj)

    for name, df in [("application_train", app), ("bureau", bureau),
                     ("previous_application", prev), ("POS_CASH_balance", pos),
                     ("credit_card_balance", card), ("installments_payments", inst)]:
        path = DATA_RAW / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"  {name:24s} {len(df):>9,} rows  {path.stat().st_size / 1e6:6.1f} MB")

    print(f"  default rate: {app['TARGET'].mean():.3%}")


if __name__ == "__main__":
    main()
