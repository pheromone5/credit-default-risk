"""
Feature engineering, PySpark implementation.

Produces the same columns as features_pandas.py. The point is not that Spark is
faster here - on a single laptop with 20k customers it is slower, because you pay
JVM startup and shuffle overhead for nothing. The point is the shape of the
answer when someone asks "when would you reach for Spark?":

    installments_payments alone is ~13.6M rows in the real dataset, and the six
    tables together are ~2.5 GB. That fits in memory on a big machine, so pandas
    wins today. It stops fitting the moment the bank runs this over its full
    book with several years of monthly history, and the aggregation logic is
    identical either way - so writing it in Spark costs little and removes the
    rewrite later.

Run both paths and compare wall-clock time at each data size. Put that table in
the README. Numbers beat adjectives.

    python features_spark.py            # writes data/processed/features_spark.parquet
    python features_spark.py --compare  # diff against the pandas output
"""
from __future__ import annotations

import argparse
import time

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from config import DATA_PROC, DATA_RAW, ID, RAW_FILES

EMPLOYED_SENTINEL = 365243


def get_spark(app_name: str = "credit-features") -> SparkSession:
    return (SparkSession.builder
            .appName(app_name)
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.driver.memory", "2g")
            .config("spark.ui.enabled", "false")
            .getOrCreate())


def _read(spark: SparkSession, key: str) -> DataFrame:
    return (spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(str(DATA_RAW / RAW_FILES[key])))


def _agg(df: DataFrame, prefix: str, specs: dict[str, list[str]]) -> DataFrame:
    """specs maps column -> list of {count,sum,mean,max,min,std}."""
    fns = {"count": F.count, "sum": F.sum, "mean": F.avg,
           "max": F.max, "min": F.min, "std": F.stddev}
    exprs = [fns[fn](F.col(col)).alias(f"{prefix}_{col}_{fn}".upper())
             for col, how in specs.items() for fn in how]
    return df.groupBy(ID).agg(*exprs)


def application_features(spark: SparkSession) -> DataFrame:
    df = _read(spark, "application")
    df = (df
          .withColumn("FLAG_NOT_EMPLOYED",
                      (F.col("DAYS_EMPLOYED") == EMPLOYED_SENTINEL).cast("int"))
          .withColumn("DAYS_EMPLOYED",
                      F.when(F.col("DAYS_EMPLOYED") == EMPLOYED_SENTINEL, None)
                       .otherwise(F.col("DAYS_EMPLOYED")))
          .withColumn("CREDIT_INCOME_RATIO", F.col("AMT_CREDIT") / F.col("AMT_INCOME_TOTAL"))
          .withColumn("ANNUITY_INCOME_RATIO", F.col("AMT_ANNUITY") / F.col("AMT_INCOME_TOTAL"))
          .withColumn("CREDIT_GOODS_RATIO", F.col("AMT_CREDIT") / F.col("AMT_GOODS_PRICE"))
          .withColumn("CREDIT_TERM", F.col("AMT_ANNUITY") / F.col("AMT_CREDIT"))
          .withColumn("EMPLOYED_BIRTH_RATIO", F.col("DAYS_EMPLOYED") / F.col("DAYS_BIRTH"))
          .withColumn("INCOME_PER_PERSON",
                      F.col("AMT_INCOME_TOTAL") / F.greatest(F.col("CNT_FAM_MEMBERS"), F.lit(1)))
          .withColumn("AGE_YEARS", -F.col("DAYS_BIRTH") / 365.25))

    ext = [F.col(f"EXT_SOURCE_{i}") for i in (1, 2, 3)]
    arr = F.array(*ext)
    present = F.filter(arr, lambda x: x.isNotNull())
    df = (df
          .withColumn("EXT_SOURCE_MEAN", F.aggregate(present, F.lit(0.0), lambda a, b: a + b)
                      / F.greatest(F.size(present), F.lit(1)))
          .withColumn("EXT_SOURCE_MIN", F.array_min(present))
          .withColumn("EXT_SOURCE_NA_COUNT", F.lit(3) - F.size(present)))
    # row-wise sample std across the present EXT_SOURCE values (ddof=1, like pandas)
    n_present = F.size(present)
    sq_dev = F.aggregate(present, F.lit(0.0),
                         lambda acc, x: acc + F.pow(x - F.col("EXT_SOURCE_MEAN"), F.lit(2)))
    df = df.withColumn("EXT_SOURCE_STD",
                       F.when(n_present > 1, F.sqrt(sq_dev / (n_present - 1))))
    return df


def bureau_features(spark: SparkSession) -> DataFrame:
    b = (_read(spark, "bureau")
         .withColumn("DEBT_CREDIT_RATIO",
                     F.col("AMT_CREDIT_SUM_DEBT") /
                     F.when(F.col("AMT_CREDIT_SUM") == 0, None).otherwise(F.col("AMT_CREDIT_SUM")))
         .withColumn("IS_ACTIVE", (F.col("CREDIT_ACTIVE") == "Active").cast("int"))
         .withColumn("HAS_OVERDUE", (F.col("AMT_CREDIT_SUM_OVERDUE") > 0).cast("int")))
    return _agg(b, "BUR", {
        "SK_ID_BUREAU": ["count"],
        "IS_ACTIVE": ["sum", "mean"],
        "HAS_OVERDUE": ["sum", "mean"],
        "AMT_CREDIT_SUM": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_DEBT": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_OVERDUE": ["sum", "max"],
        "CREDIT_DAY_OVERDUE": ["mean", "max"],
        "DEBT_CREDIT_RATIO": ["mean", "max"],
        "DAYS_CREDIT": ["min", "max", "mean"],
    })


def previous_features(spark: SparkSession) -> DataFrame:
    p = (_read(spark, "previous")
         .withColumn("APP_CREDIT_RATIO",
                     F.col("AMT_APPLICATION") /
                     F.when(F.col("AMT_CREDIT") == 0, None).otherwise(F.col("AMT_CREDIT")))
         .withColumn("IS_REFUSED", (F.col("NAME_CONTRACT_STATUS") == "Refused").cast("int"))
         .withColumn("IS_APPROVED", (F.col("NAME_CONTRACT_STATUS") == "Approved").cast("int")))
    return _agg(p, "PREV", {
        "SK_ID_PREV": ["count"],
        "IS_REFUSED": ["sum", "mean"],
        "IS_APPROVED": ["sum", "mean"],
        "AMT_APPLICATION": ["mean", "max"],
        "AMT_CREDIT": ["mean", "max", "sum"],
        "AMT_ANNUITY": ["mean", "max"],
        "APP_CREDIT_RATIO": ["mean", "max"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "CNT_PAYMENT": ["mean", "max"],
    })


def installment_features(spark: SparkSession) -> DataFrame:
    i = (_read(spark, "installments")
         .withColumn("DAYS_LATE", F.col("DAYS_ENTRY_PAYMENT") - F.col("DAYS_INSTALMENT"))
         .withColumn("IS_LATE", (F.col("DAYS_ENTRY_PAYMENT") - F.col("DAYS_INSTALMENT") > 0).cast("int"))
         .withColumn("IS_LATE_30", (F.col("DAYS_ENTRY_PAYMENT") - F.col("DAYS_INSTALMENT") > 30).cast("int"))
         .withColumn("PAYMENT_RATIO",
                     F.col("AMT_PAYMENT") /
                     F.when(F.col("AMT_INSTALMENT") == 0, None).otherwise(F.col("AMT_INSTALMENT")))
         .withColumn("UNDERPAID", (F.col("AMT_PAYMENT") < F.col("AMT_INSTALMENT") - 1).cast("int"))
         .withColumn("SHORTFALL", F.greatest(F.col("AMT_INSTALMENT") - F.col("AMT_PAYMENT"), F.lit(0.0))))

    base = _agg(i, "INS", {
        "NUM_INSTALMENT_NUMBER": ["count"],
        "DAYS_LATE": ["mean", "max", "std", "sum"],
        "IS_LATE": ["mean", "sum"],
        "IS_LATE_30": ["mean", "sum"],
        "PAYMENT_RATIO": ["mean", "min", "std"],
        "UNDERPAID": ["mean", "sum"],
        "SHORTFALL": ["sum", "max"],
        "AMT_PAYMENT": ["sum", "mean"],
        "AMT_INSTALMENT": ["sum", "mean"],
    })
    recent = _agg(i.filter(F.col("DAYS_INSTALMENT") >= -365), "INS_RECENT",
                  {"DAYS_LATE": ["mean", "max"], "IS_LATE": ["mean"]})
    return base.join(recent, on=ID, how="left")


def credit_card_features(spark: SparkSession) -> DataFrame:
    c = (_read(spark, "credit_card")
         .withColumn("UTILISATION",
                     F.col("AMT_BALANCE") /
                     F.when(F.col("AMT_CREDIT_LIMIT_ACTUAL") == 0, None)
                      .otherwise(F.col("AMT_CREDIT_LIMIT_ACTUAL")))
         .withColumn("HAS_DPD", (F.col("SK_DPD") > 0).cast("int")))
    return _agg(c, "CC", {
        "MONTHS_BALANCE": ["count", "min"],
        "UTILISATION": ["mean", "max", "std"],
        "AMT_BALANCE": ["mean", "max"],
        "SK_DPD": ["mean", "max", "sum"],
        "HAS_DPD": ["mean", "sum"],
        "CNT_DRAWINGS_CURRENT": ["mean", "max"],
    })


def pos_cash_features(spark: SparkSession) -> DataFrame:
    p = _read(spark, "pos_cash").withColumn("HAS_DPD", (F.col("SK_DPD") > 0).cast("int"))
    return _agg(p, "POS", {
        "MONTHS_BALANCE": ["count", "min"],
        "CNT_INSTALMENT": ["mean", "max"],
        "CNT_INSTALMENT_FUTURE": ["mean", "min"],
        "SK_DPD": ["mean", "max", "sum"],
        "HAS_DPD": ["mean", "sum"],
    })


def build(spark: SparkSession | None = None, save: bool = True):
    spark = spark or get_spark()
    df = application_features(spark)
    for fn in (bureau_features, previous_features, installment_features,
               credit_card_features, pos_cash_features):
        df = df.join(fn(spark), on=ID, how="left")

    zero_fill = [c for c in df.columns if c.endswith(("_COUNT", "_SUM"))]
    df = df.fillna(0, subset=zero_fill)

    pdf = df.toPandas()
    if save:
        out = DATA_PROC / "features_spark.parquet"
        pdf.to_parquet(out, index=False)
        print(f"spark features -> {out}  shape={pdf.shape}")
    return pdf


def compare():
    import pandas as pd
    a = pd.read_parquet(DATA_PROC / "features.parquet").sort_values(ID).reset_index(drop=True)
    b = pd.read_parquet(DATA_PROC / "features_spark.parquet").sort_values(ID).reset_index(drop=True)
    shared = [c for c in a.columns if c in b.columns]
    print(f"columns: pandas={a.shape[1]} spark={b.shape[1]} shared={len(shared)}")
    print("pandas-only:", sorted(set(a.columns) - set(b.columns)))
    print("spark-only :", sorted(set(b.columns) - set(a.columns)))
    bad = []
    for c in shared:
        if a[c].dtype.kind in "if" and b[c].dtype.kind in "if":
            d = (a[c].fillna(-999) - b[c].fillna(-999)).abs().max()
            if d > 1e-3:
                bad.append((c, float(d)))
    print(f"numeric columns disagreeing by >1e-3: {len(bad)}")
    for c, d in sorted(bad, key=lambda x: -x[1])[:10]:
        print(f"  {c:40s} max abs diff {d:.4g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    if args.compare:
        compare()
    else:
        t0 = time.time()
        build()
        print(f"elapsed {time.time() - t0:.1f}s (includes JVM startup)")
