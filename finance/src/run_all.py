"""
Run the whole pipeline in order.

    python run_all.py                 # everything, pandas feature path
    python run_all.py --engine spark  # use the PySpark aggregation instead
    python run_all.py --skip-synth    # you have dropped the real Kaggle CSVs in

Each stage prints its own metrics; the summary table at the end is what belongs
in the README.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from config import REPORTS


def _stage(label: str, fn, *args, **kwargs):
    print("\n" + "=" * 78)
    print(label)
    print("=" * 78)
    t0 = time.time()
    fn(*args, **kwargs)
    dt = time.time() - t0
    print(f"[{label} finished in {dt:.1f}s]")
    return dt


def main(engine: str = "pandas", skip_synth: bool = False, skip_ablations: bool = False):
    timings = {}

    if not skip_synth:
        import synth_data
        timings["synthetic data"] = _stage("0. generate synthetic data", synth_data.main)

    import validation
    timings["splits"] = _stage("1. validation strategy",
                               lambda: print(validation.build_splits(force=True)
                                             .groupby(["split", "fold"])["TARGET"]
                                             .agg(["size", "mean"]).to_string()))

    if engine == "spark":
        import features_spark
        timings["features"] = _stage("2. features (PySpark)", features_spark.build)
        # downstream code reads features.parquet, so mirror the spark output there
        import shutil
        from config import DATA_PROC
        shutil.copy(DATA_PROC / "features_spark.parquet", DATA_PROC / "features.parquet")
    else:
        import features_pandas
        timings["features"] = _stage("2. features (pandas)", features_pandas.build)

    import build_sequences
    timings["sequences"] = _stage("3. sequence tensor", build_sequences.build)

    import train_baseline
    timings["logistic"] = _stage("4. logistic baseline", train_baseline.main)

    import train_lgbm
    timings["lgbm"] = _stage("5. LightGBM", train_lgbm.main, imbalance="weight")
    if not skip_ablations:
        _stage("5b. LightGBM, no imbalance handling", train_lgbm.main, imbalance="none")

    import train_seq_model
    timings["gru"] = _stage("6. GRU sequence model", train_seq_model.main)
    if not skip_ablations:
        _stage("6b. ablation: same network without the GRU",
               train_seq_model.main, no_seq=True)

    import blend_calibrate
    timings["blend"] = _stage("7. blend + calibrate", blend_calibrate.main)

    import explain_shap
    timings["shap"] = _stage("8. SHAP explainability", explain_shap.main)

    import business_policy
    timings["policy"] = _stage("9. approval policy + PSI", business_policy.main)

    from data_prep import all_results
    res = all_results()
    if not res.empty:
        cols = [c for c in ["model", "AUC", "Gini", "KS", "PR_AUC", "Brier",
                            "holdout_AUC", "holdout_KS", "n_features"] if c in res.columns]
        table = res[cols].sort_values("holdout_AUC", ascending=False)
        print("\n" + "=" * 78)
        print("SUMMARY  (AUC/Gini/KS/Brier are out-of-fold; holdout_* is the frozen set)")
        print("=" * 78)
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        (REPORTS / "results_table.md").write_text(table.to_markdown(index=False,
                                                                   floatfmt=".4f"))

    print("\nstage timings (s):")
    for k, v in timings.items():
        print(f"  {k:<16} {v:7.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["pandas", "spark"], default="pandas")
    ap.add_argument("--skip-synth", action="store_true")
    ap.add_argument("--skip-ablations", action="store_true")
    main(**vars(ap.parse_args()))
