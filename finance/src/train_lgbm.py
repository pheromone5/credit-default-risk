"""
Stage 3 - LightGBM on the full engineered feature set.

Two things here are worth more in an interview than the AUC itself.

1. IMBALANCE IS AN EXPERIMENT, NOT A REFLEX.
   `--imbalance {none,weight,smote}` runs the comparison instead of assuming.
   On this kind of problem, oversampling usually does not improve ranking and
   always wrecks calibration, because it changes the base rate the model is
   fitting. Being able to say "I tested SMOTE, it cost me Brier and gained me
   nothing on AUC, so I used class weights and calibrated afterwards" is a
   stronger answer than listing SMOTE as a skill.

2. EARLY STOPPING USES THE FOLD'S OWN VALIDATION SET.
   That is a mild optimism in the OOF numbers, which is why the frozen holdout
   exists and why the CV bullet should quote the holdout figure.
"""
from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import ARTIFACTS, ID, N_FOLDS, SEED, TARGET
from data_prep import (feature_columns, load_features, object_columns,
                       record_result, save_predictions, split_frame,
                       as_categorical)
from metrics import scorecard_report

NAME = "lgbm"

PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=60,
    feature_fraction=0.75,
    bagging_fraction=0.85,
    bagging_freq=1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_estimators=2000,
    verbose=-1,
    seed=SEED,
    n_jobs=-1,
)


def _resample_smote(X: pd.DataFrame, y: np.ndarray, cat_cols):
    """SMOTE-NC, imported lazily so imbalanced-learn stays optional."""
    from imblearn.over_sampling import SMOTENC
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    sm = SMOTENC(categorical_features=cat_idx, random_state=SEED, sampling_strategy=0.3)
    Xr, yr = sm.fit_resample(X, y)
    return Xr, yr


def main(imbalance: str = "weight"):
    df = load_features()
    cat_cols = object_columns(df)
    df = as_categorical(df, cat_cols)
    dev, hold = split_frame(df)
    cols = feature_columns(dev)

    y = dev[TARGET].to_numpy()
    oof = np.zeros(len(dev))
    hold_pred = np.zeros(len(hold))
    importances = np.zeros(len(cols))
    best_iters = []

    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

    for k in range(N_FOLDS):
        tr = dev["fold"].to_numpy() != k
        va = ~tr
        Xtr, ytr = dev.loc[tr, cols], y[tr]

        params = dict(PARAMS)
        if imbalance == "weight":
            params["scale_pos_weight"] = pos_weight
        elif imbalance == "smote":
            Xtr, ytr = _resample_smote(Xtr, ytr, cat_cols)

        model = lgb.LGBMClassifier(**params)
        model.fit(Xtr, ytr,
                  eval_set=[(dev.loc[va, cols], y[va])],
                  eval_metric="auc",
                  callbacks=[lgb.early_stopping(100, verbose=False)])

        oof[va] = model.predict_proba(dev.loc[va, cols])[:, 1]
        hold_pred += model.predict_proba(hold[cols])[:, 1] / N_FOLDS
        importances += model.feature_importances_ / N_FOLDS
        best_iters.append(model.best_iteration_ or params["n_estimators"])

    rep = scorecard_report(y, oof, f"{NAME} (OOF)")
    rep_h = scorecard_report(hold[TARGET].to_numpy(), hold_pred, f"{NAME} (holdout)")
    print(f"imbalance={imbalance}  mean best_iter={np.mean(best_iters):.0f}")
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in rep.items()})
    print(f"holdout AUC={rep_h['AUC']:.4f}  KS={rep_h['KS']:.4f}")

    imp = (pd.DataFrame({"feature": cols, "gain": importances})
           .sort_values("gain", ascending=False))
    imp.to_csv(ARTIFACTS / "lgbm_feature_importance.csv", index=False)
    print("\ntop 15 features:")
    print(imp.head(15).to_string(index=False))

    if imbalance == "weight":     # only the chosen configuration feeds the blend
        save_predictions(NAME, dev[ID], oof, hold[ID], hold_pred)
        record_result({**rep, "model": NAME, "n_features": len(cols),
                       "holdout_AUC": rep_h["AUC"], "holdout_KS": rep_h["KS"]})
    else:
        record_result({**rep, "model": f"{NAME}_{imbalance}", "n_features": len(cols),
                       "holdout_AUC": rep_h["AUC"], "holdout_KS": rep_h["KS"]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--imbalance", choices=["none", "weight", "smote"], default="weight")
    main(**vars(ap.parse_args()))
