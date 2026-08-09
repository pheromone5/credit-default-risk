"""
Stage 1 - logistic regression on application-level features only.

Its job is not to win. Its job is to be the number every later model is measured
against, and to be a model whose coefficients you can read aloud. If LightGBM
only beats this by 0.005 AUC, that is worth knowing before you spend a week on a
neural network.

Deliberately kept simple: median imputation, standardisation, one-hot on the
low-cardinality categoricals, L2 penalty, class_weight balanced.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import ID, N_FOLDS, SEED, TARGET
from data_prep import (feature_columns, load_features, object_columns,
                       record_result, save_predictions, split_frame)
from metrics import scorecard_report

NAME = "logistic"


def make_pipeline(num_cols, cat_cols) -> Pipeline:
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01))])
    pre = ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(max_iter=2000, C=0.1,
                                                class_weight="balanced",
                                                random_state=SEED))])


def main():
    df = load_features(app_only=True)
    dev, hold = split_frame(df)
    cols = feature_columns(dev)
    cat_cols = object_columns(dev)
    num_cols = [c for c in cols if c not in cat_cols]

    y = dev[TARGET].to_numpy()
    oof = np.zeros(len(dev))
    hold_pred = np.zeros(len(hold))

    for k in range(N_FOLDS):
        tr = dev["fold"].to_numpy() != k
        va = ~tr
        pipe = make_pipeline(num_cols, cat_cols)
        pipe.fit(dev.loc[tr, cols], y[tr])
        oof[va] = pipe.predict_proba(dev.loc[va, cols])[:, 1]
        hold_pred += pipe.predict_proba(hold[cols])[:, 1] / N_FOLDS

    rep = scorecard_report(y, oof, f"{NAME} (OOF)")
    rep_h = scorecard_report(hold[TARGET].to_numpy(), hold_pred, f"{NAME} (holdout)")
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in rep.items()})

    save_predictions(NAME, dev[ID], oof, hold[ID], hold_pred)
    record_result({**rep, "model": NAME, "n_features": len(cols),
                   "holdout_AUC": rep_h["AUC"], "holdout_KS": rep_h["KS"]})


if __name__ == "__main__":
    main()
