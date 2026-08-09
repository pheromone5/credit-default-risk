"""
Stage 6 - explainability, framed as a compliance requirement rather than a plot.

In most ML settings interpretability is a nice-to-have. In lending it is closer
to mandatory: a declined applicant is generally entitled to be told the
principal reasons, and model risk governance expects the model owner to be able
to explain any individual decision, not just aggregate importance. A black-box
scorecard is not merely unsatisfying, it is difficult to put into production.

This script produces three things:
  1. a global SHAP summary - which features move the score across the book
  2. per-applicant waterfall plots - why THIS decision
  3. adverse_action_reasons.csv - the top drivers pushing each declined
     applicant's PD up, which is the raw material for a decline letter

Gain-based importance (already written by train_lgbm.py) answers "what did the
trees split on". SHAP answers "which direction and how much for this person".
They are different questions and interviewers do ask which you used and why.
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from config import ARTIFACTS, FIGURES, ID, SEED, TARGET
from data_prep import (as_categorical, feature_columns, load_features,
                       object_columns, split_frame)
from train_lgbm import PARAMS

warnings.filterwarnings("ignore")
N_EXPLAIN = 1500          # holdout sample used for the global plot
N_REASONS = 3             # reason codes per declined applicant


def fit_full_model(dev, cols):
    params = dict(PARAMS)
    y = dev[TARGET].to_numpy()
    params["scale_pos_weight"] = (y == 0).sum() / max((y == 1).sum(), 1)
    params["n_estimators"] = 400          # no early stopping set here; fixed budget
    model = lgb.LGBMClassifier(**params)
    model.fit(dev[cols], y)
    return model


def main():
    df = load_features()
    df = as_categorical(df, object_columns(df))
    dev, hold = split_frame(df)
    cols = feature_columns(dev)

    model = fit_full_model(dev, cols)
    explainer = shap.TreeExplainer(model)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(hold), size=min(N_EXPLAIN, len(hold)), replace=False)
    sample = hold.iloc[idx].reset_index(drop=True)
    X = sample[cols]

    sv = explainer.shap_values(X)
    if isinstance(sv, list):              # older shap returns one array per class
        sv = sv[1]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shap.summary_plot(sv, X, max_display=18, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_summary.png", dpi=140, bbox_inches="tight")
    plt.close()

    # ---- individual explanations: the riskiest and the safest applicant
    scores = model.predict_proba(X)[:, 1]
    picks = {"highest_risk": int(np.argmax(scores)), "lowest_risk": int(np.argmin(scores))}
    for label, i in picks.items():
        expl = shap.Explanation(values=sv[i], base_values=explainer.expected_value,
                                data=X.iloc[i], feature_names=cols)
        shap.plots.waterfall(expl, max_display=12, show=False)
        plt.title(f"{label}: PD={scores[i]:.3f}, actual TARGET={sample[TARGET].iloc[i]}")
        plt.tight_layout()
        plt.savefig(FIGURES / f"shap_waterfall_{label}.png", dpi=140, bbox_inches="tight")
        plt.close()

    # ---- adverse action reason codes for the riskiest decile
    cutoff = np.quantile(scores, 0.9)
    declined = np.flatnonzero(scores >= cutoff)
    rows = []
    for i in declined:
        contrib = pd.Series(sv[i], index=cols).sort_values(ascending=False)
        top = contrib.head(N_REASONS)
        rows.append({
            ID: sample[ID].iloc[i],
            "PD": round(float(scores[i]), 4),
            **{f"reason_{r + 1}": f"{f} = {sample[f].iloc[i]} "
                                  f"(+{contrib[f]:.3f} to log-odds)"
               for r, f in enumerate(top.index)},
        })
    reasons = pd.DataFrame(rows)
    reasons.to_csv(ARTIFACTS / "adverse_action_reasons.csv", index=False)

    mean_abs = (pd.DataFrame({"feature": cols, "mean_abs_shap": np.abs(sv).mean(0)})
                .sort_values("mean_abs_shap", ascending=False))
    mean_abs.to_csv(ARTIFACTS / "shap_importance.csv", index=False)

    print("figures ->", FIGURES)
    print(f"reason codes for {len(reasons)} declined applicants -> "
          f"{ARTIFACTS / 'adverse_action_reasons.csv'}")
    print("\ntop 15 features by mean |SHAP|:")
    print(mean_abs.head(15).to_string(index=False))
    print("\nexample decline explanation:")
    if len(reasons):
        print(reasons.iloc[0].to_string())


if __name__ == "__main__":
    main()
