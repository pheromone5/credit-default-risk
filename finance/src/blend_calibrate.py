"""
Stage 5 - blending and calibration.

BLENDING
    Two stackers are compared: a plain rank average, and a logistic regression
    on the three models' out-of-fold predictions. The stacker is itself fitted
    with K-fold on the OOF matrix, so the blended OOF score is honest; the final
    stacker is then refitted on all OOF rows and applied once to the holdout.

CALIBRATION - THE STEP MOST PORTFOLIOS SKIP
    `class_weight="balanced"` and `scale_pos_weight` deliberately distort the
    base rate to help the model learn from a minority class. The consequence is
    that the output is a ranking, not a probability: it will sit far above the
    true 8% default rate. That is fine for a leaderboard and useless for a bank,
    because expected loss is PD x EAD x LGD - the PD has to be a real number.
    Isotonic regression fitted on out-of-fold scores fixes the mapping without
    touching the ranking, so AUC and KS are unchanged while Brier drops sharply.
    Look at the reliability diagram before and after; that plot is the argument.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve

from config import ARTIFACTS, FIGURES, ID, N_FOLDS, SEED, TARGET
from data_prep import load_predictions, record_result
from metrics import decile_table, format_report, scorecard_report
from validation import load_splits

COMPONENTS = ["logistic", "lgbm", "gru"]


def _rank01(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(pct=True).to_numpy()


def gather():
    splits = load_splits()
    dev = splits[splits["split"] == "dev"].set_index(ID)
    hold = splits[splits["split"] == "holdout"].set_index(ID)

    oof, hd = {}, {}
    for name in COMPONENTS:
        d_ids, d_p, h_ids, h_p = load_predictions(name)
        oof[name] = pd.Series(d_p, index=d_ids).loc[dev.index].to_numpy()
        hd[name] = pd.Series(h_p, index=h_ids).loc[hold.index].to_numpy()

    return (dev[TARGET].to_numpy(), pd.DataFrame(oof, index=dev.index),
            hold[TARGET].to_numpy(), pd.DataFrame(hd, index=hold.index))


def correlation_report(oof: pd.DataFrame) -> pd.DataFrame:
    """Blending only pays when the components disagree. Show that they do."""
    return oof.apply(_rank01).corr(method="spearman")


def stack(y_dev, oof: pd.DataFrame, y_hold, hold: pd.DataFrame):
    Xd = oof.apply(_rank01).to_numpy()
    Xh = hold.apply(_rank01).to_numpy()

    blended_oof = np.zeros(len(y_dev))
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for tr, va in skf.split(Xd, y_dev):
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(Xd[tr], y_dev[tr])
        blended_oof[va] = lr.predict_proba(Xd[va])[:, 1]

    final = LogisticRegression(max_iter=1000, C=1.0).fit(Xd, y_dev)
    blended_hold = final.predict_proba(Xh)[:, 1]
    weights = dict(zip(oof.columns, np.round(final.coef_[0], 3)))
    return blended_oof, blended_hold, weights


def calibrate(y_dev, p_dev, p_hold):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_dev, y_dev)
    return iso.transform(p_dev), iso.transform(p_hold), iso


def plot_calibration(y, curves: dict, path):
    """curves: label -> predicted probability vector.

    Include a deliberately miscalibrated model (the weighted LightGBM) so the
    plot shows the problem, not just the fix. A reliability diagram of an
    already-calibrated model is a straight line and teaches nobody anything.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    hi = 0.0
    for lab, p in curves.items():
        frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, "o-", label=lab)
        hi = max(hi, float(mean_pred.max()), float(frac.max()))
    lim = min(1.0, hi * 1.12)
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="perfectly calibrated")
    ax.axhline(y.mean(), color="grey", lw=0.8, ls=":",
               label=f"portfolio default rate {y.mean():.1%}")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("predicted probability of default")
    ax.set_ylabel("observed default rate")
    ax.set_title("Reliability diagram, holdout")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_roc(y, preds: dict, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=name)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC, holdout")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    y_dev, oof, y_hold, hold = gather()

    print("rank correlation between components (low = blending should help):")
    print(correlation_report(oof).round(3).to_string(), "\n")

    rows = [scorecard_report(y_hold, hold[c], c) for c in COMPONENTS]

    rank_avg_hold = hold.apply(_rank01).mean(axis=1).to_numpy()
    rows.append(scorecard_report(y_hold, rank_avg_hold, "rank_average"))

    blended_oof, blended_hold, weights = stack(y_dev, oof, y_hold, hold)
    rows.append(scorecard_report(y_hold, blended_hold, "stacked"))
    print(f"stacker weights: {weights}\n")

    cal_oof, cal_hold, _ = calibrate(y_dev, blended_oof, blended_hold)
    rows.append(scorecard_report(y_hold, cal_hold, "stacked + calibrated"))

    print("HOLDOUT")
    print(format_report(rows))

    # Report the verdict rather than assuming the ensemble won. It often does not.
    best_single = max(COMPONENTS, key=lambda c: scorecard_report(y_hold, hold[c], c)["AUC"])
    best_auc = scorecard_report(y_hold, hold[best_single], best_single)["AUC"]
    stacked_auc = scorecard_report(y_hold, blended_hold, "stacked")["AUC"]
    delta = stacked_auc - best_auc
    verdict = "beats" if delta > 0 else "does NOT beat"
    print(f"\nVERDICT: stacking {verdict} the best single model "
          f"({best_single}, AUC {best_auc:.4f}) by {delta:+.4f} AUC. "
          f"Component rank correlations are "
          f"{correlation_report(oof).to_numpy()[np.triu_indices(len(COMPONENTS), 1)].min():.2f}"
          f"-{correlation_report(oof).to_numpy()[np.triu_indices(len(COMPONENTS), 1)].max():.2f}; "
          "the more correlated the components, the less an ensemble can add.")
    print(f"\nobserved default rate {y_hold.mean():.4f} | "
          f"mean PD before calibration {blended_hold.mean():.4f} | "
          f"after {cal_hold.mean():.4f}")

    plot_calibration(y_hold, {
        "LightGBM, scale_pos_weight (raw)": hold["lgbm"].to_numpy(),
        "GRU, pos_weight (raw)": hold["gru"].to_numpy(),
        "stacked + isotonic": cal_hold,
    }, FIGURES / "calibration.png")
    plot_roc(y_hold, {**{c: hold[c].to_numpy() for c in COMPONENTS},
                      "stacked": blended_hold}, FIGURES / "roc.png")

    np.savez(ARTIFACTS / "pred_final.npz",
             dev_ids=oof.index.to_numpy(), oof=cal_oof,
             hold_ids=hold.index.to_numpy(), hold=cal_hold)

    dt = decile_table(y_hold, cal_hold)
    dt.to_csv(ARTIFACTS / "decile_table.csv", index=False)
    print("\nrisk bands on the holdout (D10 = riskiest decile):")
    print(dt.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    final = scorecard_report(y_hold, cal_hold, "final")
    record_result({**final, "model": "final_calibrated_ensemble",
                   "holdout_AUC": final["AUC"], "holdout_KS": final["KS"]})


if __name__ == "__main__":
    main()
