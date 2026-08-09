"""
Metrics, in the vocabulary a credit risk team actually uses.

AUC is the headline, but a bank asks three further questions:
  * KS   - how far apart are the good and bad score distributions?
  * Gini - the same information, rescaled (Gini = 2*AUC - 1)
  * Brier / calibration - is the score a usable probability of default, or only
    a ranking? Expected loss is PD x EAD x LGD, so PD has to mean something.
PSI answers a fourth: has the population moved since the model was fitted?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score, roc_curve)


def ks_statistic(y_true, y_score) -> float:
    """Max separation between the cumulative good and bad distributions."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def gini(y_true, y_score) -> float:
    return float(2 * roc_auc_score(y_true, y_score) - 1)


def population_stability_index(expected, actual, n_bins: int = 10) -> float:
    """PSI between two distributions of a score or feature.

    Rule of thumb used across retail credit:
      < 0.10  stable
      0.10-0.25  monitor
      > 0.25  investigate / recalibrate
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    edges = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    eps = 1e-6
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def scorecard_report(y_true, y_score, label: str = "") -> dict:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    return {
        "model": label,
        "AUC": float(roc_auc_score(y_true, y_score)),
        "Gini": gini(y_true, y_score),
        "KS": ks_statistic(y_true, y_score),
        "PR_AUC": float(average_precision_score(y_true, y_score)),
        "Brier": float(brier_score_loss(y_true, y_score)),
    }


def decile_table(y_true, y_score, n_bins: int = 10) -> pd.DataFrame:
    """Risk-band table: the artefact a credit officer will actually read."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_score, dtype=float)})
    df["band"] = pd.qcut(df["p"].rank(method="first"), n_bins,
                         labels=[f"D{i}" for i in range(1, n_bins + 1)])
    out = (df.groupby("band", observed=True)
             .agg(n=("y", "size"), bads=("y", "sum"),
                  min_score=("p", "min"), max_score=("p", "max"),
                  mean_score=("p", "mean"))
             .reset_index())
    out["bad_rate"] = out["bads"] / out["n"]
    out["lift"] = out["bad_rate"] / df["y"].mean()
    return out.iloc[::-1].reset_index(drop=True)


def format_report(rows: list[dict]) -> str:
    df = pd.DataFrame(rows)
    cols = ["model", "AUC", "Gini", "KS", "PR_AUC", "Brier"]
    df = df[cols]
    return df.to_string(index=False, float_format=lambda v: f"{v:.4f}")
