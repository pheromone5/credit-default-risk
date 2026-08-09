"""
Stage 7 - from a probability to a decision.

This is the part that separates a portfolio project from a leaderboard entry. A
model that ranks well is an input; a bank needs a cut-off, and 0.5 is never the
right one at an 8% base rate.

THE ECONOMICS (assumptions, stated openly - change them in config.py)
    approve a good loan  -> +PROFIT_PER_GOOD_LOAN x principal
    approve a bad loan   -> -LGD x principal
    reject anyone        ->  0

    expected value of approving applicant i
        = (1 - PD_i) * PROFIT * A_i  -  PD_i * LGD * A_i

    which is positive exactly when  PD_i < PROFIT / (PROFIT + LGD).

That closed form is worth knowing: with the defaults it puts the break-even PD
at 0.12 / (0.12 + 0.65) = 15.6%. The script also sweeps every threshold
empirically, because the closed form assumes the PD is calibrated - which is why
Stage 5 exists.

STABILITY
    PSI compares the dev and holdout distributions of the score and of the top
    features. Under a random split it should be near zero; the number only
    becomes interesting when you re-score a later vintage. Wiring the check in
    now is the point - a model nobody monitors is a model nobody can keep.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (ARTIFACTS, DATA_PROC, FIGURES, ID, LGD,
                    PROFIT_PER_GOOD_LOAN, REPORTS, TARGET)
from metrics import population_stability_index
from validation import load_splits


def load_final():
    z = np.load(ARTIFACTS / "pred_final.npz")
    splits = load_splits().set_index(ID)
    hold = pd.DataFrame({"pd": z["hold"]}, index=z["hold_ids"])
    hold[TARGET] = splits.loc[hold.index, TARGET]
    dev = pd.DataFrame({"pd": z["oof"]}, index=z["dev_ids"])
    dev[TARGET] = splits.loc[dev.index, TARGET]

    feats = pd.read_parquet(DATA_PROC / "features.parquet").set_index(ID)
    exposure = feats.loc[hold.index, "AMT_CREDIT"].to_numpy(dtype=float)
    return dev, hold, exposure


def policy_curve(y, pd_hat, exposure, n_points: int = 200) -> pd.DataFrame:
    """Sweep the cut-off; report what the portfolio looks like at each one."""
    order = np.argsort(pd_hat)
    y, pd_hat, exposure = y[order], pd_hat[order], exposure[order]

    thresholds = np.quantile(pd_hat, np.linspace(0.01, 1.0, n_points))
    rows = []
    total_bad_value = float((y * exposure * LGD).sum())

    for t in thresholds:
        approve = pd_hat <= t
        n = int(approve.sum())
        if n == 0:
            continue
        bads = int(y[approve].sum())
        good_val = float(((1 - y[approve]) * exposure[approve]).sum()) * PROFIT_PER_GOOD_LOAN
        bad_val = float((y[approve] * exposure[approve]).sum()) * LGD
        rows.append({
            "threshold": float(t),
            "approval_rate": n / len(y),
            "bad_rate": bads / n,
            "n_approved": n,
            "n_bad_approved": bads,
            "profit": good_val - bad_val,
            "loss_avoided": total_bad_value - bad_val,
        })
    return pd.DataFrame(rows)


def plot_policy(curve: pd.DataFrame, base_rate: float, best: pd.Series, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    ax1.plot(curve["approval_rate"], curve["bad_rate"], lw=2)
    ax1.axhline(base_rate, color="k", ls="--", lw=1,
                label=f"approve everyone: {base_rate:.1%}")
    ax1.scatter([best["approval_rate"]], [best["bad_rate"]], color="crimson", zorder=5,
                label="profit-optimal cut-off")
    ax1.set_xlabel("approval rate")
    ax1.set_ylabel("default rate among approved")
    ax1.set_title("Approval rate vs bad rate")
    ax1.legend()

    ax2.plot(curve["approval_rate"], curve["profit"] / 1e6, lw=2)
    ax2.axvline(best["approval_rate"], color="crimson", ls="--", lw=1)
    ax2.set_xlabel("approval rate")
    ax2.set_ylabel("expected profit (millions)")
    ax2.set_title("Expected profit vs approval rate")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def stability_report(dev, hold) -> pd.DataFrame:
    feats = pd.read_parquet(DATA_PROC / "features.parquet").set_index(ID)
    top = (pd.read_csv(ARTIFACTS / "lgbm_feature_importance.csv")
             .head(10)["feature"].tolist())

    rows = [{"item": "model score",
             "psi": population_stability_index(dev["pd"], hold["pd"])}]
    for f in top:
        if f in feats.columns and pd.api.types.is_numeric_dtype(feats[f]):
            rows.append({"item": f,
                         "psi": population_stability_index(
                             feats.loc[dev.index, f], feats.loc[hold.index, f])})
    out = pd.DataFrame(rows)
    out["flag"] = pd.cut(out["psi"], [-np.inf, 0.10, 0.25, np.inf],
                         labels=["stable", "monitor", "investigate"])
    return out


def main():
    dev, hold, exposure = load_final()
    y = hold[TARGET].to_numpy()
    p = hold["pd"].to_numpy()
    base_rate = y.mean()

    curve = policy_curve(y, p, exposure)
    curve.to_csv(ARTIFACTS / "policy_curve.csv", index=False)
    best = curve.loc[curve["profit"].idxmax()]

    breakeven = PROFIT_PER_GOOD_LOAN / (PROFIT_PER_GOOD_LOAN + LGD)

    lines = []
    lines.append("APPROVAL POLICY (holdout)")
    lines.append(f"  assumptions: margin {PROFIT_PER_GOOD_LOAN:.0%} of principal on a good "
                 f"loan, loss given default {LGD:.0%}")
    lines.append(f"  closed-form break-even PD: {breakeven:.1%}")
    lines.append(f"  empirical profit-optimal cut-off: PD <= {best['threshold']:.3f}")
    lines.append(f"  -> approves {best['approval_rate']:.1%} of applicants "
                 f"({int(best['n_approved']):,} of {len(y):,})")
    lines.append(f"  -> default rate among approved {best['bad_rate']:.2%} "
                 f"vs {base_rate:.2%} if everyone is approved "
                 f"({(1 - best['bad_rate'] / base_rate):.0%} reduction)")
    lines.append(f"  -> expected profit {best['profit']:,.0f} vs "
                 f"{curve['profit'].iloc[-1]:,.0f} under approve-everyone")

    for target_ar in (0.5, 0.7, 0.9):
        row = curve.iloc[(curve["approval_rate"] - target_ar).abs().idxmin()]
        lines.append(f"  at {row['approval_rate']:.0%} approval: bad rate "
                     f"{row['bad_rate']:.2%}, cut-off PD {row['threshold']:.3f}")

    psi = stability_report(dev, hold)
    lines.append("")
    lines.append("STABILITY (dev vs holdout; a random split, so near-zero is expected)")
    for _, r in psi.iterrows():
        lines.append(f"  {r['item']:<32} PSI {r['psi']:.4f}  {r['flag']}")

    text = "\n".join(lines)
    print(text)
    (REPORTS / "policy_summary.txt").write_text(text + "\n")
    psi.to_csv(ARTIFACTS / "psi.csv", index=False)
    plot_policy(curve, base_rate, best, FIGURES / "approval_vs_bad_rate.png")


if __name__ == "__main__":
    main()
