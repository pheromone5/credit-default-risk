"""
Stage 4 - the deep learning component.

    monthly panel (B, 24, 6) ---> GRU ---> h_seq (32)
    categoricals   (B, C)    ---> entity embeddings ---> concat
    static numerics(B, N)    ---> standardised
                                     |
                              concat -> 2 dense layers + dropout -> sigmoid(PD)

WHY A RECURRENT ENCODER AND NOT MORE AGGREGATES
-----------------------------------------------
The gradient-boosted model already sees mean/max/std of delinquency. What it
cannot see is order. Two customers with a mean DPD of 8 days are the same row to
LightGBM; one of them has been improving for a year and the other has been
deteriorating. The GRU reads the panel in time order, so shape is available to
it. Whether that shape carries signal is an empirical question - answer it with
the ablation below, do not assert it.

WHY ENTITY EMBEDDINGS
---------------------
One-hot on ORGANIZATION_TYPE gives a wide, sparse block with no notion that two
organisation types might behave alike. A learned dense vector per level lets the
network place similar categories near each other. This is the standard technique
for tabular deep learning and it is worth naming explicitly.

EXPECTED RESULT - READ THIS BEFORE YOU ARE DISAPPOINTED
-------------------------------------------------------
On real tabular credit data this network will most likely score BELOW LightGBM
on its own. That is the well-documented state of play: gradient-boosted trees
dominate tabular tasks. The reason to keep it is in blend_calibrate.py - if its
errors are decorrelated from the tree's, the ensemble beats either component.
Reporting that honestly is a better interview answer than a suspicious win.

    python train_seq_model.py                 # full model
    python train_seq_model.py --no-seq        # ablation: static features only
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from build_sequences import load as load_sequences
from config import ID, N_FOLDS, SEED, TARGET
from data_prep import (load_features, object_columns, record_result,
                       save_predictions, split_frame)
from metrics import scorecard_report

warnings.filterwarnings("ignore", category=UserWarning)

NAME = "gru"
MAX_EPOCHS = 40
PATIENCE = 6
BATCH = 512
LR = 3e-3
HIDDEN = 32

torch.manual_seed(SEED)
np.random.seed(SEED)


# ------------------------------------------------------------------ model
class SequenceRiskNet(nn.Module):
    def __init__(self, n_numeric: int, cat_cardinalities: list[int],
                 n_channels: int, use_seq: bool = True):
        super().__init__()
        self.use_seq = use_seq

        self.embeddings = nn.ModuleList([
            nn.Embedding(card, min(16, (card + 1) // 2)) for card in cat_cardinalities])
        emb_dim = sum(e.embedding_dim for e in self.embeddings)

        seq_dim = 0
        if use_seq:
            self.gru = nn.GRU(n_channels, HIDDEN, num_layers=1, batch_first=True)
            seq_dim = HIDDEN

        self.head = nn.Sequential(
            nn.Linear(n_numeric + emb_dim + seq_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(32, 1),
        )

    def forward(self, x_num, x_cat, x_seq):
        parts = [x_num]
        if self.embeddings:
            parts += [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        if self.use_seq:
            _, h = self.gru(x_seq)
            parts.append(h[-1])
        return self.head(torch.cat(parts, dim=1)).squeeze(1)


# ------------------------------------------------------------------ data
def prepare():
    """Static application features + the monthly tensor, aligned on SK_ID_CURR."""
    df = load_features(app_only=True)
    cat_cols = object_columns(df)
    num_cols = [c for c in df.columns
                if c not in {ID, TARGET, "split", "fold"} and c not in cat_cols]

    codes, cards = [], []
    for c in cat_cols:
        cat = pd.Categorical(df[c].astype("string").fillna("__NA__"))
        codes.append(cat.codes.astype(np.int64))
        cards.append(len(cat.categories))
    X_cat = np.stack(codes, axis=1) if codes else np.zeros((len(df), 0), dtype=np.int64)

    seq_ids, X_seq_all, channels = load_sequences()
    order = pd.Series(np.arange(len(seq_ids)), index=seq_ids).loc[df[ID].to_numpy()].to_numpy()
    X_seq = X_seq_all[order]

    return df, num_cols, cat_cols, cards, X_cat, X_seq, channels


def _standardise(train_block: np.ndarray, *blocks):
    med = np.nanmedian(train_block, axis=0)
    train_filled = np.where(np.isnan(train_block), med, train_block)
    mu, sd = train_filled.mean(0), train_filled.std(0) + 1e-6
    out = []
    for b in blocks:
        f = np.where(np.isnan(b), med, b)
        out.append(np.clip((f - mu) / sd, -6, 6).astype(np.float32))
    return out


def _fit_fold(train_pack, val_pack, cards, n_channels, use_seq):
    Xn, Xc, Xs, y = train_pack
    Vn, Vc, Vs, vy = val_pack

    model = SequenceRiskNet(Xn.shape[1], cards, n_channels, use_seq)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    pos_weight = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ds = torch.utils.data.TensorDataset(
        torch.tensor(Xn), torch.tensor(Xc), torch.tensor(Xs), torch.tensor(y, dtype=torch.float32))
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True)

    v = (torch.tensor(Vn), torch.tensor(Vc), torch.tensor(Vs))
    best_auc, best_state, stale = -1.0, None, 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for bn, bc, bs, by in dl:
            opt.zero_grad()
            loss = loss_fn(model(bn, bc, bs), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(*v)).numpy()
        auc = roc_auc_score(vy, p)
        if auc > best_auc + 1e-4:
            best_auc, stale = auc, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_auc


def main(no_seq: bool = False):
    use_seq = not no_seq
    name = NAME if use_seq else f"{NAME}_noseq"

    df, num_cols, cat_cols, cards, X_cat, X_seq, channels = prepare()
    is_dev = (df["split"] == "dev").to_numpy()
    dev, hold = split_frame(df)

    X_num_all = df[num_cols].to_numpy(dtype=np.float64)
    y_all = df[TARGET].to_numpy()
    folds = df["fold"].to_numpy()

    dev_idx = np.flatnonzero(is_dev)
    hold_idx = np.flatnonzero(~is_dev)

    oof = np.zeros(len(dev_idx))
    hold_pred = np.zeros(len(hold_idx))
    fold_aucs = []

    for k in range(N_FOLDS):
        tr_idx = dev_idx[folds[dev_idx] != k]
        va_idx = dev_idx[folds[dev_idx] == k]

        Xtr, Xva, Xho = _standardise(X_num_all[tr_idx],
                                     X_num_all[tr_idx], X_num_all[va_idx], X_num_all[hold_idx])
        model, auc = _fit_fold(
            (Xtr, X_cat[tr_idx], X_seq[tr_idx], y_all[tr_idx]),
            (Xva, X_cat[va_idx], X_seq[va_idx], y_all[va_idx]),
            cards, X_seq.shape[2], use_seq)
        fold_aucs.append(auc)

        with torch.no_grad():
            pv = torch.sigmoid(model(torch.tensor(Xva), torch.tensor(X_cat[va_idx]),
                                     torch.tensor(X_seq[va_idx]))).numpy()
            ph = torch.sigmoid(model(torch.tensor(Xho), torch.tensor(X_cat[hold_idx]),
                                     torch.tensor(X_seq[hold_idx]))).numpy()
        oof[folds[dev_idx] == k] = pv
        hold_pred += ph / N_FOLDS
        print(f"  fold {k}: val AUC {auc:.4f}")

    rep = scorecard_report(y_all[dev_idx], oof, f"{name} (OOF)")
    rep_h = scorecard_report(y_all[hold_idx], hold_pred, f"{name} (holdout)")
    print(f"\nsequence encoder: {'on' if use_seq else 'OFF (ablation)'}")
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in rep.items()})
    print(f"holdout AUC={rep_h['AUC']:.4f}  KS={rep_h['KS']:.4f}")

    save_predictions(name, dev[ID], oof, hold[ID], hold_pred)
    record_result({**rep, "model": name, "n_features": len(num_cols) + len(cat_cols),
                   "holdout_AUC": rep_h["AUC"], "holdout_KS": rep_h["KS"]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-seq", action="store_true",
                    help="ablation: drop the GRU, keep static features only")
    main(**vars(ap.parse_args()))
