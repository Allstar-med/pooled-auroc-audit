"""Experiment A: causal multi-stage temporal convolutional network (MS-TCN) on frozen features.

Usage:
  python mstcn.py MANIFEST_CSV FEATURES_NPY OUT_CSV [--seed 1] [--epochs 60] [--every 10] [--smoke]

The network sees each operation as a sequence of frame features in time order. All convolutions are causal
(dilated, padded on the left only), so the output for a frame uses that frame and earlier frames only and can be
run during the operation. Two stages, ten layers per stage, 64 channels, as in MS-TCN with causal padding.
All labels (columns y_*) are trained jointly with class-weighted binary cross-entropy summed over stages.
Checkpoints are taken every --every epochs; each is one configuration, selected per label on the validation
split (sel_E1, sel_W) by oea_io.write_epoch_predictions. The test split is never used for any choice.
"""
import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn

from oea_io import write_epoch_predictions

STAGES, LAYERS, CH, DROP, LR, WD = 2, 10, 64, 0.5, 5e-4, 1e-4


class CausalLayer(torch.nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.d = dilation
        self.conv = torch.nn.Conv1d(ch, ch, 3, dilation=dilation)
        self.out = torch.nn.Conv1d(ch, ch, 1)
        self.drop = torch.nn.Dropout(DROP)

    def forward(self, x):                      # x: (1, ch, T)
        h = Fn.relu(self.conv(Fn.pad(x, (2 * self.d, 0))))
        return x + self.drop(self.out(h))


class Stage(torch.nn.Module):
    def __init__(self, d_in, n_out):
        super().__init__()
        self.inp = torch.nn.Conv1d(d_in, CH, 1)
        self.layers = torch.nn.ModuleList([CausalLayer(CH, 2 ** i) for i in range(LAYERS)])
        self.cls = torch.nn.Conv1d(CH, n_out, 1)

    def forward(self, x):
        h = self.inp(x)
        for l in self.layers:
            h = l(h)
        return self.cls(h)


class MSTCN(torch.nn.Module):
    def __init__(self, d_in, n_out):
        super().__init__()
        self.stages = torch.nn.ModuleList([Stage(d_in, n_out)] + [Stage(n_out, n_out) for _ in range(STAGES - 1)])

    def forward(self, x):
        outs = [self.stages[0](x)]
        for s in self.stages[1:]:
            outs.append(s(torch.sigmoid(outs[-1])))
        return outs


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("features"); ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--every", type=int, default=10); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    seed_all(a.seed)
    m = pd.read_csv(a.manifest)
    F = np.load(a.features).astype(np.float32)
    assert len(F) == len(m), "features and manifest differ in length"
    if a.smoke:
        keep = []
        ys = [c for c in m.columns if c.startswith("y_")]
        pos = m.assign(_p=m[ys].sum(1)).groupby(["split", "unit"])._p.sum()      # videos with most positives
        for s, k in (("train", 4), ("val", 2), ("test", 2)):
            keep += list(pos[s].sort_values(ascending=False).index[:k])
        sel = m.unit.isin(keep).values
        m, F = m[sel].reset_index(drop=True), F[sel]; a.epochs, a.every = 4, 2
    labels = [c for c in m.columns if c.startswith("y_")]
    labels = [l for l in labels if m.loc[m.split == "train", l].nunique() == 2 and m.loc[m.split == "val", l].nunique() == 2]
    tr = (m.split == "train").values
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    F = (F - mu) / sd
    Y = m[labels].values.astype(np.float32)
    # sequences: indices of each unit in time order
    seqs = {u: g.sort_values("t").index.values for u, g in m.groupby("unit")}
    split_of = m.groupby("unit").split.first()
    train_units = [u for u in seqs if split_of[u] == "train"]
    eval_units = [u for u in seqs if split_of[u] != "train"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = MSTCN(F.shape[1], len(labels)).to(dev)
    pos = Y[tr].sum(0); neg = tr.sum() - pos
    pw = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1, 50), device=dev).view(1, -1, 1)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    Ft, Yt = torch.from_numpy(F), torch.from_numpy(Y)
    ev = (m.split != "train").values
    pos_in_ev = np.full(len(m), -1); pos_in_ev[np.where(ev)[0]] = np.arange(ev.sum())
    logits, epochs, t0 = [], [], time.time()
    for ep in range(1, a.epochs + 1):
        net.train(); random.shuffle(train_units); tot = 0.0
        for u in train_units:
            idx = seqs[u]
            x = Ft[idx].T.unsqueeze(0).to(dev); y = Yt[idx].T.unsqueeze(0).to(dev)
            loss = sum(Fn.binary_cross_entropy_with_logits(o, y, pos_weight=pw) for o in net(x))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if ep % a.every == 0:
            net.eval(); L = np.zeros((ev.sum(), len(labels)), np.float32)
            with torch.no_grad():
                for u in eval_units:
                    idx = seqs[u]
                    o = net(Ft[idx].T.unsqueeze(0).to(dev))[-1][0].T.cpu().numpy()
                    L[pos_in_ev[idx]] = o
            assert np.isfinite(L).all()
            logits.append(L); epochs.append(ep)
            print(f"epoch {ep} loss {tot / len(train_units):.4f} {time.time() - t0:.0f}s", flush=True)
    bb = os.path.basename(a.features).rsplit(".", 1)[0].split("__")[-1]
    write_epoch_predictions(m, ev, labels, np.stack(logits), epochs, f"mstcn_{bb}|seed{a.seed}", "mstcn", a.out,
                            dict(manifest=a.manifest, features=a.features, seed=a.seed, epochs=a.epochs, every=a.every,
                                 stages=STAGES, layers=LAYERS, channels=CH, dropout=DROP, lr=LR, weight_decay=WD,
                                 labels=labels, smoke=a.smoke, device=dev, torch=torch.__version__,
                                 seconds=round(time.time() - t0)))


if __name__ == "__main__":
    main()
