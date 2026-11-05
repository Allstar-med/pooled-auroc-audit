"""Linear heads on frozen features: one per-frame head and one causal temporal head per label.

Usage:
  python train_heads.py MANIFEST_CSV FEATURES_NPY OUT_CSV [--labels y_C1,y_C2] [--ema 0.8]

For each label and each value of the regularisation grid, a logistic regression is fitted on the training
split and scored on the validation and test splits. The per-frame head sees the features of the frame only.
The causal temporal head also sees an exponential moving average of the features of the earlier frames of the
same video (no future frames), so it can be run during an operation.

The output is a long table with one row per frame, label and configuration (columns dataset, split, unit,
frame_id, t, label, y, score, model, backbone, head, C, sel_E1, sel_W, sel_head). sel_E1 marks the configuration that a
conventional pipeline would select (highest pooled validation AUROC); sel_W marks the one with the highest
within-video validation AUROC; sel_head marks the best C within each head. All choices are fixed before the test split is looked at.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from oea_core import estimands_point

# Extended after the first run: validation E1 rose monotonically as C fell for every backbone, head and label,
# so the optimum lay below the original grid (1e-3 ... 1). See PLAN.md, change 1.
C_GRID = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2, 1e-1, 1.0)


def causal_ema(F, units, order, lam):
    """EMA over the earlier frames of each unit (rows in `order` are sorted by unit, then time).
    The value for a frame excludes that frame; the first frame of a unit gets its own features."""
    out = np.empty_like(F)
    prev_u, acc = None, None
    for i in order:
        u = units[i]
        if u != prev_u:
            acc = F[i].copy(); prev_u = u
        out[i] = acc
        acc = lam * acc + (1 - lam) * F[i]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("features"); ap.add_argument("out")
    ap.add_argument("--labels"); ap.add_argument("--ema", type=float, default=0.8)
    a = ap.parse_args()
    m = pd.read_csv(a.manifest)
    F = np.load(a.features).astype(np.float32)
    assert F.shape[0] == len(m), "features and manifest differ in length"
    backbone = os.path.basename(a.features).rsplit(".", 1)[0].split("__")[-1]
    labels = a.labels.split(",") if a.labels else [c for c in m.columns if c.startswith("y_")]
    order = np.lexsort((m.t.values, m.unit.values))
    H = causal_ema(F, m.unit.values, order, a.ema)
    inputs = {"frame": F, "causal": np.hstack([F, H])}
    sp = m.split.values
    tr, va = sp == "train", sp == "val"
    keep = sp != "train"
    out, log = [], []
    for head, X in inputs.items():
        sc = StandardScaler().fit(X[tr])
        Z = sc.transform(X).astype(np.float32)
        for lab in labels:
            y = m[lab].values.astype(int)
            if y[tr].min() == y[tr].max():
                print("skip", lab, "(one class in training split)"); continue
            for C in C_GRID:
                clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
                clf.fit(Z[tr], y[tr])
                s = clf.decision_function(Z)
                ev = estimands_point(m.unit.values[va], y[va], s[va])
                log.append(dict(backbone=backbone, head=head, label=lab, C=C, val_E1=ev["E1"], val_W=ev["W"]))
                out.append(pd.DataFrame(dict(dataset=m.dataset.values[keep], split=sp[keep], unit=m.unit.values[keep],
                                             frame_id=m.frame_id.values[keep], t=m.t.values[keep], label=lab,
                                             y=y[keep], score=s[keep].astype(np.float32),
                                             model=f"{backbone}|{head}|C={C:g}", backbone=backbone, head=head, C=C)))
    log = pd.DataFrame(log)
    # selection is made across C and head, per label, on the validation split only
    for crit in ("E1", "W"):
        best = log.loc[log.groupby("label")[f"val_{crit}"].idxmax(), ["label", "head", "C"]]
        log[f"sel_{crit}"] = log.merge(best.assign(_b=1), on=["label", "head", "C"], how="left")._b.fillna(0).astype(int).values
    # best C within each head (by validation E1), for the frame-vs-causal comparison
    best = log.loc[log.groupby(["label", "head"]).val_E1.idxmax(), ["label", "head", "C"]]
    log["sel_head"] = log.merge(best.assign(_b=1), on=["label", "head", "C"], how="left")._b.fillna(0).astype(int).values
    P = pd.concat(out, ignore_index=True)
    P = P.merge(log[["label", "head", "C", "sel_E1", "sel_W", "sel_head"]], on=["label", "head", "C"], how="left")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    P.to_csv(a.out, index=False, float_format="%.6g")
    log.to_csv(a.out.replace(".csv", "") + "_validation.csv", index=False)
    json.dump(dict(manifest=a.manifest, features=a.features, ema=a.ema, C_grid=C_GRID, labels=labels),
              open(a.out.replace(".csv", "") + "_settings.json", "w"), indent=1)
    print(log.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
