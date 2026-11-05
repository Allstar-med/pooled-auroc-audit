"""Shared output writer for models trained over epochs (fine-tuned networks and MS-TCN).

Every checkpoint (epoch) is one configuration. For each label, the epoch with the highest validation E1 is
marked sel_E1 and the one with the highest validation W is marked sel_W, exactly as train_heads.py does for
the regularisation grid. The long table has the columns that audit_all.py reads.
"""
import json
import os

import numpy as np
import pandas as pd

from oea_core import estimands_point


def write_epoch_predictions(m, keep, labels, logits, epochs, backbone, head, out, settings):
    """m: manifest; keep: boolean mask of the rows that were scored (validation and test);
    logits: array (n_checkpoints, keep.sum(), n_labels); epochs: list of epoch numbers (one per checkpoint);
    backbone: name that identifies one training run (for example 'ft_resnet50_in1k|seed1')."""
    sub = m[keep].reset_index(drop=True)
    sp = sub.split.values
    va = sp == "val"
    log = []
    for j, lab in enumerate(labels):
        y = sub[lab].values.astype(int)
        if y[va].min() == y[va].max():
            print("skip", lab, "(one class in the validation split)")
            continue
        for k, ep in enumerate(epochs):
            ev = estimands_point(sub.unit.values[va], y[va], logits[k, va, j])
            log.append(dict(backbone=backbone, head=head, label=lab, epoch=ep, val_E1=ev["E1"], val_W=ev["W"]))
    log = pd.DataFrame(log)
    for crit in ("E1", "W"):
        best = log.loc[log.groupby("label")[f"val_{crit}"].idxmax(), ["label", "epoch"]]
        log[f"sel_{crit}"] = log.merge(best.assign(_b=1), on=["label", "epoch"], how="left")._b.fillna(0).astype(int).values
    rows = []
    for _, r in log.iterrows():
        j = labels.index(r.label); k = epochs.index(r.epoch)
        rows.append(pd.DataFrame(dict(
            dataset=sub.dataset.values, split=sp, unit=sub.unit.values, frame_id=sub.frame_id.values, t=sub.t.values,
            label=r.label, y=sub[r.label].values.astype(int), score=logits[k, :, j].astype(np.float32),
            model=f"{backbone}|ep{r.epoch}", backbone=backbone, head=head, C=np.nan,
            sel_E1=int(r.sel_E1), sel_W=int(r.sel_W), sel_head=0)))
    P = pd.concat(rows, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    P.to_csv(out, index=False, float_format="%.6g")
    log.to_csv(out.replace(".csv", "") + "_validation.csv", index=False)
    json.dump(settings, open(out.replace(".csv", "") + "_settings.json", "w"), indent=1)
    print(log[log.sel_E1 == 1].round(4).to_string(index=False))
    return log
