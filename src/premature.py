"""Experiment C: premature declaration of a criterion before it is first met (clinical consequence).

Usage:
  python premature.py OUT_DIR PRED_CSV [PRED_CSV ...] [--dataset Endoscapes] [--sens 0.90] [--B 10000]

For every model and label, the operating threshold is fixed on the validation split so that 90% of the positive
validation frames score at or above it (frame-level sensitivity 0.90, the kind of operating point a deployed
alert would use). On the test split it reports:
  premature_frame_rate   among frames of operations that reach the criterion, the share of frames before the
                         first positive frame that are flagged (the criterion is declared before it is met)
  premature_op_rate      share of those operations with at least one premature flag
  lead_time_median_s     median time from the first premature flag to the true onset, over operations with one
  false_op_rate          share of operations that never reach the criterion but are flagged at least once
  test_sensitivity       share of positive test frames flagged
  first_flag_correct_rate  share of operations reaching the criterion whose first flag falls on a frame where the
                         criterion is met (the first declaration is correct)
  missed_op_rate         share of those operations in which no frame meeting the criterion is flagged
  detect_delay_median_s  median time from the true onset to the first flagged frame meeting the criterion
(per-operation values are written to per_operation.csv.gz)
Two comparisons follow:
  selection.csv     for each backbone, the configuration chosen by validation E1 versus the one chosen by
                    validation W (the P6 pairs): difference in premature_frame_rate with an operation bootstrap
  rank_agreement.csv  across all configurations, Kendall tau between the premature_frame_rate ranking and the
                    rankings by test E1, W and X (from estimands computed here), with bootstrap intervals
The analysis is exploratory and was added after the analysis plan was fixed (PLAN.md, change 3).
"""
import argparse
import os

import numpy as np
import pandas as pd

import oea_core as C

COLS = ["dataset", "split", "unit", "t", "label", "y", "score", "model", "backbone", "sel_E1", "sel_W"]


def threshold(v, sens):
    s = v.score.values[v.y.values == 1]
    return float(np.quantile(s, 1 - sens)) if s.size else np.nan


def per_unit(g, thr):
    """per operation: n pre-onset frames, n flagged pre-onset frames, has_event, any flag, lead time, pos frames, flagged pos"""
    rows = []
    for u, x in g.sort_values("t").groupby("unit"):
        f = x.score.values >= thr; y = x.y.values == 1; t = x.t.values
        if y.any():
            onset = t[y].min(); pre = t < onset
            first = t[pre & f].min() if (pre & f).any() else np.nan
            hit = t[y & f]                                   # flagged frames on which the criterion is met
            rows.append(dict(unit=u, event=1, pre=int(pre.sum()), pre_flag=int((pre & f).sum()), any_pre=int((pre & f).any()),
                             lead=onset - first, pos=int(y.sum()), pos_flag=int((y & f).sum()), flag_any=int(f.any()),
                             first_correct=int(f.any() and y[np.argmax(f)]),   # the first flag of the operation is correct
                             delay=(hit.min() - onset) if hit.size else np.nan, missed=int(hit.size == 0)))
        else:
            rows.append(dict(unit=u, event=0, pre=0, pre_flag=0, any_pre=0, lead=np.nan, pos=0, pos_flag=0,
                             flag_any=int(f.any()), first_correct=0, delay=np.nan, missed=0))
    return pd.DataFrame(rows).set_index("unit")


def summary(U):
    ev = U[U.event == 1]; ne = U[U.event == 0]
    return dict(premature_frame_rate=ev.pre_flag.sum() / max(ev.pre.sum(), 1),
                premature_op_rate=ev.any_pre.mean() if len(ev) else np.nan,
                lead_time_median_s=float(np.nanmedian(ev.lead)) if ev.any_pre.any() else np.nan,
                false_op_rate=ne.flag_any.mean() if len(ne) else np.nan,
                test_sensitivity=ev.pos_flag.sum() / max(ev.pos.sum(), 1), event_ops=len(ev), no_event_ops=len(ne),
                first_flag_correct_rate=ev.first_correct.mean() if len(ev) else np.nan,
                missed_op_rate=ev.missed.mean() if len(ev) else np.nan,
                detect_delay_median_s=float(np.nanmedian(ev.delay)) if ev.delay.notna().any() else np.nan)


def boot_rate(U, c):
    """premature_frame_rate for each bootstrap draw (rows of c are unit multiplicities, columns follow U.index)"""
    return (c @ U.pre_flag.values) / np.maximum(c @ U.pre.values, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("preds", nargs="+")
    ap.add_argument("--dataset", default="Endoscapes"); ap.add_argument("--sens", type=float, default=0.90)
    ap.add_argument("--B", type=int, default=10000); ap.add_argument("--seed", type=int, default=20261101)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    parts = []
    for p in a.preds:
        for ch in pd.read_csv(p, usecols=lambda c: c in COLS, chunksize=2_000_000):
            parts.append(ch[(ch.dataset == a.dataset) & ch.split.isin(["val", "test"])])
    D = pd.concat(parts, ignore_index=True)
    res, sel, rank, perop = [], [], [], []
    for lab, L in D.groupby("label"):
        T = L[L.split == "test"]
        units = np.sort(T.unit.unique()); c = C.draws(len(units), a.B, a.seed)
        U, est = {}, {}
        for m, g in T.groupby("model"):
            v = L[(L.model == m) & (L.split == "val")]
            if v.empty:
                continue
            thr = threshold(v, a.sens)
            U[m] = per_unit(g, thr).reindex(units)
            perop.append(U[m].rename_axis("unit").reset_index().assign(label=lab, model=m, threshold=thr))
            est[m] = C.point(C.prepare(units, g.unit.values, g.y.values, g.score.values))
            res.append(dict(label=lab, model=m, threshold=thr, E1=est[m]["E1"], W=est[m]["W"], X=est[m]["X"], **summary(U[m])))
        # P6 pairs: configuration chosen by validation E1 vs by validation W, per backbone
        for bb, g in L[L.split == "test"].groupby("backbone"):
            mE = g[g.sel_E1 == 1].model.unique(); mW = g[g.sel_W == 1].model.unique()
            if len(mE) != 1 or len(mW) != 1 or mE[0] not in U or mW[0] not in U:
                continue
            rE, rW = boot_rate(U[mE[0]], c), boot_rate(U[mW[0]], c)
            d0 = summary(U[mW[0]])["premature_frame_rate"] - summary(U[mE[0]])["premature_frame_rate"]
            lo, hi = np.nanquantile(rW - rE, [0.025, 0.975])
            sel.append(dict(label=lab, backbone=bb, model_selE1=mE[0], model_selW=mW[0], differ=int(mE[0] != mW[0]),
                            premature_selE1=summary(U[mE[0]])["premature_frame_rate"],
                            premature_selW=summary(U[mW[0]])["premature_frame_rate"], diff_selW_minus_selE1=d0, lo=lo, hi=hi))
        # ranking agreement across all configurations
        ms = [m for m in U]
        if len(ms) >= 3:
            prem = np.array([summary(U[m])["premature_frame_rate"] for m in ms])
            r = dict(label=lab, models=len(ms))
            bprem = np.stack([boot_rate(U[m], c[:2000]) for m in ms], 1)       # draws x models
            for k in ("E1", "W", "X"):
                val = np.array([est[m][k] for m in ms])
                r[f"tau_{k}"] = C.kendall_tau(val, -prem)
            # bootstrap for tau(W) - tau(E1): recompute estimands on the same draws
            R = {m: C.prepare(units, T[T.model == m].unit.values, T[T.model == m].y.values, T[T.model == m].score.values) for m in ms}
            bE = np.stack([C.from_counts(R[m], c[:2000])["E1"] for m in ms], 1)
            bW = np.stack([C.from_counts(R[m], c[:2000])["W"] for m in ms], 1)
            dt = np.array([C.kendall_tau(bW[b], -bprem[b]) - C.kendall_tau(bE[b], -bprem[b]) for b in range(len(bprem))])
            r["tau_W_minus_tau_E1"] = r["tau_W"] - r["tau_E1"]
            r["diff_lo"], r["diff_hi"] = np.nanquantile(dt, [0.025, 0.975])
            rank.append(r)
        print(lab, "models:", len(U))
    pd.DataFrame(res).to_csv(os.path.join(a.out, "premature.csv"), index=False, float_format="%.6g")
    pd.DataFrame(sel).to_csv(os.path.join(a.out, "selection.csv"), index=False, float_format="%.6g")
    pd.DataFrame(rank).to_csv(os.path.join(a.out, "rank_agreement.csv"), index=False, float_format="%.6g")
    pd.concat(perop, ignore_index=True).to_csv(os.path.join(a.out, "per_operation.csv.gz"), index=False, float_format="%.6g")
    print(pd.DataFrame(rank).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
