"""Audit of a model zoo across datasets and labels (analysis plan: PLAN.md).

Usage:
  python audit_all.py OUT_DIR PRED_CSV [PRED_CSV ...] [--zoo selected|all] [--B 10000] [--seed 20261101]

PRED_CSV files are long tables written by train_heads.py, mstcn.py or finetune.py. Each analysis uses the test
split; the validation split is used only to fix the onset threshold of the timing analysis.

Writes to OUT_DIR:
  estimands.csv        E1, X, W, E2, alpha, n_eff with percentile and t intervals (P1)
  time_baseline.csv    W and E2 of each model minus those of elapsed time, t intervals, BH q (P2)
  concordance.csv      Kendall tau between model rankings by E1 and by W, pairwise reversals (P3)
  pairwise.csv         every model pair: differences in E1 and W with intervals
  timing.csv           onset error of the first positive frame per video (P5)
  timing_rank.csv      agreement of E1 and W rankings with the timing ranking (P5)
  heads.csv            causal head minus frame head of the same backbone (P4)
  selection.csv        test W of the configuration chosen by validation E1 vs by validation W (P6)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

import oea_core as C

KEYS = ("E1", "X", "W", "E2")
TIME = "elapsed_time"
CAP_S = 300.0      # onset error cap (s) when no onset is predicted
WINDOW_S = 30.0    # onset window for the hit rate


def exclude_test_frames(table, excluded):
    """Apply the same dataset/label/frame screen to every test analysis."""
    if excluded is None or excluded.empty:
        return table
    keys = ["dataset", "unit", "frame_id", "label"]
    screen = pd.MultiIndex.from_frame(excluded[keys].drop_duplicates())
    rows = pd.MultiIndex.from_frame(table[keys])
    return table.loc[~rows.isin(screen)].copy()


def validate_predictions(table):
    keys = ["dataset", "split", "unit", "frame_id", "label", "model"]
    if table[keys + ["y", "score", "t"]].isna().any().any():
        raise ValueError("Prediction keys, labels, times and scores must not be missing")
    if table.duplicated(keys).any():
        raise ValueError("Duplicate prediction for the same model and frame")
    if not table.y.isin([0, 1]).all() or not np.isfinite(table[["score", "t"]].to_numpy()).all():
        raise ValueError("Labels must be binary and scores and times must be finite")
    if (table.groupby(keys[:-1], observed=True)[["y", "t"]].nunique() > 1).any().any():
        raise ValueError("Models disagree on a frame's label or time")


def t_interval(est, boot, neff):
    se = np.nanstd(boot, ddof=1)
    q = tdist.ppf(0.975, max(neff - 1, 1))
    p = 2 * tdist.sf(abs(est) / se, max(neff - 1, 1)) if se > 0 else np.nan
    return est - q * se, est + q * se, p


def audit_cell(T, B, seed):
    """T: test rows of one dataset and label, all models (plus the time baseline). Returns tables."""
    units = np.sort(T.unit.unique())
    models = list(pd.unique(T.model))
    c = C.draws(len(units), B, seed)
    R, pt, bt = {}, {}, {}
    for m in models:
        g = T[T.model == m]
        R[m] = C.prepare(units, g.unit.values, g.y.values, g.score.values)
        pt[m] = C.point(R[m]); bt[m] = C.from_counts(R[m], c)
    est = []
    for m in models:
        row = dict(model=m, **pt[m])
        for k in KEYS:
            row[k + "_lo"], row[k + "_hi"] = np.nanquantile(bt[m][k], [0.025, 0.975])
        for k in ("W", "E2"):
            row[k + "_tlo"], row[k + "_thi"], _ = t_interval(pt[m][k], bt[m][k], pt[m]["neff"])
        est.append(row)
    est = pd.DataFrame(est)
    # P2: contrast with elapsed time
    tb = []
    for m in models:
        if m == TIME:
            continue
        for k in ("W", "E2"):
            d = pt[m][k] - pt[TIME][k]
            lo, hi, p = t_interval(d, bt[m][k] - bt[TIME][k], pt[m]["neff"])
            tb.append(dict(model=m, estimand=k, model_value=pt[m][k], time_value=pt[TIME][k], diff=d,
                           lo=lo, hi=hi, p=p))
    tb = pd.DataFrame(tb)
    # P3: ranking concordance among learned models
    zoo = [m for m in models if m != TIME]
    pw = []
    for i in range(len(zoo)):
        for j in range(i + 1, len(zoo)):
            a, b = zoo[i], zoo[j]
            r = dict(model_a=a, model_b=b)
            for k in ("E1", "W"):
                d = bt[a][k] - bt[b][k]
                r[f"d{k}"] = pt[a][k] - pt[b][k]
                r[f"d{k}_lo"], r[f"d{k}_hi"] = np.nanquantile(d, [0.025, 0.975])
                r[f"d{k}_sig"] = int(r[f"d{k}_lo"] > 0 or r[f"d{k}_hi"] < 0)
            r["reversed"] = int(np.sign(r["dE1"]) * np.sign(r["dW"]) < 0)
            r["sig_reversed"] = int(r["reversed"] and r["dE1_sig"] and r["dW_sig"])
            r["sig_discordant"] = int(r["dE1_sig"] != r["dW_sig"])
            pw.append(r)
    pw = pd.DataFrame(pw)
    conc = {}
    if len(zoo) >= 3:
        e1 = np.array([pt[m]["E1"] for m in zoo]); w = np.array([pt[m]["W"] for m in zoo])
        taus = np.array([C.kendall_tau(np.array([bt[m]["E1"][b] for m in zoo]),
                                       np.array([bt[m]["W"][b] for m in zoo])) for b in range(min(B, 2000))])
        conc = dict(models=len(zoo), tau_E1_W=C.kendall_tau(e1, w),
                    tau_lo=np.nanquantile(taus, 0.025), tau_hi=np.nanquantile(taus, 0.975),
                    pairs=len(pw), reversed_frac=pw.reversed.mean(), sig_reversed=int(pw.sig_reversed.sum()),
                    sig_discordant_frac=pw.sig_discordant.mean(),
                    best_E1=zoo[int(np.argmax(e1))], best_W=zoo[int(np.argmax(w))],
                    same_best=int(np.argmax(e1) == np.argmax(w)))
    return est, tb, pw, conc, (units, c, pt)


def onsets(g):
    """first time with y = 1 per unit (NaN if none) and the last time of the unit"""
    g = g.sort_values(["unit", "t"])
    first = g[g.y == 1].groupby("unit").t.min()
    last = g.groupby("unit").t.max()
    return first.reindex(last.index), last


def predicted_onset(g, thr):
    g = g.sort_values(["unit", "t"])
    hit = g[g.score >= thr].groupby("unit").t.min()
    return hit


def onset_errors(g, thr):
    true, last = onsets(g)
    pred = predicted_onset(g, thr).reindex(true.index)
    ev = true.notna()
    err = (pred - true)[ev]
    abs_cap = err.abs().fillna(CAP_S).clip(upper=CAP_S)
    return dict(abs_cap=abs_cap, err=err, false_alarm=pred[~ev].notna(), ev=ev)


def choose_threshold(V):
    qs = np.unique(np.quantile(V.score, np.linspace(0.02, 0.98, 49)))
    obj = [onset_errors(V, q)["abs_cap"].mean() for q in qs]
    return float(qs[int(np.nanargmin(obj))])


def timing(T, V, units, c, models):
    rows, per_unit = [], {}
    for m in models:
        if m == TIME or not (V.model == m).any():
            continue
        thr = choose_threshold(V[V.model == m])
        e = onset_errors(T[T.model == m], thr)
        ac = e["abs_cap"]
        per_unit[m] = ac.reindex(units)
        rows.append(dict(model=m, threshold=thr, event_units=int(e["ev"].sum()),
                         mean_abs_err_capped=ac.mean(), median_abs_err_capped=ac.median(),
                         within_window=float((e["err"].abs() <= WINDOW_S).sum() / e["ev"].sum()),
                         missed=float(e["err"].isna().mean()), early=float((e["err"] < -WINDOW_S).mean()),
                         false_alarm_units=float(e["false_alarm"].mean()) if len(e["false_alarm"]) else np.nan))
    tm = pd.DataFrame(rows)
    if len(tm):
        A = np.vstack([per_unit[m].values for m in tm.model])  # models x units, NaN for units without event
        has = ~np.isnan(A[0])
        cb = c[:, has]; Ah = A[:, has]
        boot = (cb @ Ah.T) / cb.sum(1, keepdims=True)
        tm["mae_lo"], tm["mae_hi"] = np.nanquantile(boot, 0.025, axis=0), np.nanquantile(boot, 0.975, axis=0)
        return tm, boot
    return tm, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("preds", nargs="+")
    ap.add_argument("--zoo", choices=["selected", "all"], default="selected")
    ap.add_argument("--B", type=int, default=10000); ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--exclude", help="CSV with dataset, unit, frame_id, label: test frames left out of that label")
    a = ap.parse_args()
    if a.B < 2:
        ap.error("--B must be at least 2")
    os.makedirs(a.out, exist_ok=True)
    EX = pd.read_csv(a.exclude) if a.exclude else None
    COLS = ["dataset", "split", "unit", "frame_id", "t", "label", "y", "score", "model", "backbone", "head", "C",
            "sel_E1", "sel_W", "sel_head"]
    DT = dict(dataset="category", split="category", unit="category", label="category", model="category",
              backbone="category", head="category", y="int8", score="float32", t="float32", sel_E1="int8", sel_W="int8")

    STR = ("dataset", "split", "unit", "label", "model", "backbone", "head")

    def as_str(d):
        d = d.copy()
        for col in STR:
            d[col] = d[col].astype(str)
        return d

    def load(p, test_only=False):
        """read one prediction file with compact (categorical) types"""
        parts = []
        for ch in pd.read_csv(p, usecols=lambda c: c in COLS, dtype=DT, chunksize=2_000_000):
            if test_only:
                ch = ch[ch.split == "test"]
            elif a.zoo == "selected":
                ch = ch[ch.sel_E1 == 1]
            else:
                # zoo "all": every regularisation value of the linear heads, but only the validation-selected
                # checkpoints (by E1 or by W) of models trained over epochs (MS-TCN, fine-tuning)
                ch = ch[~ch["head"].isin(["mstcn", "finetune"]) | (ch.sel_E1 == 1) | (ch.sel_W == 1)]
            parts.append(ch)
        d = pd.concat(parts, ignore_index=True)
        for col in STR:
            d[col] = d[col].astype(str).astype("category")
        return d

    D = pd.concat([load(p) for p in a.preds], ignore_index=True)
    for col in STR:
        D[col] = D[col].astype(str).astype("category")
    validate_predictions(D)
    E, TB, PW, CO, TM, TR = [], [], [], [], [], []
    for ds, lab in sorted(set(zip(D.dataset.astype(str), D.label.astype(str)))):
        g = as_str(D[(D.dataset == ds) & (D.label == lab)])
        T = g[g.split == "test"]; V = g[g.split == "val"]
        T = exclude_test_frames(T, EX)
        # analyse the frames that every model scored
        k = T.groupby(["unit", "frame_id"]).model.transform("nunique")
        dropped = int((k < T.model.nunique()).sum())
        T = T[k == T.model.nunique()]
        if T.empty:
            raise ValueError(f"{ds} {lab}: no test frames remain shared by all models")
        if dropped:
            print(f"{ds} {lab}: {dropped} prediction rows on frames not scored by every model left out")
        ref = T.drop_duplicates(["unit", "frame_id"])
        # elapsed-time baseline on the frames that every model scored
        T = pd.concat([T, ref.assign(model=TIME, score=ref.t, backbone=TIME, head="none")], ignore_index=True)
        n_models = T.groupby("model").size()
        assert n_models.nunique() == 1, f"{ds} {lab}: models scored different frames: {n_models.to_dict()}"
        est, tb, pw, conc, (units, c, pt) = audit_cell(T, a.B, a.seed)
        tag = dict(dataset=ds, label=lab)
        E.append(est.assign(**tag)); TB.append(tb.assign(**tag)); PW.append(pw.assign(**tag))
        if conc:
            CO.append(dict(**tag, **conc))
        tm, boot = timing(T, V, units, c, list(pd.unique(T.model)))
        if len(tm):
            tm = tm.assign(**tag); TM.append(tm)
            if len(tm) >= 3:
                e1 = np.array([pt[m]["E1"] for m in tm.model]); w = np.array([pt[m]["W"] for m in tm.model])
                mae = tm.mean_abs_err_capped.values
                TR.append(dict(**tag, models=len(tm), tau_E1_timing=C.kendall_tau(e1, -mae),
                               tau_W_timing=C.kendall_tau(w, -mae)))
        print(ds, lab, "models:", T.model.nunique(), "units:", len(units))
    TB = pd.concat(TB, ignore_index=True)
    TB["q_BH"] = np.nan
    for (ds, k), idx in TB.groupby(["dataset", "estimand"]).groups.items():
        TB.loc[idx, "q_BH"] = C.bh(TB.loc[idx, "p"].values)
    out = {"estimands": pd.concat(E), "time_baseline": TB, "pairwise": pd.concat(PW),
           "concordance": pd.DataFrame(CO), "timing": pd.concat(TM) if TM else pd.DataFrame(),
           "timing_rank": pd.DataFrame(TR)}
    # P6: does choosing the configuration by validation W instead of E1 change the test W?
    sel, hd = [], []
    for p in a.preds:
      full = as_str(load(p, test_only=True))
      validate_predictions(full)
      full = exclude_test_frames(full, EX)
      for (ds, lab, bb), g in full.groupby(["dataset", "label", "backbone"]):
        # only backbones with a validation-selected configuration (not the imported fixed models)
        if g[g.sel_E1 == 1].model.nunique() != 1 or g.model.nunique() < 2:
            continue
        mE = g[g.sel_E1 == 1].model.iloc[0]; mW = g[g.sel_W == 1].model.iloc[0]
        units = np.sort(g.unit.unique())
        r = {}
        for nm, m in (("selE1", mE), ("selW", mW)):
            h = g[g.model == m]; r[nm] = C.point(C.prepare(units, h.unit.values, h.y.values, h.score.values))
        sel.append(dict(dataset=ds, label=lab, backbone=bb, model_selE1=mE, model_selW=mW, differ=int(mE != mW),
                        test_W_selE1=r["selE1"]["W"], test_W_selW=r["selW"]["W"],
                        test_E1_selE1=r["selE1"]["E1"], test_E1_selW=r["selW"]["E1"]))
      # P4: causal head minus frame head of the same backbone (best C of each head), matched bootstrap
      if "sel_head" in full.columns:
        F = full[full.sel_head == 1]
        for (ds, lab, bb), g in F.groupby(["dataset", "label", "backbone"]):
            if set(g["head"]) != {"frame", "causal"}:
                continue
            units = np.sort(g.unit.unique()); c = C.draws(len(units), a.B, a.seed)
            R = {h: C.prepare(units, x.unit.values, x.y.values, x.score.values) for h, x in g.groupby("head")}
            pp = {h: C.point(R[h]) for h in R}; b = {h: C.from_counts(R[h], c) for h in R}
            for k in ("E1", "W"):
                d = pp["causal"][k] - pp["frame"][k]
                if k == "W":
                    lo, hi, pv = t_interval(d, b["causal"][k] - b["frame"][k], pp["frame"]["neff"])
                else:
                    lo, hi = np.nanquantile(b["causal"][k] - b["frame"][k], [0.025, 0.975]); pv = np.nan
                hd.append(dict(dataset=ds, label=lab, backbone=bb, estimand=k, frame=pp["frame"][k],
                               causal=pp["causal"][k], diff=d, lo=lo, hi=hi, p=pv))
      del full
    out["selection"] = pd.DataFrame(sel)
    out["heads"] = pd.DataFrame(hd)
    for k, v in out.items():
        v.to_csv(os.path.join(a.out, k + ".csv"), index=False, float_format="%.6g")
    print(out["concordance"].round(3).to_string(index=False))
    print(TB[TB.estimand == "W"][["dataset", "label", "model", "diff", "lo", "hi", "q_BH"]].round(3).to_string(index=False))
    if len(out["timing_rank"]):
        print(out["timing_rank"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
