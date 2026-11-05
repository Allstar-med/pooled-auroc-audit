"""Experiment D: does the temporal structure of a label predict whether pooled AUROC misleads?

Usage:
  python label_structure.py OUT_DIR --manifests M1.csv [M2.csv ...] --results RESULTS_DIR [RESULTS_DIR ...]

Label descriptors (test split, no model needed):
  prevalence          share of positive frames
  between_share       share of the label variance that lies between operations:
                      sum_u n_u (p_u - p)^2 / (N p (1 - p)), with p_u the prevalence in operation u
  mixed_ops           share of operations with both classes
  time_tau            mean over mixed operations of Kendall tau between elapsed time and the label
                      (+1: the label switches on once and stays on; 0: no relation to elapsed time)
  switches_per_hour   label changes per hour of operation, averaged over operations
  alpha               share of positive-negative pairs that lie within one operation
Model-side outcomes (from audit_all.py, zoo "all", read from RESULTS_DIR/estimands.csv and concordance.csv):
  gap_median          median over configurations of |E1 - W|
  tau_E1_W            Kendall tau between the rankings by E1 and by W
  beat_time_frac      share of selected configurations whose W exceeds that of elapsed time with q < 0.05
                      (RESULTS_DIR/../selected/time_baseline.csv, if present)
Output: label_structure.csv (one row per dataset and label) and correlations.csv (Spearman correlation of
each descriptor with each outcome across labels, with a bootstrap interval over labels).
The analysis is exploratory and was added after the analysis plan was fixed (PLAN.md, change 3).
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def describe(m):
    out = []
    T = m[m.split == "test"]
    for lab in [c for c in m.columns if c.startswith("y_")]:
        y = T[lab].values.astype(float); p = y.mean()
        if p in (0, 1):
            continue
        g = T.groupby("unit")[lab]
        n_u, p_u = g.size(), g.mean()
        between = float((n_u * (p_u - p) ** 2).sum() / (len(y) * p * (1 - p)))
        taus, sw = [], []
        for u, x in T.sort_values("t").groupby("unit"):
            v = x[lab].values
            dur_h = max((x.t.max() - x.t.min()) / 3600, 1e-9)
            sw.append(np.abs(np.diff(v)).sum() / dur_h)
            if 0 < v.mean() < 1:
                taus.append(kendalltau(x.t.values, v).statistic)
        P = T.groupby("unit")[lab].sum(); N = T.groupby("unit")[lab].apply(lambda s: (s == 0).sum())
        alpha = float((P * N).sum() / (P.sum() * N.sum()))
        out.append(dict(dataset=T.dataset.iloc[0], label=lab, prevalence=p, between_share=between,
                        mixed_ops=float((p_u.between(0, 1, inclusive="neither")).mean()),
                        time_tau=float(np.nanmean(taus)) if taus else np.nan,
                        switches_per_hour=float(np.mean(sw)), alpha=alpha))
    return pd.DataFrame(out)


def outcomes(res_dir):
    E = pd.read_csv(os.path.join(res_dir, "estimands.csv"))
    E = E[E.model != "elapsed_time"].assign(gap=lambda d: (d.E1 - d.W).abs())
    o = E.groupby(["dataset", "label"]).agg(gap_median=("gap", "median"), W_median=("W", "median"),
                                            E1_median=("E1", "median")).reset_index()
    try:
        co = pd.read_csv(os.path.join(res_dir, "concordance.csv"))[["dataset", "label", "tau_E1_W"]]
    except pd.errors.EmptyDataError:   # fewer than three models per label
        co = pd.DataFrame(columns=["dataset", "label", "tau_E1_W"])
    o = o.merge(co, on=["dataset", "label"], how="left")
    tb_path = os.path.join(os.path.dirname(os.path.normpath(res_dir)), "selected", "time_baseline.csv")
    if os.path.exists(tb_path):
        tb = pd.read_csv(tb_path); tb = tb[tb.estimand == "W"]
        b = tb.groupby(["dataset", "label"]).apply(lambda g: float(((g.q_BH < 0.05) & (g["diff"] > 0)).mean()))
        o = o.merge(b.rename("beat_time_frac").reset_index(), on=["dataset", "label"], how="left")
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--results", nargs="+", required=True); ap.add_argument("--B", type=int, default=5000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    Dsc = pd.concat([describe(pd.read_csv(p)) for p in a.manifests], ignore_index=True)
    Out = pd.concat([outcomes(r) for r in a.results], ignore_index=True).drop_duplicates(["dataset", "label"])
    S = Dsc.merge(Out, on=["dataset", "label"], how="inner")
    S.to_csv(os.path.join(a.out, "label_structure.csv"), index=False, float_format="%.6g")
    rng = np.random.default_rng(20261101)
    rows = []
    for d in ("between_share", "time_tau", "switches_per_hour", "prevalence", "mixed_ops"):
        for o in [c for c in ("gap_median", "tau_E1_W", "beat_time_frac", "W_median") if c in S]:
            x = S[[d, o]].dropna()
            if len(x) < 5:
                continue
            r = spearmanr(x[d], x[o]).statistic
            bs = []
            for _ in range(a.B):
                i = rng.integers(0, len(x), len(x))
                if x[d].values[i].std() > 0 and x[o].values[i].std() > 0:
                    bs.append(spearmanr(x[d].values[i], x[o].values[i]).statistic)
            lo, hi = np.nanquantile(bs, [0.025, 0.975])
            rows.append(dict(descriptor=d, outcome=o, labels=len(x), spearman=r, lo=lo, hi=hi))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out, "correlations.csv"), index=False, float_format="%.4g")
    print(S.round(3).to_string(index=False)); print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
