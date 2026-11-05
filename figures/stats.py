"""Numbers quoted in the text of the manuscript, computed from ../results (written to out/stats.json)."""
import json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import figs as F
from scipy.stats import spearmanr
import os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
S = {}
Ea = pd.read_csv(f"{R}/all/estimands.csv"); Ea = Ea[Ea.model != "elapsed_time"]; Ea["fam"] = Ea.model.map(F.fam)
Es = pd.read_csv(f"{R}/selected/estimands.csv"); Es = Es[Es.model != "elapsed_time"]
S["n_configs"] = int(len(Ea[["dataset", "model"]].drop_duplicates()))
S["n_configs_ds"] = Ea.groupby("dataset").model.nunique().to_dict()
S["n_evals"] = len(Ea); S["E1_range"] = [round(Ea.E1.min(), 3), round(Ea.E1.max(), 3)]
Ea["gap"] = (Ea.E1 - Ea.W).abs()
S["gap"] = {d: dict(med=round(g.gap.median(), 3), p90=round(g.gap.quantile(.9), 3), max=round(g.gap.max(), 3)) for d, g in Ea.groupby("dataset")}
S["EX_maxdiff"] = round((Ea.E1 - Ea.X).abs().max(), 4)
for z in ("all", "selected"):
    Co = pd.read_csv(f"{R}/{z}/concordance.csv")
    S[f"tau_{z}"] = {d: [round(g.tau_E1_W.min(), 2), round(g.tau_E1_W.max(), 2)] for d, g in Co.groupby("dataset")}
    S[f"samebest_{z}"] = {d: f"{int(g.same_best.sum())}/{len(g)}" for d, g in Co.groupby("dataset")}
pw = F.with_bh(pd.read_csv(f"{R}/all/pairwise.csv"))
S["pairs"] = {d: dict(n=len(g), rev_bh=int(g.rev_bh.sum()), rev_raw=int(g.sig_reversed.sum()),
                      discord=round(g.sig_discordant.mean() * 100, 1)) for d, g in pw.groupby("dataset")}
S["rev_by_label"] = pw[pw.rev_bh == 1].groupby(["dataset", "label"]).size().to_dict()
S["rev_by_label"] = {f"{a}|{b}": int(v) for (a, b), v in S["rev_by_label"].items()}
sel = pd.read_csv(f"{R}/selected/selection.csv")
S["sel_differ"] = {d: f"{int(g.differ.sum())}/{len(g)}" for d, g in sel.groupby("dataset")}
S["sel_dW"] = {d: round((g.test_W_selW - g.test_W_selE1).mean(), 4) for d, g in sel.groupby("dataset")}
tb = pd.read_csv(f"{R}/selected/time_baseline.csv"); tb = tb[tb.estimand == "W"]
out = {}
for (d, l), g in tb.groupby(["dataset", "label"]):
    out[f"{d}|{l}"] = dict(n=len(g), time=round(g.time_value.iloc[0], 3), above=int((g["diff"] > 0).sum()),
                          sig_up=int(((g.q_BH < .05) & (g["diff"] > 0)).sum()), sig_dn=int(((g.q_BH < .05) & (g["diff"] < 0)).sum()),
                          minq=round(g.q_BH.min(), 3))
S["time"] = out
LS = pd.read_csv(f"{R}/label_structure/label_structure.csv"); cvs = LS.dataset.isin(["Endoscapes", "SAGES"])
S["within_rho"] = dict(cvs=round(spearmanr(LS[cvs].between_share, LS[cvs].gap_median).statistic, 2),
                       other=round(spearmanr(LS[~cvs].between_share, LS[~cvs].gap_median).statistic, 2))
S["between_share"] = {d: [round(g.between_share.min(), 2), round(g.between_share.max(), 2)] for d, g in LS.groupby("dataset")}
# causal vs frame heads
json.dump(S, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "stats.json"), "w"), indent=1, default=str)
print(json.dumps(S, indent=1, default=str))
