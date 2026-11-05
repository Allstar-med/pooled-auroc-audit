"""Figures 2-6 and Extended Data Fig. 1 (180 mm wide, Arial-metric font). Figure 1 is drawn by fig1.py.

Usage: python figs.py   (reads ../results, writes ./out)"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
MM = 1 / 25.4
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size": 6.5, "axes.labelsize": 6.5, "axes.titlesize": 6.5,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    "legend.frameon": False, "axes.titleweight": "normal", "lines.linewidth": 0.8, "savefig.dpi": 600})
DS = ["Endoscapes", "SAGES", "Cholec80", "Cataract101"]
DSN = {"Endoscapes": "Endoscapes", "SAGES": "SAGES", "Cholec80": "Cholec80", "Cataract101": "Cataract-101"}
COL = {"Endoscapes": "#C44E52", "SAGES": "#8172B3", "Cholec80": "#4C72B0", "Cataract101": "#55A868"}
FAM_M = {"linear": "o", "mstcn": "s", "finetune": "^"}
FAM_N = {"linear": "Linear probe", "mstcn": "Causal MS-TCN", "finetune": "Fine-tuned"}
GREY, INK = "#9A9A9A", "#222222"


def fam(s):
    if s == "elapsed_time": return "time"
    if s.startswith("ft_"): return "finetune"
    if s.startswith("mstcn_"): return "mstcn"
    return "linear"


def lab(l):
    l = l[2:]
    return {"CVS": "CVS", "C1": "C1", "C2": "C2", "C3": "C3"}.get(l, l)


BBN = {"resnet50_in1k": "ResNet-50", "vitb16_in21k": "ViT-B/16", "dinov2_vitb14_lvd": "DINOv2",
       "surgenetxl_dinov2_vitb14": "SurgeNetXL-DINOv2", "surgenetxl_dinov3_vitb16": "SurgeNetXL-DINOv3"}


def pretty(m):
    p = m.split("|")
    if p[0].startswith("mstcn_"):
        return f"{BBN[p[0][6:]]} + causal MS-TCN"
    if p[0].startswith("ft_"):
        return f"{BBN[p[0][3:]]}, fine-tuned"
    return f"{BBN[p[0]]}, {'causal' if p[1] == 'causal' else 'frame'} linear probe"


SHORT = {"CalotTriangleDissection": "Calot", "GallbladderDissection": "GB dissect.", "GallbladderPackaging": "GB pack.",
         "GallbladderRetraction": "GB retract.", "CleaningCoagulation": "Clean/coag.", "ClippingCutting": "Clip/cut",
         "SpecimenBag": "Spec. bag", "ViscousAgentInjection": "Visc. inj.", "ViscousAgentRemoval": "Visc. rem.",
         "IrrigationAspiration": "Irr./asp.", "Phacoemulsification": "Phaco", "TonifyingAntibiotics": "Tonif./abx",
         "CapsulePolishing": "Caps. pol.", "Hydrodissection": "Hydrodiss.", "LensImplant": "Lens impl.", "Preparation": "Prep."}


def short(l):
    l = l[2:]
    return SHORT.get(l, l)


def tau_strip(ax, df):
    xs = 0; ticks = []; names = []; centers = []
    for d in DS:
        c = df[df.dataset == d].sort_values("tau_E1_W")
        start = xs
        for _, r in c.iterrows():
            ax.plot([xs, xs], [r.tau_lo, r.tau_hi], color=COL[d], lw=0.7)
            ax.scatter([xs], [r.tau_E1_W], s=8, color=COL[d], zorder=3, ec="white", lw=0.3)
            ticks.append(xs); names.append(short(r.label)); xs += 1
        centers.append(((start + xs - 1) / 2, d)); xs += 1.2
    ax.set_xticks(ticks); ax.set_xticklabels(names, rotation=90, fontsize=4.8); ax.tick_params(axis="x", length=1.5, pad=1)
    for cx, d in centers:
        ax.text(cx, 1.02, DSN[d], ha="center", va="bottom", fontsize=6, color=COL[d], transform=ax.get_xaxis_transform())
    ax.axhline(0.8, color=GREY, lw=0.5, ls="--")


def plabel(ax, s, x=-0.16, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


Ea = pd.read_csv(f"{R}/all/estimands.csv"); Ea["fam"] = Ea.model.map(fam)
Es = pd.read_csv(f"{R}/selected/estimands.csv"); Es["fam"] = Es.model.map(fam)
CA = pd.read_csv(f"{R}/all/concordance.csv")
LS = pd.read_csv(f"{R}/label_structure/label_structure.csv")
TB = pd.read_csv(f"{R}/selected/time_baseline.csv"); TB = TB[TB.estimand == "W"]; TB["fam"] = TB.model.map(fam)


# ------------------------------------------------------------------ Fig. 2
def fig2():
    fig = plt.figure(figsize=(180 * MM, 100 * MM))
    m = Ea[Ea.fam != "time"]
    for k, d in enumerate(DS):
        ax = fig.add_axes([0.055 + k * 0.245, 0.56, 0.19, 0.36]); plabel(ax, "abcd"[k], x=-0.25)
        x = m[m.dataset == d]
        lo = 0.4 if d in ("Endoscapes", "SAGES") else 0.6
        ax.plot([0, 1], [0, 1], color=GREY, lw=0.5, ls="--", zorder=1)
        for f, mk in FAM_M.items():
            y = x[x.fam == f]
            ax.scatter(y.E1, y.W, s=3.5, marker=mk, color=COL[d], alpha=0.45, lw=0, zorder=2)
        ax.set_xlim(lo, 1); ax.set_ylim(lo, 1); ax.set_aspect("equal")
        ax.set_title(f"{DSN[d]} ({x.label.nunique()} labels)", fontsize=6.5, pad=2)
        ax.set_xlabel("Pooled AUROC, $E_1$")
        if k == 0: ax.set_ylabel("Within-operation AUROC, $W$")
    hs = [Line2D([], [], marker=mk, ls="", color="#555555", ms=3.5, label=FAM_N[f]) for f, mk in FAM_M.items()]
    fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, handletextpad=0.2, columnspacing=1.2)
    # e: tau per label with CI
    axe = fig.add_axes([0.055, 0.12, 0.42, 0.29]); plabel(axe, "e", x=-0.11, y=1.1)
    tau_strip(axe, CA)
    axe.set_ylim(0, 1); axe.set_ylabel("Agreement of model rankings,\nKendall $\\tau$($E_1$, $W$)")
    # f: |E1-W|
    axf = fig.add_axes([0.58, 0.12, 0.40, 0.29]); plabel(axf, "f", x=-0.13, y=1.1)
    for k, d in enumerate(DS):
        g = (m[m.dataset == d].E1 - m[m.dataset == d].W).abs().values
        vp = axf.violinplot([g], positions=[k], widths=0.7, showextrema=False)
        for b in vp["bodies"]: b.set_facecolor(COL[d]); b.set_alpha(0.55); b.set_edgecolor("none")
        q = np.quantile(g, [0.25, 0.5, 0.75]); axf.plot([k, k], [q[0], q[2]], color=INK, lw=1.2); axf.scatter([k], [q[1]], s=7, color="white", ec=INK, lw=0.6, zorder=3)
    axf.set_xticks(range(4)); axf.set_xticklabels([DSN[d] for d in DS]); axf.tick_params(axis="x", length=0)
    axf.set_ylabel("$|E_1-W|$ per configuration"); axf.set_ylim(0, 0.21)
    save(fig, "Fig2")


# ------------------------------------------------------------------ Fig. 3
def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); q = np.empty(n); prev = 1.0
    for k in range(n - 1, -1, -1):
        prev = min(prev, p[o[k]] * n / (k + 1)); q[o[k]] = prev
    return q


def with_bh(pw):
    from scipy.stats import norm
    out = []
    for _, g in pw.groupby(["dataset", "label"]):
        g = g.copy()
        for k in ("E1", "W"):
            se = (g[f"d{k}_hi"] - g[f"d{k}_lo"]) / 3.92
            p = 2 * norm.sf(np.abs(np.where(se > 0, g[f"d{k}"] / se, 0)))
            g[f"q{k}"] = bh(p)
        g["rev_bh"] = ((np.sign(g.dE1) * np.sign(g.dW) < 0) & (g.qE1 < 0.05) & (g.qW < 0.05)).astype(int)
        out.append(g)
    return pd.concat(out)


def fig3():
    pw = with_bh(pd.read_csv(f"{R}/all/pairwise.csv"))
    fig = plt.figure(figsize=(180 * MM, 62 * MM))
    s = pw[(pw.dataset == "SAGES") & (pw.label == "y_C1")]
    ax = fig.add_axes([0.05, 0.16, 0.25, 0.72]); plabel(ax, "a", x=-0.2)
    ax.scatter(s.dE1, s.dW, s=1.2, color="#C8C8C8", lw=0, rasterized=True)
    sig = s[s.rev_bh == 1]
    ax.scatter(sig.dE1, sig.dW, s=3.5, color=COL["SAGES"], lw=0, zorder=3)
    ax.axhline(0, color=INK, lw=0.4); ax.axvline(0, color=INK, lw=0.4)
    ax.set_xlabel("Difference in pooled AUROC, $\\Delta E_1$"); ax.set_ylabel("Difference in within-operation AUROC, $\\Delta W$")
    ax.set_title(f"SAGES, C1: {len(s):,} configuration pairs", fontsize=6.5, pad=2)
    ax.text(0.97, 0.04, f"{len(sig)} pairs reversed\n(FDR-controlled)", transform=ax.transAxes, ha="right", fontsize=5.6, color="#5A4A90")
    r = sig.assign(sc=sig.dE1 * (-sig.dW)).sort_values("sc", ascending=False).iloc[0]
    axb = fig.add_axes([0.38, 0.16, 0.17, 0.72]); plabel(axb, "b", x=-0.32)
    for j, k in enumerate(("E1", "W")):
        axb.plot([j, j], [r[f"d{k}_lo"], r[f"d{k}_hi"]], color=COL["SAGES"], lw=1)
        axb.scatter([j], [r[f"d{k}"]], s=14, color=COL["SAGES"], zorder=3)
    axb.axhline(0, color=INK, lw=0.4)
    axb.set_xticks([0, 1]); axb.set_xticklabels(["$\\Delta E_1$", "$\\Delta W$"]); axb.set_xlim(-0.6, 1.6)
    axb.set_ylabel("Difference, A minus B (95% CI)")
    axb.text(0.5, 1.02, "A: " + pretty(r.model_a) + "\nB: " + pretty(r.model_b), transform=axb.transAxes, ha="center", va="bottom", fontsize=5.0)
    axc = fig.add_axes([0.64, 0.16, 0.15, 0.72]); plabel(axc, "c", x=-0.42)
    for k, d in enumerate(DS):
        g = pw[pw.dataset == d]; v0 = g.sig_reversed.sum() / len(g) * 1e4; v1 = g.rev_bh.sum() / len(g) * 1e4
        axc.bar(k, v0, color=COL[d], alpha=0.3, width=0.7); axc.bar(k, v1, color=COL[d], width=0.7)
        axc.text(k, v0 + 1, f"{int(g.rev_bh.sum())}", ha="center", fontsize=5.5)
    axc.set_xticks(range(4)); axc.set_xticklabels([DSN[d] for d in DS], rotation=45, ha="right"); axc.set_ylabel("Reversed pairs\nper 10,000 configuration pairs")
    sel = pd.read_csv(f"{R}/selected/selection.csv")
    axd = fig.add_axes([0.865, 0.16, 0.13, 0.72]); plabel(axd, "d", x=-0.5)
    for k, d in enumerate(DS):
        g = sel[sel.dataset == d]; v = g.differ.mean() * 100
        axd.bar(k, v, color=COL[d], width=0.7); axd.text(k, v + 1.5, f"{int(g.differ.sum())}/{len(g)}", ha="center", fontsize=5.0)
    axd.set_xticks(range(4)); axd.set_xticklabels([DSN[d] for d in DS], rotation=45, ha="right"); axd.set_ylim(0, 80)
    axd.set_ylabel("Training runs in which selection\nby $E_1$ and by $W$ disagree (%)")
    save(fig, "Fig3"); return r


# ------------------------------------------------------------------ Fig. 5 (elapsed time)
def fig5():
    fig = plt.figure(figsize=(180 * MM, 66 * MM))
    ax = fig.add_axes([0.06, 0.2, 0.56, 0.7]); plabel(ax, "a", x=-0.08)
    labs = [("Endoscapes", l) for l in ("y_C1", "y_C2", "y_C3", "y_CVS")] + [("SAGES", l) for l in ("y_C1", "y_C2", "y_C3", "y_CVS")]
    rng = np.random.default_rng(1)
    for k, (d, l) in enumerate(labs):
        g = TB[(TB.dataset == d) & (TB.label == l)]
        x = k + rng.uniform(-0.22, 0.22, len(g))
        up = (g.q_BH < 0.05) & (g["diff"] > 0); dn = (g.q_BH < 0.05) & (g["diff"] < 0)
        ax.scatter(x[~(up | dn)], g.model_value[~(up | dn)], s=5, color=COL[d], alpha=0.35, lw=0)
        ax.scatter(x[up], g.model_value[up], s=6, color=COL[d], lw=0)
        ax.scatter(x[dn], g.model_value[dn], s=6, facecolor="white", ec=COL[d], lw=0.5)
        ax.plot([k - 0.33, k + 0.33], [g.time_value.iloc[0]] * 2, color=INK, lw=1.3)
    ax.set_xticks(range(8)); ax.set_xticklabels([lab(l) for _, l in labs]); ax.tick_params(axis="x", length=0)
    ax.text(1.5, 0.985, "Endoscapes (40 operations)", ha="center", va="top", fontsize=6); ax.text(5.5, 0.985, "SAGES (293 clips)", ha="center", va="top", fontsize=6)
    ax.axvline(3.5, color=GREY, lw=0.4); ax.set_ylim(0.45, 1.0); ax.set_ylabel("Within-operation AUROC, $W$")
    ax.legend(handles=[Line2D([], [], color=INK, lw=1.3, label="Elapsed time"),
                       Line2D([], [], marker="o", ls="", color="#777777", ms=3, label="Model, higher than time ($q$<0.05)"),
                       Line2D([], [], marker="o", ls="", mfc="white", mec="#777777", ms=3, label="Model, lower than time ($q$<0.05)"),
                       Line2D([], [], marker="o", ls="", color="#777777", alpha=0.35, ms=3, label="No significant difference")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=5.5, handletextpad=0.2, columnspacing=1.0)
    # b: all labels, fraction beating time vs time W
    axb = fig.add_axes([0.71, 0.2, 0.27, 0.7]); plabel(axb, "b", x=-0.22)
    for (d, l), g in TB.groupby(["dataset", "label"]):
        f = ((g.q_BH < 0.05) & (g["diff"] > 0)).mean() * 100
        tw = g.time_value.iloc[0]
        if tw >= 0.5:
            axb.scatter(tw, f, s=12, color=COL[d], ec="white", lw=0.3, zorder=3)
        else:
            axb.scatter(tw, f, s=12, facecolor="white", ec=COL[d], lw=0.7, zorder=3)
    axb.set_xlabel("Within-operation AUROC of elapsed time"); axb.set_ylabel("Selected models better than time (%)")
    axb.set_xlim(-0.03, 1.03); axb.set_ylim(-4, 104); axb.axvline(0.5, color=GREY, lw=0.4, ls=":")
    axb.legend(handles=[Line2D([], [], marker="o", ls="", color=COL[d], ms=3.5, label=DSN[d]) for d in DS], loc="lower left", fontsize=5.5, handletextpad=0.1)
    save(fig, "Fig5")


# ------------------------------------------------------------------ Fig. 4 (label structure)
def fig4():
    C = pd.read_csv(f"{R}/label_structure/correlations.csv")
    fig = plt.figure(figsize=(180 * MM, 62 * MM))
    pairs = [("between_share", "gap_median", "Between-operation share of label variance", "Median $|E_1-W|$", "a"),
             ("mixed_ops", "tau_E1_W", "Operations containing both classes (fraction)", "Kendall $\\tau$($E_1$, $W$)", "b")]
    for k, (xd, yd, xl, yl, p) in enumerate(pairs):
        ax = fig.add_axes([0.055 + k * 0.29, 0.18, 0.22, 0.72]); plabel(ax, p, x=-0.25)
        for d in DS:
            x = LS[LS.dataset == d]
            ax.scatter(x[xd], x[yd], s=12, color=COL[d], ec="white", lw=0.3, zorder=3, label=DSN[d])
        r = C[(C.descriptor == xd) & (C.outcome == yd)].iloc[0]
        ax.text(0.04 if k == 0 else 0.96, 0.95 if k == 0 else 0.04, f"Spearman $\\rho$ = {r.spearman:.2f}\n(95% CI {r.lo:.2f} to {r.hi:.2f})",
                transform=ax.transAxes, fontsize=5.6, va="top" if k == 0 else "bottom", ha="left" if k == 0 else "right")
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        if k == 0: ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.55), fontsize=5.5, handletextpad=0.1)
    # c: forest
    ax = fig.add_axes([0.73, 0.18, 0.25, 0.72]); plabel(ax, "c", x=-0.42)
    dn = {"between_share": "Between-operation share", "mixed_ops": "Operations with both classes", "switches_per_hour": "Label changes per hour",
          "time_tau": "Monotonic in time", "prevalence": "Prevalence"}
    on = {"gap_median": ("|E$_1$−W|", INK), "tau_E1_W": ("τ(E$_1$,W)", "#8C6BB1")}
    y = 0
    yt = []
    for d in dn:
        for o, (oname, oc) in on.items():
            r = C[(C.descriptor == d) & (C.outcome == o)].iloc[0]
            ax.plot([r.lo, r.hi], [y, y], color=oc, lw=0.8); ax.scatter([r.spearman], [y], s=9, color=oc, zorder=3, marker="o" if o == "gap_median" else "s")
            y += 1
        yt.append(y - 1.5); y += 0.8
    ax.axvline(0, color=GREY, lw=0.5); ax.set_yticks(yt); ax.set_yticklabels(list(dn.values())); ax.invert_yaxis()
    ax.set_xlim(-1, 1); ax.set_xlabel("Spearman correlation across 32 labels")
    ax.legend(handles=[Line2D([], [], marker="o", color=INK, lw=0.8, ms=3, label="with median |E$_1$−W|"),
                       Line2D([], [], marker="s", color="#8C6BB1", lw=0.8, ms=3, label="with τ(E$_1$, W)")],
              loc="upper center", bbox_to_anchor=(0.45, 1.16), fontsize=5.5, ncol=2, handletextpad=0.2, columnspacing=0.6)
    save(fig, "Fig4")


# ------------------------------------------------------------------ Fig. 6
def fig6():
    fig = plt.figure(figsize=(180 * MM, 60 * MM))
    panels = (("premature_op_rate", 100, "Operations reaching the criterion with\na declaration before it was met (%)", "a", (0, 105)),
              ("lead_time_median_s", 1, "Median time from first premature\ndeclaration to the criterion (s)", "b", (0, 170)),
              ("false_op_rate", 100, "Operations never reaching the criterion\nwith at least one declaration (%)", "c", (0, 105)))
    for k, (key, sc, ttl, p, yl) in enumerate(panels):
        ax = fig.add_axes([0.07 + k * 0.33, 0.14, 0.25, 0.74]); plabel(ax, p, x=-0.24)
        for j, (d, f) in enumerate((("Endoscapes", "premature"), ("SAGES", "premature_sages"))):
            Pm = pd.read_csv(f"{R}/{f}/premature.csv")
            for i, l in enumerate(("y_C1", "y_C2", "y_C3", "y_CVS")):
                v = Pm[Pm.label == l][key].dropna().values * sc; x = i + (j - 0.5) * 0.36
                vp = ax.violinplot([v], positions=[x], widths=0.32, showextrema=False)
                for b in vp["bodies"]: b.set_facecolor(COL[d]); b.set_alpha(0.6); b.set_edgecolor("none")
                ax.scatter([x], [np.median(v)], s=6, color="white", ec=INK, lw=0.5, zorder=3)
        ax.set_xticks(range(4)); ax.set_xticklabels(["C1", "C2", "C3", "CVS"]); ax.tick_params(axis="x", length=0)
        ax.set_ylim(*yl); ax.set_ylabel(ttl)
        if k == 0:
            ax.legend(handles=[Rectangle((0, 0), 1, 1, fc=COL[d], alpha=0.6, label=DSN[d] + " (242 configurations)") for d in ("Endoscapes", "SAGES")],
                      loc="lower left", fontsize=5.5)
    save(fig, "Fig6")




def edfig1():
    cs = pd.read_csv(f"{R}/selected/concordance.csv")
    fig, ax = plt.subplots(figsize=(150 * MM, 52 * MM))
    tau_strip(ax, cs)
    ax.set_ylim(-0.1, 1); ax.set_ylabel("Kendall $\\tau$($E_1$, $W$),\nselected configurations")
    save(fig, "EDFig1")


if __name__ == "__main__":
    fig2(); r = fig3(); fig4(); fig5(); fig6(); edfig1()
    print("example reversal:", r.model_a, r.model_b, round(r.dE1, 3), round(r.dW, 3))
