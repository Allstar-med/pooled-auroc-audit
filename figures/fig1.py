"""Fig. 1 (redesigned): what pooled frame-level AUROC compares.
a  schematic of within- and between-operation pairs; b  the real pair matrix of one label (Endoscapes test, C1);
c  within-operation pair share against operations with both classes; d  effective number of operations behind W."""
import os
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figs as F
from figs import MM, COL, DSN, INK, GREY, plabel, save

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
GREEN, RED, POS, NEG = "#2E7D4F", "#B03A2E", "#4A4A4A", "#E6E6E6"


def endoscapes_test(label="C1"):
    """Expand anonymous per-operation counts for the pair-count diagram (no frame identities)."""
    if label != "C1":
        raise ValueError("The public aggregate contains C1 only")
    with open(os.path.join(DATA, "endoscapes_C1_counts.json"), encoding="utf-8") as f:
        aggregate = json.load(f)
    rows = []
    for anonymous_unit, (positive, negative) in enumerate(aggregate["operation_counts"]):
        rows.extend([(anonymous_unit, 1)] * positive)
        rows.extend([(anonymous_unit, 0)] * negative)
    return pd.DataFrame(rows, columns=["vid", "y"])


def panel_a(fig, rect):
    ax = fig.add_axes(rect); ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
    plabel(ax, "a", x=0.0, y=0.97)
    ops = [("Operation 1", 0.52), ("Operation 2", 0.72), ("Operation 3", None)]
    x0, x1, h = 18, 64, 5.2
    ys = [44, 30, 16]
    for (nm, on), y in zip(ops, ys):
        ax.text(x0 - 2, y, nm, ha="right", va="center", fontsize=6)
        ax.add_patch(FancyBboxPatch((x0, y - h / 2), x1 - x0, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=NEG, ec="none"))
        if on is not None:
            xs = x0 + on * (x1 - x0)
            ax.add_patch(FancyBboxPatch((xs, y - h / 2), x1 - xs, h, boxstyle="round,pad=0,rounding_size=1.2",
                                        fc=POS, ec="none"))
            ax.add_patch(Rectangle((xs, y - h / 2), 1.5, h, fc=POS, ec="none"))
    ax.annotate("", xy=(x1 + 1, 8.6), xytext=(x0, 8.6), arrowprops=dict(arrowstyle="-|>", lw=0.5, color=INK, mutation_scale=5))
    ax.text((x0 + x1) / 2, 6.2, "time in operation", ha="center", va="top", fontsize=5.6, color="#555555")
    # within-operation pair (operation 1) and between-operation pairs
    p1 = (x0 + 0.80 * (x1 - x0), ys[0]); n1 = (x0 + 0.22 * (x1 - x0), ys[0])
    ax.annotate("", xy=(n1[0], n1[1] + h / 2 + 0.3), xytext=(p1[0], p1[1] + h / 2 + 0.3),
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.9, connectionstyle="arc3,rad=0.42"))
    for (a, b) in [((0.80, 0), (0.35, 2)), ((0.86, 1), (0.70, 2)), ((0.66, 0), (0.30, 1))]:
        pa = (x0 + a[0] * (x1 - x0), ys[a[1]] - h / 2 - 0.2); pb = (x0 + b[0] * (x1 - x0), ys[b[1]] + h / 2 + 0.2)
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=RED, lw=0.6, ls=(0, (2.2, 1.3)))
    for p in (p1, n1):
        ax.plot(*p, "o", ms=2.2, mfc="white", mec=GREEN, mew=0.6, zorder=5)
    # legend and estimand box
    lx = 66
    ax.add_patch(Rectangle((lx, 51.5), 3.2, 3, fc=POS, ec="none")); ax.text(lx + 4.5, 53, "criterion met", va="center", fontsize=5.6)
    ax.add_patch(Rectangle((lx, 46.5), 3.2, 3, fc=NEG, ec="none")); ax.text(lx + 4.5, 48, "not met", va="center", fontsize=5.6)
    ax.plot([lx, lx + 3.2], [43, 43], color=GREEN, lw=0.9); ax.text(lx + 4.5, 43, "pair within one operation", va="center", fontsize=5.6)
    ax.plot([lx, lx + 3.2], [38, 38], color=RED, lw=0.6, ls=(0, (2.2, 1.3))); ax.text(lx + 4.5, 38, "pair across operations", va="center", fontsize=5.6)
    ax.add_patch(FancyBboxPatch((lx - 1.5, 4), 33, 28.5, boxstyle="round,pad=0,rounding_size=1.5", fc="#F6F6F6", ec="#CFCFCF", lw=0.4))
    ax.text(lx + 14.5, 27.5, "Pooled frame-level AUROC", ha="center", va="center", fontsize=6, fontweight="bold")
    ax.text(lx + 14.5, 21.3, r"$E_1=\alpha\,W+(1-\alpha)\,X$", ha="center", va="center", fontsize=7.5)
    ax.text(lx + 1, 14.6, r"$W$  ranks frames within operations", va="center", fontsize=5.6, color=GREEN)
    ax.text(lx + 1, 10.6, r"$X$  ranks frames across operations", va="center", fontsize=5.6, color=RED)
    ax.text(lx + 1, 6.8, r"$\alpha$  share of pairs within operations", va="center", fontsize=5.6)


def panel_b(fig, rect, label="C1"):
    m = endoscapes_test(label)
    P = m[m.y == 1]; N = m[m.y == 0]
    M = (P.vid.values[:, None] == N.vid.values[None, :]).astype(np.uint8)
    alpha = M.mean()
    ax = fig.add_axes(rect); plabel(ax, "b", x=-0.12, y=1.03)
    ax.imshow(M, cmap=ListedColormap(["#EFEFEF", COL["Endoscapes"]]), aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.4); s.set_color("#999999")
    ax.set_xlabel(f"Frames without the criterion ({len(N):,}), grouped by operation", labelpad=2)
    ax.set_ylabel(f"Frames with the criterion ({len(P):,})", labelpad=2)
    ax.set_title(f"Endoscapes test set, criterion C1: {len(P) * len(N) / 1e3:,.0f} thousand pairs", fontsize=6, pad=3)
    # zoom on one stretch of the diagonal
    r0, r1 = 100, 190
    cols = np.where(M[r0:r1].any(0))[0]
    c0, c1 = max(cols.min() - 60, 0), min(cols.max() + 60, M.shape[1])
    ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, ec=INK, lw=0.5))
    iz = fig.add_axes([rect[0] + rect[2] * 0.05, rect[1] + rect[3] * 0.06, rect[2] * 0.42, rect[3] * 0.46])
    iz.imshow(M[r0:r1, c0:c1], cmap=ListedColormap(["#EFEFEF", COL["Endoscapes"]]), aspect="auto", interpolation="nearest")
    iz.set_xticks([]); iz.set_yticks([])
    for s in iz.spines.values():
        s.set_visible(True); s.set_linewidth(0.5); s.set_color(INK)
    for (fx, fy), (dx, dy) in (((0.05, 0.52), (c0, r1)), ((0.47, 0.52), (c1, r1))):
        ax.annotate("", xy=(rect[0] + rect[2] * fx, rect[1] + rect[3] * fy), xycoords="figure fraction",
                    xytext=(dx, dy), textcoords="data", arrowprops=dict(arrowstyle="-", lw=0.35, color="#777777"))
    ax.text(0.97, 0.95, f"within operations:\n{alpha * 100:.1f}% of pairs", transform=ax.transAxes, fontsize=6,
            color=COL["Endoscapes"], fontweight="bold", va="top", ha="right")
    return alpha


def panel_c(fig, rect):
    A = F.Ea[F.Ea.model != "elapsed_time"].groupby(["dataset", "label"]).agg(alpha=("alpha", "first")).reset_index()
    A = A.merge(F.LS[["dataset", "label", "mixed_ops"]], on=["dataset", "label"])
    ax = fig.add_axes(rect); plabel(ax, "c", x=-0.2, y=1.03)
    for d in F.DS:
        x = A[A.dataset == d]
        ax.scatter(x.alpha * 100, x.mixed_ops * 100, s=10, color=COL[d], ec="white", lw=0.3, zorder=3, label=DSN[d])
    ax.set_xscale("log"); ax.set_xlim(0.1, 10); ax.set_xticks([0.1, 1, 10]); ax.set_xticklabels(["0.1", "1", "10"])
    ax.set_ylim(0, 105)
    ax.set_xlabel("Pairs within operations, $\\alpha$ (%)")
    ax.set_ylabel("Test operations with both classes (%)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.97), fontsize=5.6, handletextpad=0.1, borderaxespad=0.2, ncol=1)
    ax.set_title(f"{len(A)} labels, four datasets", fontsize=6, pad=3, loc="left")


def panel_d(fig, rect):
    E = F.Ea[F.Ea.model != "elapsed_time"].groupby(["dataset", "label"]).neff.first().reset_index()
    nops = {"Endoscapes": 40, "SAGES": 293, "Cholec80": 40, "Cataract101": 31}
    ax = fig.add_axes(rect); plabel(ax, "d", x=-0.2, y=1.03)
    for k, d in enumerate(F.DS):
        x = E[E.dataset == d]
        jit = np.linspace(-0.18, 0.18, len(x)) if len(x) > 1 else [0]
        ax.scatter(x.neff, k + np.asarray(jit), s=9, color=COL[d], ec="white", lw=0.3, zorder=3)
        ax.plot([nops[d]], [k], marker="|", ms=7, mew=0.9, color=INK, zorder=4)
        ax.text(nops[d] * 1.12, k, f"{nops[d]} test {'clips' if d == 'SAGES' else 'operations'}", va="center", fontsize=5.4, color="#444444")
    ax.set_xscale("log"); ax.set_xlim(2, 1500); ax.set_xticks([3, 10, 30, 100, 300, 1000]); ax.set_xticklabels(["3", "10", "30", "100", "300", "1,000"])
    ax.set_yticks(range(len(F.DS))); ax.set_yticklabels([DSN[d] for d in F.DS]); ax.invert_yaxis()
    ax.set_xlabel("Effective number of operations behind $W$, $n_{\\mathrm{eff}}$")


def main():
    fig = plt.figure(figsize=(180 * MM, 118 * MM))
    panel_a(fig, [0.0, 0.53, 0.56, 0.45])
    alpha = panel_b(fig, [0.64, 0.56, 0.35, 0.38])
    panel_c(fig, [0.07, 0.08, 0.33, 0.36])
    panel_d(fig, [0.58, 0.08, 0.36, 0.36])
    save(fig, "Fig1")
    print("alpha C1 Endoscapes", round(alpha, 4))


if __name__ == "__main__":
    main()
