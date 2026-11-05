"""Near-duplicate footage between SAGES clips: coarse screen, calibrated confirmation and review sheets.

Why two stages: a 64-bit difference hash of a whole laparoscopic frame is dominated by the dark circular border
and the overall layout, so frames of different operations often differ by only a few bits. The screen therefore
only proposes candidate clip pairs. A pair is confirmed by normalised cross-correlation (NCC) of the central
field of view (central 70% of width and height, 48 x 48 grey levels, z-scored): a frame of clip a "matches" clip b
when its best NCC against the frames of b reaches a threshold calibrated on random clip pairs (99.9th percentile
of the best-match NCC of random train-test pairs, which almost all come from different operations). A clip pair is
a duplicate when at least 5 of the informative labelled frames of a match, the rule of the published audit.

Positive controls: the 7 test clips known from the published audit to repeat another test clip (sages_test_primary.csv).
Every confirmed pair is drawn on a review sheet; a pair can be overruled in manifests/sages_dup_override.csv
(columns a, b, decision with decision = keep or remove), after visual review.
"""
import glob
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
N = 48


def frame_features(path):
    im = Image.open(path).convert("L")
    w, h = im.size
    g = np.asarray(im, dtype=np.float32)
    spread = float(g.std())
    d = np.asarray(im.resize((9, 8), Image.BILINEAR), dtype=np.int16)
    h64 = int(np.packbits((d[:, 1:] > d[:, :-1]).flatten()).view(">u8")[0])
    c = im.crop((int(0.15 * w), int(0.15 * h), int(0.85 * w), int(0.85 * h))).resize((N, N), Image.BILINEAR)
    v = np.asarray(c, dtype=np.float32).ravel()
    v = (v - v.mean()) / (v.std() + 1e-6) / np.sqrt(v.size)
    return h64, v, spread


def load(frames_dir, units):
    F = {}
    for u in units:
        files = sorted(glob.glob(os.path.join(frames_dir, f"{u}_*.jpg")), key=lambda f: int(f.rsplit("_", 1)[1][:-4]))
        rows = [(f,) + frame_features(f) for f in files]
        rows = [r for r in rows if r[3] >= 8]                      # drop nearly uniform frames
        F[u] = dict(files=[r[0] for r in rows], h=np.array([r[1] for r in rows], dtype=np.uint64),
                    v=np.stack([r[2] for r in rows]) if rows else np.zeros((0, N * N), np.float32))
    return F


def screen(F, A, B, thr=6, min_frames=5, same=False):
    """candidate pairs from the 64-bit hash (generous on purpose)"""
    bk = [(b, h) for b in B for h in F[b]["h"]]
    if not bk:
        return []
    Bc = np.array([b for b, _ in bk]); Bh = np.array([h for _, h in bk], dtype=np.uint64)
    out = []
    for a in A:
        if not len(F[a]["h"]):
            continue
        x = np.bitwise_xor(F[a]["h"][:, None], Bh[None, :])
        close = POP[x.view(np.uint8).reshape(x.shape + (8,))].sum(-1) <= thr
        for b in np.unique(Bc[close.any(0)]):
            if same and b <= a:          # each unordered pair once
                continue
            if close[:, Bc == b].any(1).sum() >= min_frames:
                out.append((a, b))
    return out


def best_ncc(F, a, b):
    if not len(F[a]["v"]) or not len(F[b]["v"]):
        return np.zeros(0), np.zeros(0, int)
    S = F[a]["v"] @ F[b]["v"].T
    return S.max(1), S.argmax(1)


def calibrate(F, A, B, n_pairs=3000, seed=20261103):
    rng = np.random.default_rng(seed); vals = []
    A, B = list(A), list(B)
    for _ in range(n_pairs):
        a, b = A[rng.integers(len(A))], B[rng.integers(len(B))]
        if a != b:
            vals.append(best_ncc(F, a, b)[0])
    vals = np.concatenate(vals)
    return float(np.quantile(vals, 0.999)), vals


def confirm(F, pairs, tau, min_frames=5):
    rows = []
    for a, b in pairs:
        s, _ = best_ncc(F, a, b)
        rows.append(dict(a=a, b=b, frames=len(s), matched=int((s >= tau).sum()), median_ncc=float(np.median(s)) if len(s) else np.nan,
                         duplicate=int((s >= tau).sum() >= min_frames)))
    return pd.DataFrame(rows, columns=["a", "b", "frames", "matched", "median_ncc", "duplicate"])


def sheet(F, a, b, path, k=4):
    s, j = best_ncc(F, a, b)
    order = np.argsort(-s)[:k]
    tiles = []
    for i in order:
        A = Image.open(F[a]["files"][i]).convert("RGB").resize((320, 180))
        B = Image.open(F[b]["files"][j[i]]).convert("RGB").resize((320, 180))
        t = Image.new("RGB", (650, 200), "white"); t.paste(A, (0, 20)); t.paste(B, (330, 20))
        ImageDraw.Draw(t).text((4, 4), f"{a[:12]} vs {b[:12]}  NCC {s[i]:.3f}", fill="black")
        tiles.append(t)
    out = Image.new("RGB", (650, 200 * len(tiles)), "white")
    for i, t in enumerate(tiles):
        out.paste(t, (0, 200 * i))
    out.save(path, quality=85)


def run(meta, frames_dir, here, work):
    """meta: one row per clip (unit, official). Returns (train clips duplicating a test clip, linked training clips, report)"""
    tr = list(meta[meta.official == "train"].unit); te = list(meta[meta.official == "test"].unit)
    F = load(frames_dir, tr + te)
    tau, null = calibrate(F, tr, te)
    prim = pd.read_csv(os.path.join(here, "sages_test_primary.csv"))
    groups = prim.groupby("footage_group_id").video_id.apply(list)
    controls = [tuple(g) for g in groups if len(g) == 2 and all(u in F for u in g)]
    ctrl = confirm(F, [(b, a) for a, b in controls], tau)          # non-primary clip vs primary clip
    tt = confirm(F, screen(F, tr, te), tau)
    ww = confirm(F, screen(F, tr, tr, same=True), tau)
    ov_path = os.path.join(work, "manifests", "sages_dup_override.csv")
    if os.path.exists(ov_path):
        ov = pd.read_csv(ov_path)
        for _, r in ov.iterrows():
            for D in (tt, ww):
                hit = (D.a == r.a) & (D.b == r.b)
                D.loc[hit, "duplicate"] = int(str(r.decision).strip().lower() == "remove")
    rev = os.path.join(work, "sages_dup_review"); os.makedirs(rev, exist_ok=True)
    for f in glob.glob(os.path.join(rev, "*.jpg")):
        os.remove(f)
    for _, r in tt[tt.duplicate == 1].iterrows():
        sheet(F, r.a, r.b, os.path.join(rev, f"train_test_{r.a[:8]}_{r.b[:8]}.jpg"))
    for _, r in ctrl.iterrows():
        sheet(F, r.a, r.b, os.path.join(rev, f"control_{r.a[:8]}_{r.b[:8]}.jpg"))
    # borderline pairs (3-4 matched frames) for a look at what the rule lets through
    for _, r in tt[(tt.duplicate == 0) & (tt.matched >= 3)].head(10).iterrows():
        sheet(F, r.a, r.b, os.path.join(rev, f"borderline_{r.a[:8]}_{r.b[:8]}.jpg"))
    pd.concat([tt.assign(kind="train-test"), ww.assign(kind="train-train"), ctrl.assign(kind="control")]) \
        .to_csv(os.path.join(rev, "pairs.csv"), index=False)
    null_rate = float((null >= tau).mean())
    report = dict(tau=round(tau, 4), screened_train_test=len(tt), confirmed_train_test=int(tt.duplicate.sum()),
                  train_clips_removed=int(tt[tt.duplicate == 1].a.nunique()), screened_train_train=len(ww),
                  confirmed_train_train=int(ww.duplicate.sum()), controls_detected=f"{int(ctrl.duplicate.sum())}/{len(ctrl)}",
                  null_frame_match_rate=round(null_rate, 5))
    dup = set(tt[tt.duplicate == 1].a)
    linked = set(ww[ww.duplicate == 1].a) | set(ww[ww.duplicate == 1].b)
    return dup, linked, report
