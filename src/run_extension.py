"""Second stage of the pipeline: temporal networks, fine-tuning, further datasets and the audit.

Usage:
  python run_extension.py --work WORK --weights WEIGHTS
         --endoscapes ENDOSCAPES_ROOT --cholec80 CHOLEC80_ROOT
         [--sages SAGES_ROOT] [--cataract101 CATARACT101_ROOT] [--heico HEICO_ROOT]
         [--sages-dedup coarse|refined] [--seeds 1 2 3] [--skip-finetune] [--smoke]

Run run_all.py first: it writes the Endoscapes and Cholec80 manifests, features and linear-probe predictions.
This script reuses everything already in WORK and adds causal MS-TCN on the frozen features of every dataset and
backbone, end-to-end fine-tuning on Endoscapes, Cholec80 and SAGES, Cataract-101 and SAGES with the full pipeline,
the audit of the selected and full zoos, the premature-declaration analysis and the label-structure analysis.
Every step is skipped if its output exists, so the command can be re-run after an interruption. --smoke runs one
tiny fine-tuning and MS-TCN job on Endoscapes and stops. The results are written to WORK/results and packed into
WORK/results.zip; images, features and predictions stay in WORK.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BACKBONES = [("resnet50_in1k", None), ("vitb16_in21k", None), ("dinov2_vitb14_lvd", None),
             ("surgenetxl_dinov2_vitb14", "DINOv2_ViTb14_size336_SurgeNetXL.pth"),
             ("surgenetxl_dinov3_vitb16", "DINOv3_ViTb16_size336_SurgeNetXL.pth")]
FT_BACKBONES = [("resnet50_in1k", None), ("surgenetxl_dinov2_vitb14", "DINOv2_ViTb14_size336_SurgeNetXL.pth")]
DEDUP = "coarse"
STEP = {"HeiCo": 5.0, "Cataract101": 2.0}   # seconds between sampled frames


def sh(*args):
    print(">>", " ".join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=HERE)


def probe(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=r_frame_rate:format=duration", "-of", "json", video],
                         check=True, capture_output=True, text=True).stdout
    j = json.loads(out); num, den = j["streams"][0]["r_frame_rate"].split("/")
    return float(j["format"]["duration"]), float(num) / float(den)


def extract(video, frames_dir, unit, step):
    """one image every `step` seconds: FRAMES_DIR/<unit>_<k>.jpg, k = 0, 1, ... at time k*step"""
    if glob.glob(os.path.join(frames_dir, f"{unit}_0.jpg")):
        return
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", video, "-vf", f"fps=1/{step:g}", "-start_number", "0",
                    "-q:v", "2", os.path.join(frames_dir, f"{unit}_%d.jpg")], check=True)
    print("frames", unit, flush=True)


def prepare_heico(root, W):
    frames = os.path.join(W, "frames", "heico"); os.makedirs(frames, exist_ok=True)
    meta_csv = os.path.join(W, "manifests", "heico_meta.csv")
    rows = []
    for v in sorted(glob.glob(os.path.join(root, "**", "*.avi"), recursive=True)):
        folder = os.path.dirname(v)
        number = os.path.basename(folder)
        kind = re.sub(r"[^a-z]", "", os.path.basename(os.path.dirname(folder)).lower())[:12]
        if not number.isdigit():
            print("skipped (procedure folder is not a number):", v); continue
        csvs = [c for c in glob.glob(os.path.join(folder, "*.csv")) if "phase" in os.path.basename(c).lower()]
        assert len(csvs) == 1, f"expected one phase annotation csv in {folder}, found {csvs}"
        unit = f"heico_{kind}_{int(number):02d}"
        dur, fps = probe(v)
        extract(v, frames, unit, STEP["HeiCo"])
        rows.append(dict(unit=unit, video=v, phase_file=csvs[0], type=kind, number=int(number), duration_s=dur, fps=fps))
    assert len(rows) == 30, f"HeiCo should have 30 procedures, found {len(rows)}"
    pd.DataFrame(rows).to_csv(meta_csv, index=False)
    m = os.path.join(W, "manifests", "heico.csv")
    sh("manifests.py", "heico", root, frames, STEP["HeiCo"], meta_csv, m)
    return m


def prepare_cataract(root, W):
    frames = os.path.join(W, "frames", "cataract101"); os.makedirs(frames, exist_ok=True)
    meta_csv = os.path.join(W, "manifests", "cataract101_meta.csv")
    rows = []
    for v in sorted(glob.glob(os.path.join(root, "**", "*.mp4"), recursive=True)):
        num = re.findall(r"\d+", os.path.splitext(os.path.basename(v))[0])
        if not num:
            continue
        unit = f"cat_{int(num[-1])}"
        dur, fps = probe(v)
        extract(v, frames, unit, STEP["Cataract101"])
        rows.append(dict(unit=unit, video=v, duration_s=dur, fps=fps))
    assert len(rows) == 101, f"Cataract-101 should have 101 videos, found {len(rows)}"
    pd.DataFrame(rows).to_csv(meta_csv, index=False)
    m = os.path.join(W, "manifests", "cataract101.csv")
    sh("manifests.py", "cataract101", root, frames, STEP["Cataract101"], meta_csv, m)
    return m


def dhash64(path):
    """64-bit difference hash and grey-level spread of an image (spread < 8: nearly uniform, not used for matching)"""
    from PIL import Image
    import numpy as np
    g = np.asarray(Image.open(path).convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0]), float(np.asarray(Image.open(path).convert("L")).std())


def near_dup_clips(HA, HB, same=False, thr=6, min_frames=5):
    """clip pairs (a, b) where at least `min_frames` informative frames of clip a have a frame of clip b within
    `thr` bits (64-bit difference hash). HA, HB: dict clip -> list of hashes (uninformative frames removed)."""
    import numpy as np
    pop = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    bk = [(b, h) for b, hs in HB.items() for h in hs]
    if not bk:
        return []
    Bc = np.array([b for b, _ in bk]); Bh = np.array([h for _, h in bk], dtype=np.uint64)
    pairs = []
    for a, hs in HA.items():
        if not hs:
            continue
        x = np.bitwise_xor(np.array(hs, dtype=np.uint64)[:, None], Bh[None, :])
        d = pop[x.view(np.uint8).reshape(x.shape + (8,))].sum(-1)
        close = d <= thr
        for b in np.unique(Bc[close.any(0)]):
            if same and b == a:
                continue
            if close[:, Bc == b].any(1).sum() >= min_frames:
                pairs.append((a, b))
    return pairs


def prepare_sages(root, W):
    """1000 clips: official train (700) and test (300). Writes manifests/sages_meta.csv and the manifest."""
    import hashlib
    frames = os.path.join(W, "frames", "sages"); os.makedirs(frames, exist_ok=True)
    rows = []
    for c in sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)):
        b = os.path.basename(c)
        if b.startswith("._") or b.startswith(".~") or "frame" not in b.lower():
            continue
        folder = os.path.dirname(c)
        unit = os.path.basename(folder) if b.lower() == "frame.csv" else re.sub(r"[_-]?frames?\.csv$", "", b, flags=re.I)
        parts = [p.lower() for p in os.path.normpath(c).split(os.sep)]
        split = "test" if "test" in parts else ("train" if "train" in parts else None)
        vids = [v for v in glob.glob(os.path.join(folder, "*.mp4")) if not os.path.basename(v).startswith("._")]
        if len(vids) != 1:
            vids = [v for v in glob.glob(os.path.join(root, "**", f"{unit}*.mp4"), recursive=True)
                    if not os.path.basename(v).startswith("._")]
        assert split and len(vids) == 1, f"cannot pair {c} with one video and a split ({vids})"
        rows.append(dict(unit=unit, official=split, frame_csv=c, video=vids[0]))
    meta = pd.DataFrame(rows).drop_duplicates("unit")
    n = meta.official.value_counts().to_dict()
    assert n.get("train") == 700 and n.get("test") == 300, f"SAGES should have 700 train and 300 test clips, found {n}"
    for _, r in meta.iterrows():                       # labelled frames only: decode index 0, 150, 300, ... (every 5 s)
        if not os.path.exists(os.path.join(frames, f"{r.unit}_0.jpg")):
            subprocess.run(["ffmpeg", "-loglevel", "error", "-i", r.video, "-vf", "select=not(mod(n\\,150))",
                            "-fps_mode", "passthrough", "-start_number", "0", "-q:v", "2",
                            os.path.join(frames, f"{r.unit}_%d.jpg")], check=True)
    # near-duplicate footage across clips. "coarse" (used in the paper): a training clip is excluded when at least
    # 5 informative frames lie within 6 bits of a frame of a test clip (64-bit difference hash). "refined": the
    # coarse screen confirmed by central-field cross-correlation (sages_dedup.py, Supplementary Note 3).
    if DEDUP == "refined":
        import sages_dedup
        train_dups_of_test, linked, rep = sages_dedup.run(meta, frames, HERE, W)
        print("SAGES duplicate check:", rep, flush=True)
    else:
        H = {}
        for _, r in meta.iterrows():
            hs = []
            for f in glob.glob(os.path.join(frames, f"{r.unit}_*.jpg")):
                h, sd = dhash64(f)
                if sd >= 8:
                    hs.append(h)
            H[r.unit] = hs
        tr = {u: H[u] for u in meta[meta.official == "train"].unit}; te = {u: H[u] for u in meta[meta.official == "test"].unit}
        train_dups_of_test = {a for a, _ in near_dup_clips(tr, te)}
        within_train = near_dup_clips(tr, tr, same=True)
        linked = {a for a, b in within_train} | {b for a, b in within_train}
    prim = pd.read_csv(os.path.join(HERE, "sages_test_primary.csv")).set_index("video_id").primary_include
    meta["include"] = 1; meta["reason"] = ""
    t = meta.official == "test"
    meta.loc[t & (meta.unit.map(prim).fillna(1) == 0), ["include", "reason"]] = [0, "test duplicate (not primary clip)"]
    meta.loc[meta.unit.isin(train_dups_of_test), ["include", "reason"]] = [0, "training clip duplicates a test clip"]
    if DEDUP == "refined" and len(train_dups_of_test) > 40 and not os.path.exists(os.path.join(W, "manifests", "sages_dup_override.csv")):
        raise SystemExit(f"STOP: {len(train_dups_of_test)} training clips confirmed as duplicates of test clips (> 40). "
                         f"Look at the sheets in {os.path.join(W, 'sages_dup_review')} and send the printed line and pairs.csv.")
    # validation: 100 training clips in a fixed order (SHA-256 of the clip id), none sharing footage with another training clip
    cand = meta[(meta.official == "train") & (meta.include == 1) & ~meta.unit.isin(linked)].copy()
    cand["key"] = cand.unit.map(lambda u: hashlib.sha256(u.encode()).hexdigest())
    val = set(cand.sort_values("key").unit[:100])
    meta["split"] = meta.official.where(~meta.unit.isin(val), "val")
    meta_csv = os.path.join(W, "manifests", "sages_meta.csv"); meta.to_csv(meta_csv, index=False)
    print("SAGES clips:", meta.groupby(["split", "include"]).size().to_dict(),
          "| training clips duplicating a test clip:", len(train_dups_of_test),
          "| training clips sharing footage with another training clip:", len(linked))
    m = os.path.join(W, "manifests", "sages.csv")
    sh("manifests.py", "sages", meta_csv, frames, m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--weights", required=True)
    ap.add_argument("--endoscapes"); ap.add_argument("--cholec80"); ap.add_argument("--heico"); ap.add_argument("--cataract101"); ap.add_argument("--sages")
    ap.add_argument("--sages-dedup", choices=["coarse", "refined"], default="coarse")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--skip-finetune", action="store_true"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--exclude", help="Local frame-screen CSV required for the Endoscapes paper analysis")
    a = ap.parse_args()
    if a.endoscapes and not a.smoke and (not a.exclude or not os.path.isfile(a.exclude)):
        ap.error("Endoscapes requires --exclude PATH to the local frame-screen CSV")
    exclusion_args = ["--exclude", os.path.abspath(a.exclude)] if a.exclude else []
    global DEDUP
    DEDUP = a.sages_dedup
    W = os.path.abspath(a.work)
    for d in ("manifests", "features", "preds", "results"):
        os.makedirs(os.path.join(W, d), exist_ok=True)
    wpath = lambda f: os.path.join(a.weights, f) if f else None

    jobs = []   # (dataset, short, manifest)
    if a.endoscapes:
        m = os.path.join(W, "manifests", "endoscapes.csv")
        if not os.path.exists(m):
            sh("manifests.py", "endoscapes", a.endoscapes, m)
        jobs.append(("Endoscapes", "endo", m))
    if a.smoke:
        assert jobs, "--smoke needs --endoscapes"
        ds, short, m = jobs[0]; sd = os.path.join(W, "smoke"); os.makedirs(sd, exist_ok=True)
        npy = os.path.join(W, "features", f"{ds}__resnet50_in1k.npy")
        sh("mstcn.py", m, npy, os.path.join(sd, "mstcn.csv"), "--smoke")
        for bb, wf in FT_BACKBONES:
            extra = ["--weights", wpath(wf)] if wf else []
            sh("finetune.py", m, bb, os.path.join(sd, f"ft_{bb}.csv"), "--smoke", "--batch", "8", *extra)
        print("\nSmoke test passed. Run the same command without --smoke.")
        return
    if a.cholec80:
        m = os.path.join(W, "manifests", "cholec80.csv")
        assert os.path.exists(m), "run run_all.py with --cholec80 first"
        jobs.append(("Cholec80", "c80", m))
    if a.sages:
        m = os.path.join(W, "manifests", "sages.csv")
        if not os.path.exists(m):
            m = prepare_sages(a.sages, W)
        jobs.append(("SAGES", "sages", m))
    if a.heico:
        m = os.path.join(W, "manifests", "heico.csv")
        if not os.path.exists(m):
            m = prepare_heico(a.heico, W)
        jobs.append(("HeiCo", "heico", m))
    if a.cataract101:
        m = os.path.join(W, "manifests", "cataract101.csv")
        if not os.path.exists(m):
            m = prepare_cataract(a.cataract101, W)
        jobs.append(("Cataract101", "cat", m))

    for ds, short, m in jobs:
        for bb, wf in BACKBONES:
            npy = os.path.join(W, "features", f"{ds}__{bb}.npy")
            if not os.path.exists(npy):
                sh("extract_features.py", m, bb, os.path.join(W, "features"), *(["--weights", wpath(wf)] if wf else []))
            pred = os.path.join(W, "preds", f"{short}_{bb}.csv")
            if not os.path.exists(pred):
                sh("train_heads.py", m, npy, pred)
            for s in a.seeds:                                                     # causal MS-TCN
                pred = os.path.join(W, "preds", f"{short}_mstcn_{bb}_s{s}.csv")
                if not os.path.exists(pred):
                    sh("mstcn.py", m, npy, pred, "--seed", s)
        if ds in ("Endoscapes", "Cholec80", "SAGES") and not a.skip_finetune:              # fine-tuning
            for bb, wf in FT_BACKBONES:
                for s in a.seeds:
                    pred = os.path.join(W, "preds", f"{short}_ft_{bb}_s{s}.csv")
                    if not os.path.exists(pred):
                        sh("finetune.py", m, bb, pred, "--seed", s, *(["--weights", wpath(wf)] if wf else []))

    preds = sorted(p for p in glob.glob(os.path.join(W, "preds", "*.csv")) if not p.endswith("_validation.csv"))
    print("prediction files in the audit:", len(preds), flush=True)
    R = os.path.join(W, "results")
    sh("audit_all.py", os.path.join(R, "selected"), *preds, "--B", a.B, *exclusion_args)
    sh("audit_all.py", os.path.join(R, "all"), *preds, "--zoo", "all", "--B", a.B, *exclusion_args)
    for ds, short, out in (("Endoscapes", "endo", "premature"), ("SAGES", "sages", "premature_sages")):
        pp = [p for p in preds if os.path.basename(p).startswith(short + "_")]
        if pp:
            sh("premature.py", os.path.join(R, out), *pp, "--dataset", ds, "--B", a.B)
    sh("label_structure.py", os.path.join(R, "label_structure"), "--manifests",
       *[m for _, _, m in jobs], "--results", os.path.join(R, "all"))

    out = os.path.join(W, "results.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        files = (glob.glob(os.path.join(R, "*", "*.csv")) + glob.glob(os.path.join(R, "*", "*.csv.gz"))
                 + glob.glob(os.path.join(W, "features", "*.json"))
                 + glob.glob(os.path.join(W, "preds", "*_validation.csv")) + glob.glob(os.path.join(W, "preds", "*_settings.json"))
                 + glob.glob(os.path.join(W, "manifests", "*_meta.csv")))
        for f in files:
            z.write(f, os.path.relpath(f, W))
    print("\nDone:", out)


if __name__ == "__main__":
    main()
