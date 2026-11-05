"""Frame manifests: one row per labelled frame with split, unit, time and labels.

Usage:
  python manifests.py endoscapes ENDOSCAPES_ROOT OUT_CSV
  python manifests.py cholec80 CHOLEC80_ROOT FRAMES_DIR OUT_CSV
  python manifests.py check MANIFEST_CSV

Every manifest has the columns
  dataset, split, unit, frame_id, t, image_path, and one 0/1 column per label (prefix "y_").
t is the time of the frame in seconds from the first labelled frame of its unit.
A manifest for SAGES 2024 (or any other dataset) can be written by hand in the same format
and checked with `python manifests.py check`.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

REQUIRED = ["dataset", "split", "unit", "frame_id", "t", "image_path"]


def endoscapes(root):
    """Endoscapes-CVS201 keyframes (one every 5 s). Frame indices are at 25 per second
    (frames are released at 1 fps with indices stepping by 25; README of the release)."""
    m = pd.read_csv(os.path.join(root, "all_metadata.csv"))
    m = m[m.is_ds_keyframe == True].copy()  # noqa: E712
    split_of = {}
    for split, fn in (("train", "train_vids.txt"), ("val", "val_vids.txt"), ("test", "test_vids.txt")):
        for v in open(os.path.join(root, fn)).read().split():
            split_of[int(float(v))] = split
    m["split"] = m.vid.map(split_of)
    assert m.split.notna().all(), "video without split"
    m["unit"] = "endo_" + m.vid.astype(str)
    m["frame_id"] = m.frame.astype(int)
    m["t"] = (m.frame - m.groupby("vid").frame.transform("min")) / 25.0
    m["image_path"] = [os.path.join(root, s, f"{v}_{f}.jpg") for s, v, f in zip(m.split, m.vid, m.frame)]
    for c in ("C1", "C2", "C3"):
        m[f"y_{c}"] = (m[c] >= 0.5).astype(int)  # majority of three annotators
    m["y_CVS"] = m.y_C1 * m.y_C2 * m.y_C3  # all three criteria met
    m["dataset"] = "Endoscapes"
    return m[REQUIRED + ["y_C1", "y_C2", "y_C3", "y_CVS"]].sort_values(["unit", "t"]).reset_index(drop=True)


STEP_S = 5  # seconds between sampled Cholec80 frames (Endoscapes keyframe spacing)
CHOLEC80_TOOLS = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]
CHOLEC80_PHASES = ["Preparation", "CalotTriangleDissection", "ClippingCutting", "GallbladderDissection",
                   "GallbladderPackaging", "CleaningCoagulation", "GallbladderRetraction"]


def cholec80(root, frames_dir):
    """Cholec80 tool presence and phase, one frame every 5 s (every 125th frame of the 25-fps videos),
    the keyframe spacing of Endoscapes. Standard split: videos 1-32 train, 33-40 validation, 41-80 test.
    frames_dir must hold one image every 5 s named videoXX_<k>.jpg, where k = 0, 1, 2, ... is the index of the
    5-s step (time = 5k s), as written by
      ffmpeg -i videoXX.mp4 -vf fps=1/5 -start_number 0 FRAMES_DIR/videoXX_%d.jpg"""
    rows, dropped = [], []
    for v in range(1, 81):
        tool = pd.read_csv(os.path.join(root, "tool_annotations", f"video{v:02d}-tool.txt"), sep="\t")
        phase = pd.read_csv(os.path.join(root, "phase_annotations", f"video{v:02d}-phase.txt"), sep="\t")
        phase = phase.set_index("Frame").Phase
        tool = tool[tool.Frame % (25 * STEP_S) == 0]
        split = "train" if v <= 32 else ("val" if v <= 40 else "test")
        for _, r in tool.iterrows():
            f = int(r.Frame)
            sec = f // 25
            k_img = sec // STEP_S
            cand = glob.glob(os.path.join(frames_dir, f"video{v:02d}_{k_img}.*"))
            if not cand:   # 5-s point at the very end of the video for which ffmpeg wrote no image (29 in total)
                dropped.append((v, f))
                continue
            row = dict(dataset="Cholec80", split=split, unit=f"c80_{v:02d}", frame_id=f, t=float(sec),
                       image_path=cand[0] if cand else os.path.join(frames_dir, f"video{v:02d}_{k_img}.jpg"))
            for k in CHOLEC80_TOOLS:
                row[f"y_{k}"] = int(r[k])
            ph = phase.get(f, None)
            for k in CHOLEC80_PHASES:
                row[f"y_{k}"] = int(ph == k)
            rows.append(row)
    print(f"Cholec80: {len(rows)} frames; dropped {len(dropped)} annotated 5-s points without an image")
    assert len(dropped) < 100, "many images missing: check FRAMES_DIR"
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- experiment B: other procedures
HEICO_PHASES = {0: "GeneralPreparation", 1: "LymphNodeVesselDissection", 2: "RetroperitonealPancreas",
                3: "RetroperitonealDuodenum", 4: "MobilisationSigmoidDescending", 5: "MobilisationSplenicFlexure",
                6: "MobilisationTransverse", 7: "MobilisationAscending", 8: "RectumDissection",
                9: "ExtraabdominalAnastomosis", 10: "IntraabdominalAnastomosis", 11: "Stoma", 12: "Finalisation",
                13: "Exception"}
CATARACT_PHASES = {1: "Incision", 2: "ViscousAgentInjection", 3: "Rhexis", 4: "Hydrodissection",
                   5: "Phacoemulsification", 6: "IrrigationAspiration", 7: "CapsulePolishing", 8: "LensImplant",
                   9: "ViscousAgentRemoval", 10: "TonifyingAntibiotics"}


def _phase_at(starts, phases, t):
    """phase in force at times t, given phase start times (sorted); NaN before the first start"""
    i = np.searchsorted(starts, t, side="right") - 1
    out = np.where(i >= 0, np.asarray(phases)[np.clip(i, 0, None)], -1)
    return out


def _frames_of(frames_dir, unit, step):
    """images written by run_extension.extract(): FRAMES_DIR/<unit>_<k>.jpg at time k*step seconds"""
    files = glob.glob(os.path.join(frames_dir, f"{unit}_*.jpg"))
    ks = sorted(int(os.path.basename(f)[len(unit) + 1:-4]) for f in files)
    return np.array(ks), np.array(ks) * float(step)


def heico(root, frames_dir, step, meta_csv):
    """Heidelberg colorectal (HeiCo): 30 procedures (10 proctocolectomy, 10 rectal resection, 10 sigmoid resection).
    Phase files give phase ids and their start times; the time unit (frames, s or ms) is detected per file from the
    video duration and fps in meta_csv (written by run_extension.py) and printed for checking.
    Split, fixed in advance: within each procedure type, procedures 1-5 train, 6 validation, 7-10 test."""
    meta = pd.read_csv(meta_csv).set_index("unit")
    rows = []
    for unit in sorted(meta.index):
        r = meta.loc[unit]
        ph = pd.read_csv(r.phase_file, sep=None, engine="python")
        num = ph.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        tcol, pcol = num.columns[0], num.columns[1]
        start = num[tcol].values.astype(float); phase = num[pcol].values.astype(int)
        o = np.argsort(start); start, phase = start[o], phase[o]
        if start.max() <= r.duration_s * 1.05:
            unit_t, scale = "s", 1.0
        elif start.max() <= r.duration_s * r.fps * 1.05:
            unit_t, scale = "frames", 1.0 / r.fps
        else:
            unit_t, scale = "ms", 1e-3
        start = start * scale
        print(f"{unit}: phase times read as {unit_t}; first starts {np.round(start[:3], 1).tolist()} s; duration {r.duration_s:.0f} s")
        ks, t = _frames_of(frames_dir, unit, step)
        cur = _phase_at(start, phase, t)
        n = int(r.number)
        split = "train" if n <= 5 else ("val" if n == 6 else "test")
        for k, tt, c in zip(ks, t, cur):
            row = dict(dataset="HeiCo", split=split, unit=unit, frame_id=int(k), t=float(tt),
                       image_path=os.path.join(frames_dir, f"{unit}_{k}.jpg"))
            for pid, name in HEICO_PHASES.items():
                row[f"y_{name}"] = int(c == pid)
            rows.append(row)
    m = pd.DataFrame(rows)
    return m


def cataract101(root, frames_dir, step, meta_csv):
    """Cataract-101: 101 cataract operations, 10 phases. annotations.csv lists, per video, the frame at which each
    phase starts (VideoID, FrameNo, Phase); videos.csv gives the frame rate. Split, fixed in advance by sorted
    video id: first 60 train, next 10 validation, last 31 test."""
    ann = pd.read_csv(os.path.join(root, "annotations.csv"), sep=None, engine="python")
    vids = pd.read_csv(os.path.join(root, "videos.csv"), sep=None, engine="python").set_index("VideoID")
    meta = pd.read_csv(meta_csv).set_index("unit")
    ids = sorted(ann.VideoID.unique())
    split_of = {v: ("train" if i < 60 else ("val" if i < 70 else "test")) for i, v in enumerate(ids)}
    rows = []
    for v in ids:
        unit = f"cat_{v}"
        if unit not in meta.index:
            print("no frames for", unit); continue
        a = ann[ann.VideoID == v].sort_values("FrameNo")
        fps = float(vids.loc[v, "FPS"]) if "FPS" in vids.columns else float(meta.loc[unit, "fps"])
        start = a.FrameNo.values / fps; phase = a.Phase.values.astype(int)
        ks, t = _frames_of(frames_dir, unit, step)
        cur = _phase_at(start, phase, t)
        for k, tt, c in zip(ks, t, cur):
            row = dict(dataset="Cataract101", split=split_of[v], unit=unit, frame_id=int(k), t=float(tt),
                       image_path=os.path.join(frames_dir, f"{unit}_{k}.jpg"))
            for pid, name in CATARACT_PHASES.items():
                row[f"y_{name}"] = int(c == pid)
            rows.append(row)
    return pd.DataFrame(rows)



SAGES_FPS = 30.0


def sages(meta_csv, frames_dir):
    """SAGES CVS Challenge 2024: 90-s clips (one per operation), labelled every 5 s by three raters for C1-C3.
    meta_csv (written by run_extension.py) lists one row per clip with its official split, label file, the
    validation assignment and the exclusion flags. Labels: majority of three raters per criterion; CVS = all three.
    Frames: FRAMES_DIR/<unit>_<k>.jpg is decode index 150k (every 5 s at 30 fps)."""
    meta = pd.read_csv(meta_csv)
    meta = meta[meta.include == 1]
    rows = []
    for _, r in meta.iterrows():
        f = pd.read_csv(r.frame_csv)
        f.columns = [c.strip().lower() for c in f.columns]
        fid = "frame_id" if "frame_id" in f.columns else f.columns[0]
        for _, x in f.iterrows():
            k = int(x[fid]) // 150
            row = dict(dataset="SAGES", split=r.split, unit=f"sages_{r.unit}", frame_id=int(x[fid]),
                       t=float(x[fid]) / SAGES_FPS, image_path=os.path.join(frames_dir, f"{r.unit}_{k}.jpg"))
            for c in (1, 2, 3):
                votes = [int(x[f"c{c}_rater{j}"]) for j in (1, 2, 3)]
                row[f"y_C{c}"] = int(sum(votes) >= 2)
            row["y_CVS"] = row["y_C1"] * row["y_C2"] * row["y_C3"]
            rows.append(row)
    return pd.DataFrame(rows)


def check(m):
    missing = [c for c in REQUIRED if c not in m.columns]
    assert not missing, f"missing columns: {missing}"
    labels = [c for c in m.columns if c.startswith("y_")]
    assert labels, "no label columns (y_*)"
    assert set(m.split) <= {"train", "val", "test"}, set(m.split)
    assert not m.duplicated(["unit", "frame_id"]).any(), "duplicate (unit, frame_id)"
    for s in ("train", "val", "test"):
        assert (m.split == s).any(), f"no {s} rows"
    leak = set(m[m.split == "test"].unit) & set(m[m.split != "test"].unit)
    assert not leak, f"units in both test and train/val: {sorted(leak)[:5]}"
    for c in labels:
        assert set(m[c].unique()) <= {0, 1}, c
    exist = m.image_path.map(os.path.exists)
    summary = m.groupby("split").agg(units=("unit", "nunique"), frames=("unit", "size"))
    print(summary.to_string())
    print("labels:", labels)
    print(f"images found: {int(exist.sum())} of {len(m)}")
    return labels


if __name__ == "__main__":
    kind = sys.argv[1]
    if kind == "endoscapes":
        m = endoscapes(sys.argv[2]); m.to_csv(sys.argv[3], index=False); check(m)
    elif kind == "cholec80":
        m = cholec80(sys.argv[2], sys.argv[3]); m.to_csv(sys.argv[4], index=False); check(m)
    elif kind in ("heico", "cataract101"):
        # python manifests.py heico ROOT FRAMES_DIR STEP_S META_CSV OUT_CSV
        fn = heico if kind == "heico" else cataract101
        m = fn(sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5])
        lab = [c for c in m.columns if c.startswith("y_")]
        m = m.drop(columns=[c for c in lab if m[c].sum() == 0])   # phases that never occur
        m.to_csv(sys.argv[6], index=False); check(m)
    elif kind == "sages":
        # python manifests.py sages META_CSV FRAMES_DIR OUT_CSV
        m = sages(sys.argv[2], sys.argv[3]); m.to_csv(sys.argv[4], index=False); check(m)
    elif kind == "check":
        check(pd.read_csv(sys.argv[2]))
    else:
        raise SystemExit(__doc__)
