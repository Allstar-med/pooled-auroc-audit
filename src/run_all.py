"""First stage of the pipeline: Endoscapes and Cholec80 manifests, frozen features, linear probes and the audit.

Usage (only the datasets you pass are run):
  python run_all.py --work D:/oea_run --weights D:/weights
                    [--endoscapes D:/data/endoscapes]
                    [--cholec80 D:/data/cholec80]          (folder with videos/, tool_annotations/, phase_annotations/)

At the end, WORK/results_stage1.zip holds the audit tables, the validation logs and the feature reports
(a few MB). The images, features and predictions stay on this computer.
"""
import argparse
import glob
import os
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
BACKBONES = [("resnet50_in1k", None), ("vitb16_in21k", None), ("dinov2_vitb14_lvd", None),
             ("surgenetxl_dinov2_vitb14", "DINOv2_ViTb14_size336_SurgeNetXL.pth"),
             ("surgenetxl_dinov3_vitb16", "DINOv3_ViTb16_size336_SurgeNetXL.pth")]


def sh(*args):
    print(">>", " ".join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=HERE)


def cholec80_frames(root, frames):
    os.makedirs(frames, exist_ok=True)
    for v in range(1, 81):
        if glob.glob(os.path.join(frames, f"video{v:02d}_0.*")):
            continue  # already extracted
        src = glob.glob(os.path.join(root, "videos", f"video{v:02d}.*"))
        assert src, f"video{v:02d} not found under {root}/videos"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", src[0], "-vf", "fps=1/5", "-start_number", "0",
                        "-q:v", "2", os.path.join(frames, f"video{v:02d}_%d.jpg")], check=True)
        print("frames", v, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--weights", required=True)
    ap.add_argument("--endoscapes"); ap.add_argument("--cholec80")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--exclude", help="Local frame-screen CSV required for the Endoscapes paper analysis")
    a = ap.parse_args()
    if a.endoscapes and (not a.exclude or not os.path.isfile(a.exclude)):
        ap.error("Endoscapes requires --exclude PATH to the local frame-screen CSV")
    exclusion_args = ["--exclude", os.path.abspath(a.exclude)] if a.exclude else []
    W = os.path.abspath(a.work)
    for d in ("manifests", "features", "preds", "results"):
        os.makedirs(os.path.join(W, d), exist_ok=True)
    jobs = []
    if a.endoscapes:
        m = os.path.join(W, "manifests", "endoscapes.csv")
        sh("manifests.py", "endoscapes", a.endoscapes, m); jobs.append(("Endoscapes", "endo", m))
    if a.cholec80:
        fr = os.path.join(a.cholec80, "frames_5s")
        cholec80_frames(a.cholec80, fr)
        m = os.path.join(W, "manifests", "cholec80.csv")
        sh("manifests.py", "cholec80", a.cholec80, fr, m); jobs.append(("Cholec80", "c80", m))
    for ds, short, m in jobs:
        for bb, wfile in BACKBONES:
            npy = os.path.join(W, "features", f"{ds}__{bb}.npy")
            if not os.path.exists(npy):
                extra = ["--weights", os.path.join(a.weights, wfile)] if wfile else []
                sh("extract_features.py", m, bb, os.path.join(W, "features"), *extra)
            pred = os.path.join(W, "preds", f"{short}_{bb}.csv")
            if not os.path.exists(pred):
                sh("train_heads.py", m, npy, pred)
    preds = sorted(p for p in glob.glob(os.path.join(W, "preds", "*.csv"))
                   if not p.endswith("_validation.csv"))
    sh("audit_all.py", os.path.join(W, "results", "selected"), *preds, "--B", a.B, *exclusion_args)
    sh("audit_all.py", os.path.join(W, "results", "all"), *preds, "--zoo", "all", "--B", a.B, *exclusion_args)
    out = os.path.join(W, "results_stage1.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in (glob.glob(os.path.join(W, "results", "*", "*.csv")) + glob.glob(os.path.join(W, "features", "*.json"))
                  + glob.glob(os.path.join(W, "preds", "*_validation.csv")) + glob.glob(os.path.join(W, "preds", "*_settings.json"))):
            z.write(f, os.path.relpath(f, W))
    print("\nDone:", out)


if __name__ == "__main__":
    main()
