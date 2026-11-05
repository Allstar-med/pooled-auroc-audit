"""Experiment A: end-to-end fine-tuning of a backbone with one multi-label head (GPU).

Usage:
  python finetune.py MANIFEST_CSV BACKBONE OUT_CSV [--weights PATH] [--seed 1] [--epochs 12]
                     [--batch 32] [--workers 4] [--smoke]

BACKBONE is resnet50_in1k or surgenetxl_dinov2_vitb14 (the latter needs --weights, as in extract_features.py).
All labels of the manifest (columns y_*) are trained jointly with class-weighted binary cross-entropy.
The whole network is trained on the training split; after every epoch the validation and test frames are
scored, so every epoch is one configuration. The epoch is selected per label on the validation split
(highest E1 for sel_E1, highest W for sel_W); the test split is never used for any choice.
--smoke runs 2 epochs on 4 training videos and 2 validation/test videos, to check the set-up in minutes.
"""
import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image

from extract_features import MEAN, STD, build
from oea_io import write_epoch_predictions

SETTINGS = {  # backbone: (input size, backbone learning rate)
    "resnet50_in1k": (224, 1e-4),
    "surgenetxl_dinov2_vitb14": (224, 2e-5),
}
HEAD_LR, WD, WARMUP_EPOCHS = 1e-3, 0.05, 1


class Frames(torch.utils.data.Dataset):
    def __init__(self, paths, Y, size, train):
        self.paths, self.Y = paths, Y
        if train:
            self.tf = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.8, 1.25),
                                                         interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
                torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(MEAN, STD)])
        else:
            self.tf = torchvision.transforms.Compose([
                torchvision.transforms.Resize((size, size), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), torch.from_numpy(self.Y[i])


class Net(torch.nn.Module):
    def __init__(self, backbone, dim, n_out):
        super().__init__()
        self.backbone, self.head = backbone, torch.nn.Linear(dim, n_out)

    def forward(self, x):
        return self.head(self.backbone(x))


def make_backbone(name, weights, size):
    model, native, report = build(name, weights)
    if name.startswith("surgenetxl") and size != native:
        # fine-tuning at 224 px to fit an 8 GB GPU; the position embedding is resampled from the 336-px checkpoint
        model.set_input_size(img_size=size)
    if hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
    dim = 2048 if name == "resnet50_in1k" else model.num_features
    return model.train(), dim, report


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("backbone", choices=sorted(SETTINGS)); ap.add_argument("out")
    ap.add_argument("--weights"); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    seed_all(a.seed)
    m = pd.read_csv(a.manifest)
    labels = [c for c in m.columns if c.startswith("y_")]
    if a.smoke:
        keep_units = []
        pos = m.assign(_p=m[labels].sum(1)).groupby(["split", "unit"])._p.sum()   # videos with most positives
        for s, k in (("train", 4), ("val", 2), ("test", 2)):
            keep_units += list(pos[s].sort_values(ascending=False).index[:k])
        m = m[m.unit.isin(keep_units)].reset_index(drop=True); a.epochs = 2
    # labels with a single class in training or validation cannot be trained or selected
    labels = [l for l in labels if m.loc[m.split == "train", l].nunique() == 2 and m.loc[m.split == "val", l].nunique() == 2]
    Y = m[labels].values.astype(np.float32)
    tr = (m.split == "train").values; ev = ~tr
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    size, lr = SETTINGS[a.backbone]
    bb, dim, report = make_backbone(a.backbone, a.weights, size)
    net = Net(bb, dim, len(labels)).to(dev)
    pos = Y[tr].sum(0); neg = tr.sum() - pos
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.clip(neg / np.maximum(pos, 1), 1, 50), device=dev))
    opt = torch.optim.AdamW([{"params": net.backbone.parameters(), "lr": lr},
                             {"params": net.head.parameters(), "lr": HEAD_LR}], weight_decay=WD)
    g = torch.Generator(); g.manual_seed(a.seed)
    dl_tr = torch.utils.data.DataLoader(Frames(m.image_path.values[tr], Y[tr], size, True), batch_size=a.batch,
                                        shuffle=True, drop_last=True, num_workers=a.workers, generator=g,
                                        pin_memory=dev == "cuda", persistent_workers=a.workers > 0)
    dl_ev = torch.utils.data.DataLoader(Frames(m.image_path.values[ev], Y[ev], size, False), batch_size=a.batch * 2,
                                        shuffle=False, num_workers=a.workers, pin_memory=dev == "cuda")
    steps = a.epochs * len(dl_tr); warm = WARMUP_EPOCHS * len(dl_tr)
    base = [pg["lr"] for pg in opt.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    logits, epochs, step, t0 = [], [], 0, time.time()
    for ep in range(1, a.epochs + 1):
        net.train()
        for x, y in dl_tr:
            f = min(1.0, (step + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, step / steps)))  # warm-up, then cosine
            for pg, b in zip(opt.param_groups, base):
                pg["lr"] = b * f
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                loss = loss_fn(net(x.to(dev, non_blocking=True)), y.to(dev, non_blocking=True))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); step += 1
        net.eval(); out = []
        with torch.inference_mode(), torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            for x, _ in dl_ev:
                out.append(net(x.to(dev, non_blocking=True)).float().cpu().numpy())
        L = np.concatenate(out)
        assert np.isfinite(L).all(), "non-finite outputs"
        logits.append(L.astype(np.float32)); epochs.append(ep)
        print(f"epoch {ep}/{a.epochs} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    name = f"ft_{a.backbone}|seed{a.seed}"
    write_epoch_predictions(m, ev, labels, np.stack(logits), epochs, name, "finetune", a.out,
                            dict(manifest=a.manifest, backbone=a.backbone, seed=a.seed, epochs=a.epochs, batch=a.batch,
                                 input_size=size, backbone_lr=lr, head_lr=HEAD_LR, weight_decay=WD,
                                 weight_report=report, labels=labels, smoke=a.smoke, device=dev,
                                 torch=torch.__version__, seconds=round(time.time() - t0)))


if __name__ == "__main__":
    main()
