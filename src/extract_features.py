"""Frozen-backbone features for every frame of a manifest (GPU recommended).

Usage:
  python extract_features.py MANIFEST_CSV BACKBONE OUT_DIR [--weights PATH] [--batch 64] [--workers 4]

BACKBONE is one of the keys of BACKBONES below. SurgeNet weights must be downloaded first and passed
with --weights (see README). Output: OUT_DIR/<dataset>__<backbone>.npy (float16, one row per manifest
row, same order) and a .json file with the settings and the weight-loading report.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import timm
import torchvision
from PIL import Image

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# name: (constructor, input size, needs --weights)
BACKBONES = {
    "resnet50_in1k": (lambda: torchvision.models.resnet50(weights="IMAGENET1K_V2"), 224, False),
    "vitb16_in21k": (lambda: timm.create_model("vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True, num_classes=0), 224, False),
    "dinov2_vitb14_lvd": (lambda: timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=336), 336, False),
    "surgenetxl_dinov2_vitb14": (lambda: timm.create_model("vit_base_patch14_dinov2", pretrained=False, num_classes=0, img_size=336), 336, True),
    # the SurgeNetXL DINOv3 checkpoint has no storage (register) tokens and a trained fused qkv bias,
    # so the default timm DINOv3 (4 register tokens, no qkv bias) would keep random tokens and drop the bias
    "surgenetxl_dinov3_vitb16": (lambda: timm.create_model("vit_base_patch16_dinov3_qkvb", pretrained=False, num_classes=0, img_size=336, num_reg_tokens=0), 336, True),
}
# pretraining-only tensors of the SurgeNetXL checkpoints that the backbones do not use
PRETRAIN_ONLY = ("mask_token", "rope_embed.periods")


def build(name, weights):
    ctor, size, need_w = BACKBONES[name]
    model = ctor()
    report = "pretrained weights from the library"
    if name == "resnet50_in1k":
        model.fc = torch.nn.Identity()
    if need_w:
        assert weights and os.path.exists(weights), f"{name} needs --weights (downloaded checkpoint)"
        sd = torch.load(weights, map_location="cpu")
        if isinstance(sd, dict):
            for key in ("teacher", "student", "model", "state_dict"):
                if key in sd and isinstance(sd[key], dict):
                    sd = sd[key]; break
        sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
        k_bias = {}
        if name == "surgenetxl_dinov3_vitb16":
            # timm keeps q and v bias as parameters and k bias as a non-persistent buffer
            for k in [k for k in sd if k.endswith("attn.qkv.bias")]:
                q, kb, v = sd.pop(k).chunk(3)
                sd[k.replace("qkv.bias", "q_bias")] = q; sd[k.replace("qkv.bias", "v_bias")] = v
                k_bias[k.replace(".qkv.bias", "")] = kb
        msg = model.load_state_dict(sd, strict=False)
        for mod, kb in k_bias.items():
            model.get_submodule(mod).k_bias.copy_(kb)
        n_model = len(model.state_dict())
        loaded = n_model - len(msg.missing_keys)
        unexpected = [k for k in msg.unexpected_keys if k not in PRETRAIN_ONLY]
        report = (f"loaded {loaded}/{n_model} tensors (+{len(k_bias)} k_bias buffers); missing={msg.missing_keys[:10]}; "
                  f"unexpected={msg.unexpected_keys[:10]}")
        # any missing tensor stays at its random initialisation, and an unused checkpoint tensor means a different architecture
        assert not msg.missing_keys and not unexpected, "weights did not match the architecture: " + report
    return model.eval(), size, report


class Frames(torch.utils.data.Dataset):
    def __init__(self, paths, size):
        self.paths = paths
        self.tf = torchvision.transforms.Compose([
            torchvision.transforms.Resize((size, size), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("backbone", choices=sorted(BACKBONES)); ap.add_argument("out_dir")
    ap.add_argument("--weights"); ap.add_argument("--batch", type=int, default=64); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    m = pd.read_csv(a.manifest)
    missing = [p for p in m.image_path if not os.path.exists(p)]
    assert not missing, f"{len(missing)} images missing, e.g. {missing[:3]}"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, size, report = build(a.backbone, a.weights)
    model = model.to(dev)
    dl = torch.utils.data.DataLoader(Frames(m.image_path.tolist(), size), batch_size=a.batch,
                                     num_workers=a.workers, shuffle=False, pin_memory=dev == "cuda")
    feats = []
    t0 = time.time()
    with torch.inference_mode(), torch.autocast(dev, enabled=dev == "cuda"):
        for x in dl:
            f = model(x.to(dev, non_blocking=True))
            feats.append(f.float().cpu().numpy().astype(np.float16))
    F = np.concatenate(feats)
    assert F.shape[0] == len(m) and np.isfinite(F).all()
    os.makedirs(a.out_dir, exist_ok=True)
    stem = os.path.join(a.out_dir, f"{m.dataset.iloc[0]}__{a.backbone}")
    np.save(stem + ".npy", F)
    json.dump(dict(manifest=os.path.abspath(a.manifest), backbone=a.backbone, input_size=size, weights=a.weights,
                   weight_report=report, frames=len(m), dim=int(F.shape[1]), device=dev,
                   torch=torch.__version__, timm=timm.__version__, seconds=round(time.time() - t0, 1)),
              open(stem + ".json", "w"), indent=1)
    print(stem, F.shape, report)


if __name__ == "__main__":
    main()
