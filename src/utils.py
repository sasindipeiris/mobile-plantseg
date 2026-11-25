import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.model.model import RepViTSeg

DEFAULT_CFG = Path(__file__).resolve().parents[1] / "config.yaml"


def load_cfg(path=DEFAULT_CFG):
    """Load the YAML config; relative entries under `paths` are resolved against the config file's folder."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    cfg["paths"] = {k: str(path.parent / v) for k, v in cfg["paths"].items()}
    return cfg


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_classes(cfg):
    return json.loads((Path(cfg["paths"]["data_root"]) / "classes.json").read_text())


def save_ckpt(path, model, cfg, classes, **extra):
    torch.save(
        {
            "model": model.state_dict(),
            "model_cfg": cfg["model"],
            "classes": classes,
            **extra,
        },
        path,
    )


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = RepViTSeg(len(ck["classes"]), **ck["model_cfg"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


def lr_lambda(iters, warmup, power, min_ratio):
    def f(it):
        w = min(1.0, (it + 1) / warmup) if warmup else 1.0
        return w * ((1 - min_ratio) * (1 - it / iters) ** power + min_ratio)

    return f


def seg_loss(logits, target, ignore, w_ce=1.0, w_dice=1.0):
    logits = F.interpolate(
        logits.float(), size=target.shape[-2:], mode="bilinear", align_corners=False
    )
    loss = w_ce * F.cross_entropy(logits, target, ignore_index=ignore)
    if w_dice:
        valid = target != ignore
        cls = torch.unique(target[valid])
        prob = logits.softmax(1)[:, cls] * valid.unsqueeze(1)
        onehot = (
            (target.unsqueeze(1) == cls.view(1, -1, 1, 1)) & valid.unsqueeze(1)
        ).float()
        inter = (prob * onehot).sum((0, 2, 3))
        dice = (2 * inter + 1) / (prob.sum((0, 2, 3)) + onehot.sum((0, 2, 3)) + 1)
        loss = loss + w_dice * (1 - dice.mean())
    return loss


def pad_input(img, imgsz=None, stride=32):
    """(1,3,H,W) -> padded input and valid (h, w). imgsz: letterbox to a fixed square (phone input)."""
    H, W = img.shape[-2:]
    if imgsz:
        s = imgsz / max(H, W)
        h, w = round(H * s), round(W * s)
        img = F.interpolate(img, size=(h, w), mode="bilinear", align_corners=False)
        ph, pw = imgsz, imgsz
    else:
        h, w = H, W
        ph, pw = math.ceil(H / stride) * stride, math.ceil(W / stride) * stride
    return F.pad(img, (0, pw - w, 0, ph - h)), (h, w)


def predict_labels(predict, img, imgsz=None):
    H, W = img.shape[-2:]
    x, (h, w) = pad_input(img, imgsz)
    logits = F.interpolate(
        predict(x).float(), size=x.shape[-2:], mode="bilinear", align_corners=False
    )
    logits = logits[..., :h, :w]
    if (h, w) != (H, W):
        logits = F.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )
    return logits.argmax(1)[0]


@torch.no_grad()
def evaluate(predict, loader, nc, ignore, device, imgsz=None):
    hist = torch.zeros(nc * nc, dtype=torch.long, device=device)
    img_ok = img_n = 0
    for img, mask in loader:
        img, mask = img.to(device), mask[0].to(device)
        pred = predict_labels(predict, img, imgsz)
        valid = mask != ignore
        t, p = mask[valid], pred[valid]
        hist += torch.bincount(nc * t + p, minlength=nc * nc)
        gt_lesion = t[t > 0]
        if gt_lesion.numel():
            img_n += 1
            p_lesion = p[p > 0]
            img_ok += int(
                p_lesion.numel() > 0
                and torch.bincount(p_lesion, minlength=nc).argmax()
                == torch.bincount(gt_lesion, minlength=nc).argmax()
            )
    return metrics(hist.view(nc, nc), img_ok, img_n)


def metrics(hist, img_ok, img_n):
    hist = hist.double().cpu()
    tp, gt, pr = hist.diag(), hist.sum(1), hist.sum(0)
    iou = tp / (gt + pr - tp)
    l_tp, l_fp, l_fn = hist[1:, 1:].sum(), hist[0, 1:].sum(), hist[1:, 0].sum()
    return {
        "mIoU": 100 * torch.nanmean(iou).item(),
        "aAcc": 100 * (tp.sum() / hist.sum()).item(),
        "lesion_IoU": 100 * (l_tp / (l_tp + l_fp + l_fn)).item(),
        "lesion_F1": 100 * (2 * l_tp / (2 * l_tp + l_fp + l_fn)).item(),
        "image_acc": 100 * img_ok / max(img_n, 1),
        "n_lesion_images": img_n,
        "IoU_per_class": [
            None if math.isnan(v) else round(100 * v, 2) for v in iou.tolist()
        ],
    }
