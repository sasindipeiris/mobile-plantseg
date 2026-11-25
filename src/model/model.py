"""RepViT backbone + FPN neck + Semantic FPN head (Kirillov et al., 2019) for lesion segmentation.

Input: RGB in [0, 1], H and W multiples of 32. Output: logits at 1/4 resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.repvit import (
    Conv2d_BN,
    repvit_m0_6,
    repvit_m0_9,
    repvit_m1_0,
    repvit_m1_1,
    repvit_m1_5,
    repvit_m2_3,
)

BACKBONES = {
    f.__name__: f
    for f in (
        repvit_m0_6,
        repvit_m0_9,
        repvit_m1_0,
        repvit_m1_1,
        repvit_m1_5,
        repvit_m2_3,
    )
}


def replace_batchnorm(net):
    for name, child in net.named_children():
        if hasattr(child, "fuse"):
            fused = child.fuse()
            setattr(net, name, fused)
            replace_batchnorm(fused)
        elif isinstance(child, nn.BatchNorm2d):
            setattr(net, name, nn.Identity())
        else:
            replace_batchnorm(child)


def conv_bn_relu(cin, cout):
    return nn.Sequential(Conv2d_BN(cin, cout, 3, 1, 1), nn.ReLU(inplace=True))


class RepViTSeg(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone="repvit_m0_9",
        ckpt=None,
        fpn_ch=128,
        head_ch=128,
        dropout=0.1,
    ):
        super().__init__()
        net = BACKBONES[backbone](num_classes=0)
        if ckpt:
            sd = torch.load(ckpt, map_location="cpu", weights_only=True)
            sd = sd.get("model", sd)
            net.features.load_state_dict(
                {k[9:]: v for k, v in sd.items() if k.startswith("features.")}
            )
        self.backbone = net.features

        cfgs = net.cfgs
        self.out_idx = [
            i + 1 for i in range(len(cfgs)) if i == len(cfgs) - 1 or cfgs[i + 1][5] == 2
        ]
        in_chs = [cfgs[i - 1][2] for i in self.out_idx]
        assert len(self.out_idx) == 4, self.out_idx

        self.lateral = nn.ModuleList(nn.Conv2d(c, fpn_ch, 1) for c in in_chs)
        self.fpn = nn.ModuleList(
            nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1) for _ in in_chs
        )
        self.heads = nn.ModuleList()
        for lvl in range(4):
            layers = []
            for j in range(max(1, lvl)):
                layers.append(conv_bn_relu(fpn_ch if j == 0 else head_ch, head_ch))
                if lvl:
                    layers.append(
                        nn.Upsample(
                            scale_factor=2, mode="bilinear", align_corners=False
                        )
                    )
            self.heads.append(nn.Sequential(*layers))
        self.drop = nn.Dropout2d(dropout)
        self.cls = nn.Conv2d(head_ch, num_classes, 1)

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        x = (x - self.mean) / self.std
        feats = []
        for i, f in enumerate(self.backbone):
            x = f(x)
            if i in self.out_idx:
                feats.append(x)
        lat = [l(f) for l, f in zip(self.lateral, feats)]
        for i in range(3, 0, -1):
            lat[i - 1] = lat[i - 1] + F.interpolate(
                lat[i], scale_factor=2, mode="nearest"
            )
        out = sum(h(c(p)) for h, c, p in zip(self.heads, self.fpn, lat))
        return self.cls(self.drop(out))

    def freeze_backbone(self, frozen=True):
        self.backbone.requires_grad_(not frozen)
        return self

    def train(self, mode=True):
        super().train(mode)
        for m in self.backbone.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
        return self

    def param_groups(self, lr, weight_decay, backbone_lr_mult=1.0):
        groups = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                groups.setdefault((n.startswith("backbone."), p.ndim > 1), []).append(p)
        return [
            {
                "params": ps,
                "lr": lr * (backbone_lr_mult if bb else 1.0),
                "weight_decay": weight_decay if decay else 0.0,
            }
            for (bb, decay), ps in groups.items()
        ]

    @torch.no_grad()
    def fuse(self):
        self.eval()
        replace_batchnorm(self)
        return self
