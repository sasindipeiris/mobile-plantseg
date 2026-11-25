import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.dataset.dataset import build_dataset
from src.model.model import RepViTSeg
from src.utils import (
    DEFAULT_CFG,
    evaluate,
    load_cfg,
    load_classes,
    lr_lambda,
    save_ckpt,
    seg_loss,
    set_seed,
    write_json,
)


def main():
    p = argparse.ArgumentParser(description="Train RepViTSeg (one stage)")
    p.add_argument("--stage", required=True)
    p.add_argument("--config", default=DEFAULT_CFG)
    a = p.parse_args()

    cfg = load_cfg(a.config)
    tc, dc, st = cfg["train"], cfg["data"], cfg["train"]["stages"][a.stage]
    root = Path(cfg["paths"]["out"])
    out = root / a.stage
    out.mkdir(parents=True, exist_ok=True)
    set_seed(tc["seed"])
    dev = "cuda"
    torch.backends.cudnn.benchmark = True

    classes = load_classes(cfg)
    nc, ignore = len(classes), dc["ignore_index"]
    if st["init"] == "imagenet":
        model = RepViTSeg(nc, **cfg["model"], ckpt=cfg["paths"]["imagenet_ckpt"])
    else:
        model = RepViTSeg(nc, **cfg["model"])
        model.load_state_dict(
            torch.load(
                root / st["init"] / "best.pt", map_location="cpu", weights_only=True
            )["model"]
        )
    model.freeze_backbone(st["freeze_backbone"]).to(dev).train()

    opt = torch.optim.AdamW(
        model.param_groups(st["lr"], st["weight_decay"], st["backbone_lr_mult"])
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda(st["iters"], st["warmup"], tc["poly_power"], tc["min_lr_ratio"])
    )
    scaler = torch.amp.GradScaler(enabled=tc["amp"])

    kw = dict(
        num_workers=dc["workers"], pin_memory=True, persistent_workers=dc["workers"] > 0
    )
    train_dl = DataLoader(
        build_dataset(cfg, "train", train=True),
        batch_size=tc["batch"],
        shuffle=True,
        drop_last=True,
        **kw,
    )
    val_ds = build_dataset(cfg, "val", train=False)
    g = torch.Generator().manual_seed(tc["seed"])
    idx = torch.randperm(len(val_ds), generator=g)[: tc["val_subset"]].tolist()
    val_dl = DataLoader(Subset(val_ds, idx), batch_size=1, **kw)

    it, best, history, t0 = 0, -1.0, [], time.time()
    while it < st["iters"]:
        for img, mask in train_dl:
            img, mask = img.to(dev, non_blocking=True), mask.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=tc["amp"]):
                logits = model(img)
            loss = seg_loss(logits, mask, ignore, tc["loss"]["ce"], tc["loss"]["dice"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            it += 1

            if it % tc["log_every"] == 0:
                s = (time.time() - t0) / it
                print(
                    f'[{a.stage}] it {it}/{st["iters"]} loss {loss.item():.4f} '
                    f'lr {max(g["lr"] for g in opt.param_groups):.2e} {s:.3f}s/it eta {s * (st["iters"] - it) / 60:.0f}min',
                    flush=True,
                )

            if it % tc["eval_every"] == 0 or it == st["iters"]:
                model.eval()
                with torch.autocast("cuda", dtype=torch.float16, enabled=tc["amp"]):
                    m = evaluate(model, val_dl, nc, ignore, dev)
                model.train()
                m.pop("IoU_per_class")
                history.append({"iter": it, **m})
                print(
                    f"[{a.stage}] val it {it}: "
                    + " ".join(f"{k} {v:.2f}" for k, v in m.items()),
                    flush=True,
                )
                save_ckpt(out / "last.pt", model, cfg, classes, iter=it, val=m)
                if m["mIoU"] > best:
                    best = m["mIoU"]
                    save_ckpt(out / "best.pt", model, cfg, classes, iter=it, val=m)
                write_json(out / "history.json", history)

            if it >= st["iters"]:
                break

    write_json(
        out / "results.json",
        {
            "stage": a.stage,
            "stage_cfg": st,
            "best_val_mIoU": best,
            "minutes": (time.time() - t0) / 60,
            "history": history,
        },
    )


if __name__ == "__main__":
    main()
