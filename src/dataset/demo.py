"""Pick a small demo set from the test split: clear images the model gets right, spread over plants and diseases.

Kept: correct diagnosis, lesion IoU >= demo.min_lesion_iou, short side >= demo.min_short_side, lesion share within
demo.lesion_frac. Picked round-robin over plants, one image per disease before any disease repeats.
Selected by model performance, so it is for demos only: report metrics on the full test split.
"""

import argparse
import csv
import shutil
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

import torch

from src.dataset.dataset import build_dataset
from src.utils import DEFAULT_CFG, load_cfg, load_model, predict_labels, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--weights", required=True)
    p.add_argument("--config", default=DEFAULT_CFG)
    a = p.parse_args()
    cfg = load_cfg(a.config)
    dc, root = cfg["demo"], Path(cfg["paths"]["data_root"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    meta = csv.DictReader(
        open(Path(cfg["paths"]["raw_root"]) / "Metadata.csv", encoding="utf-8-sig")
    )
    plant_of = {Path(r["Name"]).stem: r["Plant"] for r in meta}
    model, ck = load_model(a.weights, dev)
    classes, nc = ck["classes"], len(ck["classes"])
    ds = build_dataset(cfg, "test", train=False)

    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            img, mask = ds[i]
            pred = predict_labels(model, img[None].to(dev), cfg["export"]["imgsz"])
            mask = mask.to(dev)
            k = int(mask.max())
            gt, pr = mask > 0, pred > 0
            diag = (
                int(torch.bincount(pred[pr], minlength=nc).argmax()) if pr.any() else 0
            )
            stem = ds.imgs[i].stem
            rows.append(
                {
                    "stem": stem,
                    "plant": plant_of[stem],
                    "disease": classes[k],
                    "correct": diag == k,
                    "lesion_iou": round(((gt & pr).sum() / (gt | pr).sum()).item(), 4),
                    "lesion_frac": round(gt.float().mean().item(), 4),
                    "short_side": min(ds.read_size(i)),
                }
            )

    lo, hi = dc["lesion_frac"]
    good = [
        r
        for r in rows
        if r["correct"]
        and r["lesion_iou"] >= dc["min_lesion_iou"]
        and r["short_side"] >= dc["min_short_side"]
        and lo <= r["lesion_frac"] <= hi
    ]

    by_plant = defaultdict(lambda: defaultdict(list))
    for r in sorted(good, key=lambda r: -r["lesion_iou"]):
        by_plant[r["plant"]][r["disease"]].append(r)
    queues = [
        [r for rank in zip_longest(*d.values()) for r in rank if r]
        for _, d in sorted(by_plant.items())
    ]
    pick = [r for rnd in zip_longest(*queues) for r in rnd if r][: dc["size"]]

    for sub, ext in (("images", "jpg"), ("masks", "png")):
        out = root / sub / dc["split"]
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        for r in pick:
            shutil.copy2(root / sub / "test" / f"{r['stem']}.{ext}", out)
    with open(root / f"{dc['split']}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pick[0]))
        w.writeheader()
        w.writerows(pick)

    summary = {
        "weights": str(Path(a.weights).resolve()),
        "criteria": dc,
        "test_images": len(rows),
        "eligible": len(good),
        "picked": len(pick),
        "plants": len({r["plant"] for r in pick}),
        "diseases": len({r["disease"] for r in pick}),
        "plants_in_test": len({r["plant"] for r in rows}),
        "diseases_in_test": len({r["disease"] for r in rows}),
    }
    write_json(root / f"{dc['split']}.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
