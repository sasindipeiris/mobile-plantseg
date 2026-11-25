"""PlantSeg (Zenodo v7) -> prepared/{images,masks}/{train,val,test}, classes.json, meta.json.

Official split kept. Masks become 0 = background, k = Metadata Index + 1 (the CSV label is trusted where a mask
disagrees). Images with short side > data.prepare_short_side are downscaled; the rest are copied unchanged.
"""

import argparse
import csv
import shutil
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from src.utils import DEFAULT_CFG, load_cfg, write_json

SPLITS = {"Training": "train", "Validation": "val", "Test": "test"}


def process(job):
    src, dst, split, name, label_file, k, short = job
    mask = cv2.imread(
        str(src / "annotations" / split / label_file), cv2.IMREAD_UNCHANGED
    )
    relabelled = set(np.unique(mask).tolist()) - {0} != {k}
    mask = np.where(mask > 0, k, 0).astype(np.uint8)
    stem = Path(name).stem
    s = short / min(mask.shape)
    if s < 1:
        size = (round(mask.shape[1] * s), round(mask.shape[0] * s))
        img = cv2.resize(
            cv2.imread(str(src / "images" / split / name)),
            size,
            interpolation=cv2.INTER_AREA,
        )
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(
            str(dst / "images" / split / f"{stem}.jpg"),
            img,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
    else:
        shutil.copy2(
            src / "images" / split / name, dst / "images" / split / f"{stem}.jpg"
        )
    cv2.imwrite(str(dst / "masks" / split / f"{stem}.png"), mask)
    return split, name if relabelled else None, bool(mask.any())


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=DEFAULT_CFG)
    a = p.parse_args()
    cfg = load_cfg(a.config)
    src, dst = Path(cfg["paths"]["raw_root"]), Path(cfg["paths"]["data_root"])

    rows = list(csv.DictReader(open(src / "Metadata.csv", encoding="utf-8-sig")))
    assert len({r["Name"] for r in rows}) == len(rows), "duplicate image names"
    diseases = dict(sorted({int(r["Index"]): r["Disease"] for r in rows}.items()))
    assert list(diseases) == list(
        range(len(diseases))
    ), "class indices are not contiguous"

    for sub in ("images", "masks"):
        for split in SPLITS.values():
            (dst / sub / split).mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            src,
            dst,
            SPLITS[r["Split"]],
            r["Name"],
            r["Label file"],
            int(r["Index"]) + 1,
            cfg["data"]["prepare_short_side"],
        )
        for r in rows
    ]
    with Pool(cfg["data"]["workers"]) as pool:
        res = pool.map(process, jobs, chunksize=32)

    write_json(dst / "classes.json", ["background", *diseases.values()])
    meta = {
        "images": Counter(r[0] for r in res),
        "relabelled_masks": sorted(r[1] for r in res if r[1]),
        "empty_masks": sum(not r[2] for r in res),
        "num_classes": len(diseases) + 1,
    }
    write_json(dst / "meta.json", meta)
    print(meta)


if __name__ == "__main__":
    main()
