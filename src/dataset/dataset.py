import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(0)
GRAY = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def rescale(img, mask, short, long):
    h, w = mask.shape
    s = min(short / min(h, w), long / max(h, w))
    size = (round(w * s), round(h * s))
    return (
        cv2.resize(img, size, interpolation=cv2.INTER_LINEAR),
        cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST),
    )


def random_crop(img, mask, size, cat_max_ratio, tries=10):
    h, w = mask.shape
    ch, cw = min(size, h), min(size, w)
    for _ in range(tries):
        y, x = random.randint(0, h - ch), random.randint(0, w - cw)
        cnt = np.bincount(mask[y : y + ch, x : x + cw].ravel())
        cnt = cnt[cnt > 0]
        if len(cnt) > 1 and cnt.max() / cnt.sum() < cat_max_ratio:
            break
    return img[y : y + ch, x : x + cw], mask[y : y + ch, x : x + cw]


def photometric(img, c):
    img = img.astype(np.float32)
    if random.random() < 0.5:
        img += random.uniform(-c["brightness"], c["brightness"])
    if random.random() < 0.5:
        img *= random.uniform(*c["contrast"])
    if random.random() < 0.5:
        gray = (img @ GRAY)[..., None]
        img = gray + (img - gray) * random.uniform(*c["saturation"])
    return np.clip(img, 0, 255)


def pad_to(img, mask, size, ignore):
    ph, pw = max(size - mask.shape[0], 0), max(size - mask.shape[1], 0)
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=ignore)
    return img, mask


class PlantSeg(Dataset):
    def __init__(self, root, split, c, train):
        self.imgs = sorted((Path(root) / "images" / split).glob("*.jpg"))
        self.masks = [Path(root) / "masks" / split / f"{p.stem}.png" for p in self.imgs]
        assert self.imgs, f"no images in {root}/images/{split}"
        self.c, self.train = c, train

    def __len__(self):
        return len(self.imgs)

    def read_size(self, i):
        return cv2.imread(str(self.masks[i]), cv2.IMREAD_UNCHANGED).shape

    def __getitem__(self, i):
        c = self.c
        img = cv2.cvtColor(cv2.imread(str(self.imgs[i])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.masks[i]), cv2.IMREAD_UNCHANGED)
        if self.train:
            r = random.uniform(*c["scale_range"])
            img, mask = rescale(img, mask, c["short_side"] * r, c["max_long_side"] * r)
            img, mask = random_crop(img, mask, c["crop"], c["cat_max_ratio"])
            if random.random() < c["hflip"]:
                img, mask = img[:, ::-1], mask[:, ::-1]
            if random.random() < c["vflip"]:
                img, mask = img[::-1], mask[::-1]
            img = photometric(img, c)
            img, mask = pad_to(
                img, np.ascontiguousarray(mask), c["crop"], c["ignore_index"]
            )
        else:
            img, mask = rescale(img, mask, c["short_side"], c["max_long_side"])
        img = (
            torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
            .float()
            .div_(255)
        )
        return img, torch.from_numpy(mask.astype(np.int64))


def build_dataset(cfg, split, train):
    return PlantSeg(cfg["paths"]["data_root"], split, cfg["data"], train)
