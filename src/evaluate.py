import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset.dataset import build_dataset
from src.utils import (
    DEFAULT_CFG,
    evaluate,
    load_cfg,
    load_classes,
    load_model,
    write_json,
)


class TFLiteModel:
    def __init__(self, path, threads):
        from ai_edge_litert.interpreter import Interpreter

        self.it = Interpreter(model_path=str(path), num_threads=threads)
        self.it.allocate_tensors()
        inp, self.out = (
            self.it.get_input_details()[0],
            self.it.get_output_details()[0]["index"],
        )
        self.idx, shape = inp["index"], inp["shape"]
        self.nchw = shape[1] == 3
        self.size = int(shape[2] if self.nchw else shape[1])

    def __call__(self, x):
        x = x.numpy() if self.nchw else x.permute(0, 2, 3, 1).numpy()
        self.it.set_tensor(self.idx, x)
        self.it.invoke()
        y = torch.from_numpy(self.it.get_tensor(self.out))
        return y if self.nchw else y.permute(0, 3, 1, 2)


def main():
    p = argparse.ArgumentParser(
        description="Evaluate a .pt checkpoint or an exported .tflite model"
    )
    p.add_argument("--weights", required=True)
    p.add_argument("--split", default="test")
    p.add_argument(
        "--imgsz",
        type=int,
        help="letterbox to a fixed square, as on the phone (.tflite: from model)",
    )
    p.add_argument("--config", default=DEFAULT_CFG)
    a = p.parse_args()

    cfg = load_cfg(a.config)
    w = Path(a.weights)
    if w.suffix == ".tflite":
        dev = "cpu"
        model = TFLiteModel(w, cfg["eval"]["threads"])
        classes, imgsz = load_classes(cfg), model.size
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model, ck = load_model(w, dev)
        classes, imgsz = ck["classes"], a.imgsz

    loader = DataLoader(
        build_dataset(cfg, a.split, train=False),
        batch_size=1,
        num_workers=cfg["data"]["workers"],
    )
    m = evaluate(model, loader, len(classes), cfg["data"]["ignore_index"], dev, imgsz)
    m["IoU_per_class"] = dict(zip(classes, m["IoU_per_class"]))
    tag = f"{w.parent.name}_{w.stem}_{a.split}" + (f"_{imgsz}" if imgsz else "")
    write_json(
        Path(cfg["paths"]["out"]) / "eval" / f"{tag}.json",
        {"weights": str(w.resolve()), "split": a.split, "imgsz": imgsz, **m},
    )
    print(tag, " ".join(f"{k} {v:.2f}" for k, v in m.items() if isinstance(v, float)))


if __name__ == "__main__":
    main()
