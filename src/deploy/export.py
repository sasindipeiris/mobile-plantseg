import argparse
import copy
from pathlib import Path

import litert_torch
import torch
from litert_torch.quantize.pt2e_quantizer import (
    PT2EQuantizer,
    get_symmetric_quantization_config,
)
from litert_torch.quantize.quant_config import QuantConfig
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from src.dataset.dataset import build_dataset
from src.utils import DEFAULT_CFG, load_cfg, load_model, pad_input, set_seed, write_json


def main():
    p = argparse.ArgumentParser(
        description="Fuse and export to LiteRT: FP32 (S25 GPU) and INT8 (M02 CPU)"
    )
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int)
    p.add_argument("--config", default=DEFAULT_CFG)
    a = p.parse_args()

    cfg = load_cfg(a.config)
    ec = cfg["export"]
    S = a.imgsz or ec["imgsz"]
    set_seed(cfg["train"]["seed"])
    model, ck = load_model(a.weights)
    name = f"{ck['model_cfg']['backbone']}_{S}"
    out = Path(cfg["paths"]["out"]) / "export"
    out.mkdir(parents=True, exist_ok=True)

    ref = copy.deepcopy(model)
    model.fuse()
    x = torch.rand(2, 3, S, S)
    with torch.no_grad():
        y_ref = ref(x)
        diff = ((y_ref - model(x)).abs().max() / y_ref.abs().max()).item()
    assert diff < ec["fuse_tol"], f"fused model differs: {diff:.2e}"
    del ref

    sample = (torch.rand(1, 3, S, S),)
    files = {"fp32": out / f"{name}_fp32.tflite", "int8": out / f"{name}_int8.tflite"}
    litert_torch.convert(model, sample).export(str(files["fp32"]))

    ds = build_dataset(cfg, "train", train=False)
    idx = torch.randperm(len(ds))[: ec["calib_images"]].tolist()
    quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=False)
    )

    gm = prepare_pt2e(torch.export.export(model, sample).module(), quantizer)
    with torch.no_grad():
        for i in idx:
            gm(pad_input(ds[i][0][None], S)[0])
    gm = convert_pt2e(gm, fold_quantize=False)
    litert_torch.convert(
        gm, sample, quant_config=QuantConfig(pt2e_quantizer=quantizer)
    ).export(str(files["int8"]))

    (out / "labels.txt").write_text("\n".join(ck["classes"]) + "\n")
    write_json(
        out / f"{name}_export.json",
        {
            "weights": str(Path(a.weights).resolve()),
            "imgsz": S,
            "fuse_max_rel_diff": diff,
            "params_fused": sum(p.numel() for p in model.parameters()),
            "MB": {k: round(f.stat().st_size / 2**20, 2) for k, f in files.items()},
        },
    )
    print("\n".join(str(f) for f in files.values()))


if __name__ == "__main__":
    main()
