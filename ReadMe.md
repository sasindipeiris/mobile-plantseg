# On Device Plant Disease Lesion Segmentation with RepViT

This repo contains an end to end pipeline for training a semantic segmentation model and deploying it on mobile devices to localize and classify plant disease lesions across 115 disease classes in in-the-wild photographs.

## Methodology

- We explored RepViT, a mobile CNN with a ViT inspired design, as a backbone for plant disease lesion segmentation on mobile devices.

- Starting from ImageNet pretrained RepViT-M0.9, we first froze the backbone and trained a Feature Pyramid Network neck with a Semantic FPN head on the PlantSeg dataset using a combined cross entropy and Dice loss.

- We then finetuned the full network end to end with a 10× lower backbone learning rate and evaluated both models on the official PlantSeg test split for segmentation quality and disease classification.

- The finetuned model was then reparameterized by fusing its batch norm layers into the convolutions and exported to LiteRT (TFLite) as an FP32 model for GPU and a post training INT8 model for CPU.

- Finally, both models were deployed in an Android demo app that segments lesions in leaf photos on the phone and reports the predicted disease, the lesion area and the inference latency.

## Implementation Details

### Dataset

- **Name:** PlantSeg v7 (Zenodo `17719108`)
- **Count:** 7,774 in-the-wild images
- **Splits:** 70/10/20
- **Classes:** 116 total (`0` = background, `1–115` = disease lesions; `ignore_index = 255`)

### Model

- **Architecture:** `RepViT-M0.9` backbone + FPN neck + Semantic FPN head
- **Pretrained weights:** `repvit_m0_9_distill_300e.pth` (distilled ImageNet-1K)

### Training

- **Loss:** `1.0 × CrossEntropyLoss(ignore_index=255)` + `1.0 × Batch-Present Soft Dice`

| Parameter                   | Frozen backbone                             | Full finetune                                           |
| --------------------------- | ------------------------------------------- | ------------------------------------------------------- |
| Initialization              | ImageNet RepViT-M0.9                        | Best frozen backbone checkpoint                         |
| Backbone                    | Frozen                                      | Trainable                                               |
| Optimizer                   | AdamW                                       | AdamW                                                   |
| Learning rate schedule      | Polynomial decay (power 0.9, min 0.01×)     | Linear warmup + polynomial decay (power 0.9, min 0.01×) |
| Iterations                  | 8,000                                       | 20,000                                                  |
| Warmup iterations           | 0                                           | 1,000                                                   |
| Learning rate (neck + head) | 1e-4                                        | 1e-4                                                    |
| Learning rate (backbone)    | 0                                           | 1e-5 (0.1×)                                             |
| Weight decay                | 1e-4                                        | 0.01                                                    |
| Batch size                  | 16                                          | 16                                                      |
| Crop size                   | 512 × 512                                   | 512 × 512                                               |
| Mixed precision (AMP)       | Yes                                         | Yes                                                     |
| Validation                  | 500-image val subset every 4,000 iterations | 500-image val subset every 4,000 iterations             |

### Evaluation

The following metrics are reported in all experiments:

- **mIoU:** mean IoU over all classes (background and 115 diseases)
- **aAcc:** overall pixel accuracy
- **Lesion IoU / F1:** lesion vs. background, with any disease class counted as lesion
- **Image accuracy:** the most frequent predicted disease in the image matches the ground truth disease

## Results

All metrics below are reported on the official test split of PlantSeg.

PyTorch models, input resized so the short side is 512:

| Metric             | Frozen backbone | Full finetune |
| ------------------ | --------------- | ------------- |
| mIoU (%)           | 21.66           | 32.99         |
| aAcc (%)           | 84.10           | 86.08         |
| Lesion IoU (%)     | 56.43           | 63.11         |
| Lesion F1 (%)      | 72.15           | 77.39         |
| Image accuracy (%) | 62.97           | 71.94         |

Exported full finetune model, input letterboxed to 512 × 512 as on the phone:

| Metric             | TFLite FP32 | TFLite INT8 |
| ------------------ | ----------- | ----------- |
| mIoU (%)           | 33.18       | 14.49       |
| aAcc (%)           | 86.00       | 81.27       |
| Lesion IoU (%)     | 62.23       | 39.76       |
| Lesion F1 (%)      | 76.72       | 56.90       |
| Image accuracy (%) | 70.34       | 43.75       |

## Reproduce

### Setup

1. Create a conda environment and install the required dependencies:

```bash
conda create -n plantseg python=3.10 -y

conda activate plantseg

pip install -r requirements.txt
```

2. Download the PlantSeg dataset:

```bash
mkdir -p data/raw

wget -O data/raw/plantseg.zip "https://zenodo.org/records/17719108/files/plantseg.zip?download=1"

unzip -q data/raw/plantseg.zip -d data/raw

ls data/raw/plantseg/Metadata.csv data/raw/plantseg/images data/raw/plantseg/annotations
```

3. Download the original model weights:

```bash
wget -P out/ckpt https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m0_9_distill_300e.pth
```

4. Preprocess the raw data using:

```bash
python -m src.dataset.preprocess
```

### Experiments

1. Run the experiments (`frozen` = frozen backbone, `gentle` = full finetune):

```bash
python -m src.train --stage frozen

python -m src.train --stage gentle
```

2. Evaluate on the test set using:

```bash
python -m src.evaluate --weights out/runs/frozen/best.pt
python -m src.evaluate --weights out/runs/gentle/best.pt
python -m src.evaluate --weights out/runs/gentle/best.pt --imgsz 512   # letterboxed like the phone
```

### Deployment

1. Export to TFLite:

```bash
python -m src.deploy.export --weights out/runs/gentle/best.pt
python -m src.evaluate --weights out/runs/export/repvit_m0_9_512_fp32.tflite
python -m src.evaluate --weights out/runs/export/repvit_m0_9_512_int8.tflite
```

2. Pick demo photos for the phone:

```bash
python -m src.dataset.demo --weights out/runs/gentle/best.pt
```

Transfer the photos in `data/prepared/images/test_demo/` to the phone.

3. Phone setup (once per phone):

Settings → About phone → Software information → tap Build number 7 times → Developer options → turn on USB debugging. Connect by USB and accept the prompt.

4. Build and install the app:

Install [Android Studio](https://developer.android.com/studio) (it includes JDK 17 and `adb`).

```bash
cp out/runs/export/repvit_m0_9_512_fp32.tflite \
   out/runs/export/repvit_m0_9_512_int8.tflite \
   out/runs/export/labels.txt android/app/src/main/assets/
```

Android Studio: Open → `android/` → Settings → Build Tools → Gradle → set Gradle JDK to 17 → Sync → select the connected phone in the device dropdown (toolbar) → Run.

## Discussion & Conclusion

- Full finetuning raised mIoU from 21.66 to 32.99 and image accuracy from 62.97 to 71.94, so adapting the backbone matters for this task.

- Lesion IoU (63.11) is much higher than mIoU (32.99), which shows that the model localizes lesions well but often confuses the disease class, so the main limitation is distinguishing between 115 diseases rather than finding the lesions.

- The FP32 TFLite model matches the PyTorch model at the same 512 × 512 input (33.18 mIoU for both), so export loses no accuracy, whereas post training INT8 quantization reduces mIoU to 14.49, leaving the CPU model small but not yet accurate enough.

## Further Development

- **Improve INT8 quantization.** Post training INT8 quantization currently drops test mIoU from 33.18 (FP32) to 14.49. Keeping the most sensitive layers in higher precision and quantization aware training could recover most of this gap on low end CPU devices.

- **Benchmark against other mobile backbones.** Train alternative mobile backbones (e.g. MobileOne, EfficientFormerV2, FastViT, MobileNetV4) with the same Semantic FPN head and training protocol and compare accuracy and on device latency.

## License

- **Code:** [Apache License 2.0](LICENSE). [src/model/repvit.py](src/model/repvit.py) is taken from the official [RepViT repository](https://github.com/THU-MIG/RepViT), which is also licensed under Apache 2.0.
- **Dataset:** PlantSeg is not included in this repo. Of its images, 7,741 are licensed under CC BY-NC and 33 under CC0, as listed in `Metadata.csv`. Models trained on PlantSeg, including the exported TFLite models, are therefore for non commercial use only.

## References

1. A. Wang, H. Chen, Z. Lin, J. Han, G. Ding. **RepViT: Revisiting Mobile CNN From ViT Perspective.** CVPR 2024, pp. 15909–15920. [arXiv:2307.09283](https://arxiv.org/abs/2307.09283) · [code](https://github.com/THU-MIG/RepViT)
2. T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, S. Belongie. **Feature Pyramid Networks for Object Detection.** CVPR 2017. [arXiv:1612.03144](https://arxiv.org/abs/1612.03144)
3. A. Kirillov, R. Girshick, K. He, P. Dollár. **Panoptic Feature Pyramid Networks** (Semantic FPN head). CVPR 2019. [arXiv:1901.02446](https://arxiv.org/abs/1901.02446)
4. T. Wei, Z. Chen, X. Yu, S. Chapman, P. Melloy, Z. Huang. **PlantSeg: A Large-Scale In-the-wild Dataset for Plant Disease Segmentation.** 2024. [arXiv:2409.04038](https://arxiv.org/abs/2409.04038)
