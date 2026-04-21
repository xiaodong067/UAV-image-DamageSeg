# Post-Disaster Damage Assessment

This repository contains two complementary modules for post-disaster building damage assessment:

- `post-disaster-diffusion-lora`
  Diffusion-based damage generation with staged LoRA training, full-resolution inference, and ablation analysis.
- `post-disaster-damage-segmentation`
  Building damage semantic segmentation with model training, evaluation, and Jetson/TensorRT deployment support.

## Subprojects

The repository is organized into the following subprojects:

1. `post-disaster-diffusion-lora`
2. `post-disaster-damage-segmentation`

These names align with the paper theme and clearly separate generation and recognition tasks.

## Repository Structure

```text
.
|-- post-disaster-diffusion-lora
|-- post-disaster-damage-segmentation
`-- README.md
```

## Module Overview

### 1. Post-Disaster Diffusion LoRA

This module focuses on post-disaster building damage generation with diffusion models. It includes:

- training data preparation
- Stage 1 shared LoRA training
- Stage 2 damage-level-specific LoRA training
- full-resolution inference
- end-to-end ablation experiments

Representative entry scripts:

- `stage1_train.py`
- `stage2_train.py`
- `inference_fullres.py`
- `run_full_ablation.py`

### 2. Post-Disaster Damage Segmentation

This module focuses on semantic segmentation for building damage assessment. It includes:

- DeepLab-based training and prediction
- enhanced building-damage modeling
- mIoU evaluation
- ONNX export
- Jetson and TensorRT deployment scripts

Representative entry scripts:

- `train.py`
- `deeplab.py`
- `miou.py`
- `run_jetson.py`

## Suggested Open-Source Positioning

The parent repository name can use:

- `Post-Disaster-Damage-Assessment`

If you want a shorter repo name, you can also use:

- `DamageAssessmentLab`
- `DisasterDamageVision`

## Notes

- The current codebase keeps the two modules independent, which is good for open-source maintenance.
- Before release, it is worth cleaning hard-coded local paths, large experiment outputs, and temporary cache files.
- Some internal path strings may still reference older names such as `disaster_lora/...`, so they should be checked before release.
