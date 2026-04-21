# Disaster LoRA Module Guide

This directory contains the research code for data preparation, staged training, full-resolution inference, and ablation analysis.

## Core Scripts

### Data utilities

- [`1_data_visualization.py`](1_data_visualization.py)
  Visualizes the source image and mask distribution before training-data preparation.

- [`2_prepare_training_data.py`](2_prepare_training_data.py)
  Converts building instances into 512x512 training patches and writes `metadata.json`.

### Training

- [`stage1_train.py`](stage1_train.py)
  Trains the shared severity-aware adapter with LoRA, level tokens, and severity embedding.

- [`stage2_train.py`](stage2_train.py)
  Loads the Stage 1 adapter, merges the shared LoRA into the base UNet, and trains an L4-specific adapter with stronger boundary supervision.

### Inference and ablation

- [`inference_fullres.py`](inference_fullres.py)
  Performs instance-level full-resolution generation using the trained checkpoint.

- [`run_full_ablation.py`](run_full_ablation.py)
  Entry point for the end-to-end ablation pipeline.

- [`ablation_runner.py`](ablation_runner.py)
  Builds reference subsets, manifests, and experiment folders.

- [`ablation_inference.py`](ablation_inference.py)
  Runs batched generation for each ablation experiment.

- [`ablation_metrics.py`](ablation_metrics.py)
  Computes KID, FID, LPIPS, background preservation, and boundary-related metrics.

- [`ablation_visualize.py`](ablation_visualize.py)
  Produces comparison figures, local zooms, and sweep plots.

- [`ablation_summary.py`](ablation_summary.py)
  Aggregates experiment metrics into summary tables and plots.

### Shared helper

- [`evaluation_utils.py`](evaluation_utils.py)
  Shared visualization helper used during training.

## Configuration

- [`ablation_config.yaml`](ablation_config.yaml)
  Central experiment configuration for checkpoints, sampling quotas, prompts, evaluation options, and visualization settings.

- [`requirements.txt`](requirements.txt)
  Python dependencies for this project.

## Minimal Reproduction Order

1. Prepare the dataset with `2_prepare_training_data.py`.
2. Train Stage 1 with `stage1_train.py`.
3. Train Stage 2 with `stage2_train.py`.
4. Run `inference_fullres.py` for qualitative results.
5. Run `run_full_ablation.py` for quantitative comparisons.

## Important Local Paths

Before running experiments, update:

- data paths
- checkpoint paths
- output paths

in [`ablation_config.yaml`](ablation_config.yaml) and in any local command you use.
