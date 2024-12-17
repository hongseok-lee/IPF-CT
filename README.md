# IPF-CT Survival Model Training

This README describes the training configuration and usage for the `IPF-CT Survival` model.

## Table of Contents
- [Overview](#overview)
- [Training Configuration](#training-configuration)
- [Environment Setup](#environment-setup)
- [Command Example](#command-example)
- [Arguments Description](#arguments-description)
- [Model Details](#model-details)
- [Output Directories](#output-directories)
- [Key Features](#key-features)

---

## Overview
This project involves training a survival prediction model using CT images and corresponding volumetric densities. The model utilizes annotated masks with the capability to predict survival and specific volume features.


---

## Training Configuration
### Base Parameters
- **Random Crop Gamma:** `1`
- **Epochs:** `200`
- **Learning Rate:** `1e-4`
- **Batch Size:** `4`
- **Optimizer:** `adamw`
- **Model Type:** `clinical_volume_survival_detach_double_density`
- **Predicted Volumes:** `cropped_normal_density`, `cropped_fibrosis_density`
- **Mask Type:** `softmax`
- **Scheduler:** `onecyclelr` (cosine annealing parameters: T_0, T_mult, and T_gamma)

---

## Environment Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure GPU setup:
   - **CUDA_VISIBLE_DEVICES** must be configured correctly for multi-GPU usage.
   - Precision 16-bit training requires PyTorch and CUDA support.
3. Dataset preparation:
   - Place CT image data in the `img_dir` directory.
   - Provide dataset metadata in the `dataset_file_path` directory.

---

## Command Example
Below is the command template used for model training:

```bash
python train.py \
  --train \
  --batch_size 4 \
  --gpus 4  \
  --precision 16 \
  --max_followup 6 \
  --img_file_type png \
  --min_num_images 0 \
  --use_only_thin_cuts_for_ct \
  --slice_thickness_filter 4.0 \
  --resample_pixel_spacing_prob 1.0 \
  --fix_seed_for_multi_image_augmentations \
  --use_annotations \
  --img_mean 128.1722 \
  --img_std 87.1849 \
  --img_size 256 256 \
  --num_images 300 \
  --num_chan 3 \
  --max_epochs 20 \
  --init_lr 1e-4 \
  --lr_decay 0.1 \
  --weight_decay 1e-3 \
  --momentum 0.9 \
  --dropout 0.5 \
  --optimizer adamw \
  --patience 5 \
  --num_workers 4 \
  --profiler simple \
  --num_sanity_val_steps 0 \
  --save_dir ./ckpts \
  --img_dir ./ \
  --dataset snuh_h5 \
  --mask_name honeycomb_reticular \
  --dataset_file_path ./ \
  --scheduler onecyclelr \
  --mask_interpolation area \
  --pool_type 1 2 3 \
  --mask_type softmax \
  --focal_loss_gamma 0 \
  --model_type clinical_volume_survival_detach_double_density \
  --pixel_spacing 111 \
  --random_crop \
  --random_crop_sigma 1 \
  --pred_volume "cropped_normal_density cropped_fibrosis_density" \
  --t_mult 1 \
  --t_0 50 \
  --t_gamma 1 \
  --tuning_metric c_index mean_time_dependent_auroc survival_loss loss
```

---

## Arguments Description

| Argument                     | Description                                    |
|------------------------------|------------------------------------------------|
| `--train`                    | Flag to enable training mode.                  |
| `--batch_size`               | Batch size for training.                       |
| `--gpus`                     | GPU indices to use for training.               |
| `--precision`                | Training precision (16-bit or 32-bit).         |
| `--max_epochs`               | Number of epochs for training.                 |
| `--init_lr`                  | Initial learning rate.                         |
| `--optimizer`                | Optimizer type (e.g., adamw).                  |
| `--scheduler`                | Learning rate scheduler.                       |
| `--img_size`                 | Input image resolution.                        |
| `--model_type`               | Type of model architecture to use.             |
| `--pred_volume`              | Volume densities to predict.                   |
| `--mask_type`                | Type of mask used (e.g., softmax).             |
| `--save_dir`                 | Directory to save checkpoints.                 |
| `--results_path`             | Path to save results and metrics.              |
| `--tuning_metric`            | Metrics to optimize during training.           |

---

## Model Details
The **IPF-CT Survival Model** model trained a combination of annotated masks and volumetric density predictions to predict survival outcomes. 
---

## Output Directories
1. **Checkpoints:**
   - Path: `./ckpts`
   - Stores model checkpoints during training.
2. **Results:**
   - Path: `./results`
   - Contains evaluation metrics (C-index, AUROC, loss values, etc.).

---

## Citation
If you use this code or model, please cite appropriately.
