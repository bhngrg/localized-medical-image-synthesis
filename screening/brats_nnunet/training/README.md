# nnU-Net Screening Training

This directory contains the repository-owned training entry points for the
BraTS 2020 nnU-Net screening model.

The production configuration is:

```text
dataset       = 500
trainer       = nnUNetTrainer
configuration = 3d_fullres
plans         = nnUNetResEncUNetLPlans
device        = cuda
```

The ResEnc L plan uses a `160 x 192 x 160` 3D patch at 1 mm isotropic spacing.

Production training requires a **CUDA-capable GPU**. Ordinary CPU execution and
Apple MPS are not the supported production path for this workflow.

For nnU-Net installation and general usage, follow the
[official nnU-Net documentation](https://github.com/MIC-DKFZ/nnUNet). This
README documents only the repository-specific screening workflow.

## Cross-Validation Split

Training uses five-fold cross-validation over 369 labeled BraTS 2020 training
subjects.

```text
fold 0: train=295, validation=74
fold 1: train=295, validation=74
fold 2: train=295, validation=74
fold 3: train=295, validation=74
fold 4: train=296, validation=73
```

The split is stored in the nnU-Net preprocessed Dataset500 directory as
`splits_final.json`.

## Entry Points

Run one fold directly with:

```bash
screening/brats_nnunet/training/train_fold.sh 0
```

Valid folds are `0`, `1`, `2`, `3`, and `4`.

The Slurm production launcher is:

```text
screening/brats_nnunet/training/train_all_folds.slurm
```

Submit it from the repository root:

```bash
sbatch screening/brats_nnunet/training/train_all_folds.slurm
```

The launcher trains folds 0-4 as a Slurm array using the `fdtbiotech` account.
The folds are independent and may run concurrently as cluster resources permit.

## Path Configuration

When `train_fold.sh` is run directly, it requires the standard nnU-Net
environment variables:

```text
nnUNet_raw
nnUNet_preprocessed
nnUNet_results
```

The Slurm launcher derives these paths from `nnunet_archive_root` and
`nnunet_run_root` in `data/folders.yaml`.

The roots may instead be supplied through:

```text
NNUNET_ARCHIVE_ROOT
NNUNET_RUN_ROOT
```

Environment-variable overrides for the nnU-Net training configuration remain
available:

```text
NNUNET_DATASET_ID
NNUNET_CONFIGURATION
NNUNET_PLANS
NNUNET_DEVICE
```

See [`../../../data/README.md`](../../../data/README.md) for the shared
machine-specific path configuration.

## Completed Production Training

All five folds completed successfully.

Mean validation Dice:

```text
fold 0 = 0.9268
fold 1 = 0.9189
fold 2 = 0.9187
fold 3 = 0.9130
fold 4 = 0.9100
```

Each fold produced:

```text
checkpoint_best.pth
checkpoint_final.pth
validation/
training_log_*.txt
progress.png
debug.json
```

Training used `--npz` so validation probability outputs are retained where
supported by nnU-Net.

## Output Location

Production results are stored outside Git beneath the configured
`nnunet_run_root`:

```text
<nnunet_run_root>/
└── nnUNet_results_l40s_normal_q/
```

The repository does not store checkpoints, fold validation predictions,
training logs, or other large generated artifacts.

The completed folds are consumed by:

```text
screening/brats_nnunet/inference/predict_validation.slurm
```

## Relationship to BR-LoRA

The nnU-Net model is trained once and then frozen. Its ensemble predictions are
used only to screen the BraTS validation cohort and construct the admissible
external base-image pool.

It is not updated during BR-LoRA training and is not used as a synthesized-image
quality target.

See [`../README.md`](../README.md) for the complete screening and
compatibility workflow.
