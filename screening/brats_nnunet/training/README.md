# nnU-Net Screening Training

This directory contains the repository-owned training entry points for the BraTS 2020 nnU-Net screening model.

The production configuration is:

```text
dataset       = 500
trainer       = nnUNetTrainer
configuration = 3d_fullres
plans         = nnUNetResEncUNetLPlans
device        = cuda
```

The ResEnc L plan uses a `160 x 192 x 160` 3D patch at 1 mm isotropic spacing.

## Cross-Validation Split

Training uses five-fold cross-validation over 369 labeled BraTS 2020 training subjects.

```text
fold 0: train=295, validation=74
fold 1: train=295, validation=74
fold 2: train=295, validation=74
fold 3: train=295, validation=74
fold 4: train=296, validation=73
```

The split is stored in the nnU-Net preprocessed Dataset500 directory as `splits_final.json`.

## Entry Points

Run one fold with:

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

The completed production run launched folds 0-4 as a Slurm array on `l40s_normal_q` using the `fdtbiotech` account.

When `train_fold.sh` is run directly, it requires the standard nnU-Net environment variables:

```text
nnUNet_raw
nnUNet_preprocessed
nnUNet_results
```

The Slurm launcher derives these paths from `nnunet_archive_root` and `nnunet_run_root` in `data/folders.yaml`. The roots may instead be supplied through `NNUNET_ARCHIVE_ROOT` and `NNUNET_RUN_ROOT`.

Environment-variable overrides for the nnU-Net training configuration remain available:

```text
NNUNET_DATASET_ID
NNUNET_CONFIGURATION
NNUNET_PLANS
NNUNET_DEVICE
```

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

Training used `--npz` so validation probability outputs are retained where supported by nnU-Net.



## Relationship to the BR-LoRA Pipeline

The nnU-Net model is trained once and then frozen. The resulting five-fold
ensemble is used exclusively for external-cohort screening and is **not**
updated during BR-LoRA development.

Its outputs provide the tumor-screening predictions used to construct the
screened external cohort, which subsequently feeds the BR-LoRA synthetic
library design and downstream evaluation pipeline.


## Output Location

Production results are stored outside Git beneath the configured `nnunet_run_root`:

```text
<nnunet_run_root>/
nnUNet_results_l40s_normal_q/
```

Set `nnunet_run_root` in `data/folders.yaml`, or provide `NNUNET_RUN_ROOT` in the job environment.

The repository does not store checkpoints, fold validation predictions, training logs, or other large generated artifacts.

The completed fold models are consumed by the validation-inference launcher in:

```text
screening/brats_nnunet/inference/predict_validation.slurm
```


## Scientific Scope

This training workflow is independent of BR-LoRA optimization, Bayesian
adaptation, posterior uncertainty estimation, synthetic image generation,
and downstream segmentation experiments. Its sole purpose is to produce the
frozen screening model used for external-cohort construction.
