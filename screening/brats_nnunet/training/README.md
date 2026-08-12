# nnU-Net Screening Training

This directory contains the repository-owned training entry point for the
BraTS 2020 nnU-Net screening model.

The selected primary configuration is:

```text
dataset       = 500
configuration = 3d_fullres
plans         = nnUNetResEncUNetLPlans
device        = cuda
```

The ResEnc L plan is the primary configuration because it is the current
recommended nnU-Net preset for approximately 24 GB of GPU VRAM and provides
larger volumetric context than the standard or ResEnc M plans.

Training uses five-fold cross-validation.

Run one fold with:

```bash
screening/brats_nnunet/training/train_fold.sh 0
```

Valid folds are:

```text
0
1
2
3
4
```

The wrapper requires the standard nnU-Net environment variables:

```text
nnUNet_raw
nnUNet_preprocessed
nnUNet_results
```

The wrapper saves validation probability outputs using `--npz` so that
configuration comparison or ensembling remains possible.

Environment-variable overrides are available for development or fallback use:

```text
NNUNET_DATASET_ID
NNUNET_CONFIGURATION
NNUNET_PLANS
NNUNET_DEVICE
```

For example, ResEnc M can be tested without modifying the script:

```bash
NNUNET_PLANS=nnUNetResEncUNetMPlans \
screening/brats_nnunet/training/train_fold.sh 0
```

The repository does not store nnU-Net checkpoints or large training outputs.
Those remain under the configured `nnUNet_results` directory.
