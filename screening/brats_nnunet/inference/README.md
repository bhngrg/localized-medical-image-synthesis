# nnU-Net Validation Inference

This directory contains the Slurm launcher used to apply the completed
five-fold nnU-Net screening model to the registered BraTS 2020 validation
cohort.

Production inference requires a **CUDA-capable GPU**. Ordinary CPU execution
and Apple MPS are not the supported production path for this workflow.

For nnU-Net installation and general usage, follow the
[official nnU-Net documentation](https://github.com/MIC-DKFZ/nnUNet). This
README documents only the repository-specific validation-inference workflow.

## Input Contract

The prepared validation cohort is stored in Dataset500 `imagesTs` beneath the
configured `nnunet_archive_root`:

```text
<nnunet_archive_root>/
└── nnUNet_raw/
    └── Dataset500_BraTS2020Screening/
        └── imagesTs/
```

Set `nnunet_archive_root` in `data/folders.yaml`, or provide
`NNUNET_ARCHIVE_ROOT` in the job environment.

Expected counts:

```text
validation subjects = 125
image files          = 500
channels per subject = 4
```

Channel mapping:

```text
0000 = FLAIR
0001 = T1
0002 = T1ce
0003 = T2
```

## Model Contract

Inference uses the completed production model:

```text
dataset       = 500
trainer       = nnUNetTrainer
configuration = 3d_fullres
plans         = nnUNetResEncUNetLPlans
folds         = 0 1 2 3 4
checkpoint    = checkpoint_final.pth
device        = cuda
```

All five fold checkpoints must exist before prediction begins.

## Slurm Launcher

Submit from the repository root:

```bash
sbatch screening/brats_nnunet/inference/predict_validation.slurm
```

The launcher uses the `fdtbiotech` account and performs preflight checks on the
prepared validation inputs, nnU-Net plans, model directory, CUDA availability,
and all five final fold checkpoints before calling `nnUNetv2_predict`.

The repository launcher defaults reflect the production Falcon workflow and
may be adjusted for another Slurm environment when necessary.

## Output

Predictions are written outside Git beneath the configured `nnunet_run_root`:

```text
<nnunet_run_root>/
└── validation_predictions_l40s_normal_q/
```

Set `nnunet_run_root` in `data/folders.yaml`, or provide `NNUNET_RUN_ROOT` in
the job environment.

Expected prediction contract:

```text
BraTS20_Validation_001.nii.gz
BraTS20_Validation_002.nii.gz
...
BraTS20_Validation_125.nii.gz
```

The launcher performs a final count check and fails unless exactly 125
prediction masks are present.

See [`../../../data/README.md`](../../../data/README.md) for the shared
machine-specific path configuration.

## Downstream Use

The prediction masks remain in the configured run root and are consumed
directly by:

```bash
python screening/brats_nnunet/scripts/screen_validation_slices.py --help
python screening/brats_nnunet/scripts/screen_validation_slices.py --validate-only
```

The primary tumor-free criterion is:

```text
predicted_tumor_pixels == 0
```

The resulting screening output defines the candidate external base-image pool
used by the compatibility audits, frozen 250-case external cohort, and
10,000-case BR-LoRA library design.

Prediction masks remain generated artifacts outside Git. Selected production
inference logs that provide scientific or reproducibility provenance are
retained under `logs/screening/brats_nnunet/inference/`.

## Scientific Scope

This stage performs inference only with the frozen nnU-Net screening model. It
does not perform BR-LoRA training, Bayesian adaptation, synthetic image
generation, posterior sampling, or downstream segmentation evaluation.

See [`../README.md`](../README.md) for the complete screening and
compatibility workflow.
