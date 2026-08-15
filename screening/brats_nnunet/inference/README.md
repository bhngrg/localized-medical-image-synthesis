# nnU-Net Validation Inference

This directory contains the Falcon launcher used to apply the completed five-fold nnU-Net screening model to the official BraTS 2020 validation cohort.

## Input Contract

The prepared validation cohort is stored in Dataset500 `imagesTs`:

```text
/scratch/bhanug/archive/nnUNet_brats_screening/
nnUNet_raw/
Dataset500_BraTS2020Screening/
imagesTs/
```

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

## Falcon Launcher

Submit with:

```bash
sbatch   screening/brats_nnunet/inference/predict_validation.slurm
```

The production launcher uses:

```text
account   = fdtbiotech
partition = l40s_normal_q
GPU       = 1 x NVIDIA L40S
CPUs      = 8
memory    = 64 GB
walltime  = 8 hours
```

The script validates the input image count, subject count, plans file, model directory, and all five final checkpoints before calling `nnUNetv2_predict`.

## Output

Predictions are written outside Git to:

```text
/scratch/bhanug/nnUNet_brats_screening/
validation_predictions_l40s_normal_q/
```

Expected prediction contract:

```text
BraTS20_Validation_001.nii.gz
BraTS20_Validation_002.nii.gz
...
BraTS20_Validation_125.nii.gz
```

The launcher performs a final count check and fails if it does not find exactly 125 predicted masks.



## Relationship to the BR-LoRA Pipeline

This inference stage is executed once after the screening model has been
trained. The resulting validation predictions are treated as fixed inputs for
the downstream BR-LoRA workflow.

The predicted tumor masks identify tumor-free validation slices, which become
the candidate base-image pool used during BR-LoRA synthetic library design.


## Downstream Use

After inference completes, the prediction masks are transferred to the local workstation and consumed by:

```text
screening/brats_nnunet/scripts/screen_validation_slices.py
```

The primary tumor-free criterion is:

```text
predicted_tumor_pixels == 0
```

The resulting slice-level screening output is used to construct the fixed external base-image cohort for downstream BR-LoRA evaluation.

Prediction masks and runtime logs are generated artifacts and are not stored in Git.


## Scientific Scope

This workflow performs only inference with the frozen screening model. It does
not perform BR-LoRA training, Bayesian adaptation, synthetic image generation,
posterior sampling, or downstream evaluation. Those stages begin only after
the screened cohort has been constructed.
