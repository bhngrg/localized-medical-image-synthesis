# BraTS nnU-Net External-Cohort Screening

This subtree contains the nnU-Net workflow used to construct a screened external BraTS 2020 validation cohort for localized medical image synthesis evaluation.

The screening workflow is intentionally separate from BR-LoRA training, inference, posterior analysis, and reliability assessment. Its sole scientific purpose is to identify BraTS 2020 validation slices with no predicted whole-tumor involvement so that they can serve as candidate external base images.

## Role in the Project

```text
BraTS 2020 training release
    369 labeled subjects
    FLAIR + T1 + T1ce + T2
            |
            v
Dataset500_BraTS2020Screening
    binary whole-tumor labels
            |
            v
nnU-Net ResEnc L
    3d_fullres
    five-fold cross-validation
            |
            v
frozen fold models
            |
            v
BraTS 2020 validation release
    125 unlabeled subjects
    500 nnU-Net image files
            |
            v
five-fold ensemble prediction
            |
            v
125 predicted whole-tumor masks
            |
            v
slice-level tumor-free screening
            |
            v
external BR-LoRA base-image candidates
```

The nnU-Net model is a cohort-screening tool only. It is not trained jointly with BR-LoRA, used during BR-LoRA optimization, used to modify BR-LoRA predictions, or treated as ground truth for synthesized-image quality.

## Dataset

The registered nnU-Net dataset is:

```text
Dataset500_BraTS2020Screening
```

MRI channel mapping:

```text
0000 = FLAIR
0001 = T1
0002 = T1ce
0003 = T2
```

Segmentation target:

```text
0 = background
1 = whole_tumor
```

The binary whole-tumor target maps original BraTS tumor labels 1, 2, and 4 to 1.

Registered cohort sizes:

```text
training subjects   = 369
validation subjects = 125
training images     = 1476
training labels     = 369
validation images   = 500
```

Large nnU-Net datasets, preprocessing products, checkpoints, validation predictions, and runtime logs remain outside Git.

## Selected nnU-Net Configuration

The primary screening model uses:

```text
dataset       = 500
trainer       = nnUNetTrainer
configuration = 3d_fullres
plans         = nnUNetResEncUNetLPlans
architecture  = ResidualEncoderUNet
device        = CUDA
```

The ResEnc L plan uses a 3D patch size of `160 x 192 x 160` at 1 mm isotropic spacing.

Five-fold cross-validation uses the fixed `splits_final.json` generated for the 369 training subjects:

```text
fold 0: train=295, validation=74
fold 1: train=295, validation=74
fold 2: train=295, validation=74
fold 3: train=295, validation=74
fold 4: train=296, validation=73
```

## Training Status

All five nnU-Net folds completed successfully on Falcon using `l40s_normal_q`.

Mean validation Dice:

```text
fold 0 = 0.9268
fold 1 = 0.9189
fold 2 = 0.9187
fold 3 = 0.9130
fold 4 = 0.9100
```

Each fold produced both `checkpoint_best.pth` and `checkpoint_final.pth`.

Repository-owned training entry points are located under:

```text
screening/brats_nnunet/training/
```

## Validation Inference

External validation inference uses the 125 prepared validation subjects under Dataset500 `imagesTs`.

The production launcher is:

```text
screening/brats_nnunet/inference/predict_validation.slurm
```

It uses all five folds and `checkpoint_final.pth` to generate the five-fold ensemble predictions.

Falcon prediction output:

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

The prediction masks are transferred back to the local workstation for slice-level screening and downstream external-cohort construction.

## Slice-Level Screening

The repository contains the initial screening implementation:

```text
screening/brats_nnunet/scripts/screen_validation_slices.py
```

The primary tumor-free criterion mirrors the historical composition pipeline:

```text
predicted_tumor_pixels == 0
```

The script is currently a development skeleton and will be finalized and audited locally after the 125 validation prediction masks are available.

The final screening output will provide a fixed, auditable set of candidate tumor-free validation slices for external BR-LoRA evaluation.

## Repository Layout

```text
screening/brats_nnunet/
├── README.md
├── configs/
├── inference/
│   ├── README.md
│   └── predict_validation.slurm
├── manifests/
├── scripts/
│   ├── prepare_nnunet_dataset.py
│   └── screen_validation_slices.py
└── training/
    ├── README.md
    ├── train_all_folds.slurm
    └── train_fold.sh
```

## Reproducibility Requirements

Record the following for each production screening run:

- nnU-Net version
- dataset identifier
- dataset configuration
- trainer and plans
- cross-validation split
- fold checkpoints
- Git commit
- Falcon partition
- prediction provenance
- slice-screening rule
- final candidate-manifest provenance

Generated datasets, checkpoints, predictions, and runtime logs are not stored in the Git repository.