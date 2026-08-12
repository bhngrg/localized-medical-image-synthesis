# BraTS nnU-Net External-Cohort Screening

This subtree contains the nnU-Net workflow used to construct a screened
external BraTS 2020 validation cohort for localized medical image
synthesis evaluation.

The screening workflow is intentionally separate from BR-LoRA training,
inference, posterior analysis, and reliability assessment.

Its sole scientific purpose is to identify BraTS 2020 validation slices
with no predicted tumor involvement so that they can serve as candidate
external base images.

------------------------------------------------------------------------

## Role in the Project

``` text
BraTS 2020 Training
    four MRI modalities
    + ground-truth segmentation
            │
            ▼
       train nnU-Net
            │
            ▼
       freeze model
            │
            ▼
BraTS 2020 Validation
    four MRI modalities
    no ground-truth segmentation
            │
            ▼
     nnU-Net prediction
            │
            ▼
 predicted whole-tumor masks
            │
            ▼
 identify slices with no
 predicted tumor involvement
            │
            ▼
 definitive external base manifest
            │
            ▼
 scripts/evaluate_br_lora_external.py
```

The nnU-Net model is therefore a cohort-screening tool only.

It is **not**

-   trained jointly with BR-LoRA,
-   used during BR-LoRA optimization,
-   used to modify BR-LoRA predictions, or
-   treated as ground truth for synthesized-image quality.

------------------------------------------------------------------------

## Source Datasets

The screening workflow consumes the already registered BraTS 2020
datasets.

### Training release

Uses the registered `dataset.yaml` and the labeled BraTS 2020 training
release.

### Validation release

Uses the registered `validation_dataset.yaml`. The official validation
release contains no ground-truth segmentations and is used only after
the nnU-Net model has been trained and frozen.

------------------------------------------------------------------------

## Repository Layout

``` text
screening/brats_nnunet/
├── README.md
├── configs/
├── manifests/
└── scripts/
```

Large nnU-Net datasets, preprocessing products, checkpoints, and
predictions remain outside the Git repository.

Recommended local layout:

``` text
/Users/bhanugarg/archive/nnUNet_brats_screening/
├── nnUNet_raw/
├── nnUNet_preprocessed/
└── nnUNet_results/
```

Paths should be supplied explicitly through configuration or the
standard nnU-Net environment variables.

------------------------------------------------------------------------

## Planned nnU-Net Dataset

``` text
nnUNet_raw/
└── DatasetXXX_BraTS2020Screening/
    ├── imagesTr/
    ├── labelsTr/
    ├── imagesTs/
    └── dataset.json
```

The dataset identifier (`XXX`) will be fixed before dataset preparation.

MRI channel mapping:

``` text
0000 = FLAIR
0001 = T1
0002 = T1ce
0003 = T2
```

------------------------------------------------------------------------

## Screening Principle

After training and freezing the nnU-Net model, inference will be run on
all 125 BraTS 2020 validation subjects.

For each axial slice the workflow will record:

``` text
subject_id
slice_index
predicted_tumor_pixels
screening_status
```

Candidate external base slices will be selected using a predefined
predicted-tumor criterion.

The resulting fixed manifest will be consumed directly by

``` text
scripts/evaluate_br_lora_external.py
```

The evaluator performs **no** case discovery or tumor screening.

------------------------------------------------------------------------

## Reproducibility Requirements

Record:

-   nnU-Net version
-   dataset identifier
-   dataset configuration
-   training folds
-   trainer/configuration
-   frozen checkpoint
-   Git commit
-   prediction provenance
-   manifest-generation criteria

Large checkpoints and prediction volumes remain outside Git.

------------------------------------------------------------------------

## Planned Workflow

1.  Prepare the BraTS datasets in nnU-Net format.
2.  Validate the generated dataset.
3.  Run nnU-Net planning and preprocessing.
4.  Train the segmentation model.
5.  Freeze the screening model.
6.  Run inference on all validation subjects.
7.  Generate slice-level tumor summaries.
8.  Select candidate external base slices.
9.  Audit the retained candidates.
10. Freeze the definitive external evaluation manifest.
11. Evaluate both BR-LoRA checkpoints using 100 posterior realizations
    per case.

------------------------------------------------------------------------

## Current Status

Repository scaffolding is complete.

The next implementation step is:

``` text
screening/brats_nnunet/scripts/prepare_nnunet_dataset.py
```

This script will construct the nnU-Net dataset from the registered BraTS
dataset specifications without modifying the original raw datasets.
