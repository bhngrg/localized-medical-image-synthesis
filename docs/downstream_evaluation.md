# Downstream Segmentation Evaluation

## Purpose

The downstream evaluation tests whether BR-LoRA-generated images can improve a
separate tumor-segmentation model when synthetic images are added to the real
BraTS training data.

The segmentation model is trained independently of BR-LoRA. Synthetic images
are therefore evaluated through their effect on a downstream task rather than
through the image generator's own training objective.

Operational commands for running the experiments are documented in
[`../downstream_evaluation/README.md`](../downstream_evaluation/README.md).

## Experimental Comparison

Three training regimes are implemented:

1. **Real only**
   - real BraTS training images only.

2. **Real + BR-LoRA posterior mean**
   - the same real BraTS training data;
   - plus the frozen 10,000-case BR-LoRA synthetic library generated using the
     posterior-mean adapter parameters.

3. **Real + BR-LoRA posterior sampling**
   - the same real BraTS training data;
   - plus the same frozen 10,000-case synthetic design;
   - synthetic images are drawn from the retained BR-LoRA posterior samples.

The real training and validation cohorts are held fixed across regimes. The
only intended experimental difference is the presence and form of BR-LoRA
synthetic augmentation.

## Frozen BraTS Training Data

The real-only training manifest is:

```text
downstream_evaluation/manifests/downstream_real_training_manifest.csv
```

It contains:

```text
41,460 training slices
332 BraTS training subjects
11,975 tumor-positive slices
29,485 tumor-free slices
```

The 10,000 donor slices reserved for the BR-LoRA synthetic-library design do
not overlap with this real training manifest.

## Frozen BraTS Validation Data

The validation manifest is:

```text
downstream_evaluation/manifests/downstream_validation_manifest.csv
```

It contains:

```text
5,735 validation slices
37 held-out BraTS subjects
2,447 tumor-positive slices
3,288 tumor-free slices
```

The same validation cohort is used for all three training regimes.

## Synthetic Augmentation Design

The frozen synthetic-library design is:

```text
downstream_evaluation/manifests/br_lora_library_design_10000/
    br_lora_library_design_10000.csv
```

It contains 10,000 synthetic cases constructed from the training-side BraTS
data contract.

The posterior-mean experiment contributes one fixed accepted synthetic image
per design case.

The posterior-sampling experiment uses the same design cases but selects among
retained BR-LoRA posterior realizations. The training dataset updates its
posterior realization assignment by epoch while preserving the frozen case
design.

For details of synthetic-library construction and acceptance, see
[`synthetic_library.md`](synthetic_library.md).

## Training-Set Sizes

The implemented training-set sizes are:

```text
Real only:
    41,460 samples per epoch

Real + BR-LoRA posterior mean:
    41,460 real + 10,000 synthetic
    = 51,460 samples per epoch

Real + BR-LoRA posterior sampling:
    41,460 real + 10,000 synthetic
    = 51,460 samples per epoch
```

## Segmentation Model

The downstream model is a 2D U-Net adapted for single-channel FLAIR tumor
segmentation.

The implemented architecture follows a standard encoder-decoder U-Net with
skip connections and feature widths:

```text
64 -> 128 -> 256 -> 512
```

The model uses:

```text
input: single-channel FLAIR
output: one segmentation logit channel
upsampling: bilinear
align_corners: true
```

The implementation was adapted and rewritten using the following repository as
a reference for the downstream segmentation task:

https://github.com/edaaydinea/Low-Grade-Glioma-Segmentation

The repository-specific implementation was modified for the BraTS data
contract, BR-LoRA augmentation experiments, deterministic training controls,
and UCSF-PDGM external evaluation.

## Training Objective

Segmentation training uses the sum of:

```text
BCEWithLogitsLoss
+
Dice loss
```

Dice loss is calculated after applying the sigmoid function to the model
logits.

The implemented Dice smoothing constant is:

```text
1
```

The optimizer is:

```text
Adamax
learning rate = 0.001
```

Training runs for:

```text
20 epochs
```

unless explicitly changed through configuration.

## Data Augmentation

Training images use the implemented Albumentations transform pipeline:

```text
HorizontalFlip(p=0.5)
VerticalFlip(p=0.5)
RandomRotate90(p=0.5)
Transpose(p=0.5)
ToTensorV2()
```

Validation uses only:

```text
ToTensorV2()
```

Transform randomness is seeded as part of the downstream reproducibility
controls.

## Segmentation Threshold

Probability maps are converted to binary segmentations using:

```text
threshold = 0.5
```

The threshold is stored with the trained checkpoint and is also validated by
the external evaluator.

## Internal Validation Metrics

During training, segmentation performance is evaluated on the frozen BraTS
validation cohort.

Dice and intersection over union (IoU) are calculated for:

- all validation slices; and
- tumor-positive validation slices only.

The best checkpoint is selected using tumor-positive validation Dice.

Empty-prediction/empty-target slices receive Dice and IoU equal to 1.

## Reproducible Training

The unified downstream trainer applies explicit controls for Python, NumPy,
PyTorch, CUDA, data-loader generators, worker initialization, and
Albumentations transform randomness.

When strict reproducibility is enabled, the CUDA training path additionally
uses deterministic PyTorch algorithms and the configured cuBLAS workspace
setting.

The shared reproducibility implementation is:

```text
downstream_evaluation/segmentation/reproducibility.py
```

The downstream configuration is:

```text
downstream_evaluation/configs/segmentation.yaml
```

For additional provenance and reproducibility details, see
[`reproducibility.md`](reproducibility.md).

## External Validation

The three trained segmentation models are evaluated on the same frozen
independent UCSF-PDGM cohort.

The external cohort contains:

```text
202 baseline subjects
```

Subjects overlapping the BraTS 2021 segmentation cohort and follow-up
examinations are excluded by the frozen cohort definition.

External evaluation uses UCSF-PDGM FLAIR images and whole-tumor targets defined
as:

```text
segmentation > 0
```

The evaluator reports:

- slice-level Dice and IoU across all slices;
- slice-level Dice and IoU across tumor-positive slices;
- subject-level mean slice metrics; and
- true 3D subject-level volumetric Dice and IoU.

For acquisition, cohort derivation, preprocessing, validation, and evaluator
usage, see
[`ucsf_pdgm_external_validation.md`](ucsf_pdgm_external_validation.md).

## Experimental Isolation

The following are kept fixed across the three downstream regimes:

- real BraTS training manifest;
- held-out BraTS validation manifest;
- segmentation architecture;
- loss function;
- optimizer;
- learning rate;
- number of epochs;
- data augmentation;
- segmentation threshold; and
- evaluation metrics.

The augmentation source is the experimental factor:

```text
no synthetic augmentation
vs.
BR-LoRA posterior-mean augmentation
vs.
BR-LoRA posterior-sampling augmentation
```

This design makes the downstream comparison interpretable as an evaluation of
the effect of BR-LoRA synthetic augmentation on segmentation performance.

## Outputs and Provenance

New downstream training runs are written to non-overwriting run directories
under:

```text
outputs/downstream_segmentation/
```

Run metadata records the configuration, manifests and hashes, software
environment, hardware and Slurm information when available, Git commit, and
reproducibility settings.

Large checkpoints and generated runtime outputs are not distributed through
Git.

Repository-controlled manifests, selected logs, and summary files are retained
under `downstream_evaluation/` where appropriate.
