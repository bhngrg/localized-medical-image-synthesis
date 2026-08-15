# Downstream Evaluation

## Purpose

The downstream evaluation layer consumes frozen model outputs and frozen synthetic-library manifests. It is designed to support segmentation-oriented experiments, image-level reliability analysis, and comparative evaluation of adaptation strategies.

## Inputs

The main downstream inputs are:

- registered BraTS 2020 training data;
- registered BraTS 2020 validation data;
- screened external base-image manifests;
- the downstream-training donor pool;
- frozen BR-LoRA checkpoints;
- the 10,000-case synthetic library design;
- retained BR-LoRA posterior artifacts.

## Current Manifest Infrastructure

Key repository-controlled manifests include:

```text
downstream_evaluation/manifests/
    brats_downstream_subject_split.csv
    brats_downstream_training_donor_pool.csv
    brats_real_catalog.csv
    brats_subject_level_summary.csv
    br_lora_synthetic_250.csv
    br_lora_library_design_10000/
```

Additional external-dataset manifests may be present for later downstream evaluation.

## Evaluation Directions

The planned downstream work includes:

- segmentation performance using real data;
- segmentation performance using synthetic augmentation;
- comparisons of real-only versus real-plus-synthetic training;
- predictive uncertainty summaries;
- topology-aware structural analysis;
- repeat-synthesis stability;
- controlled perturbation analyses;
- comparison across adaptation strategies.

## Separation from Screening

The nnU-Net model under `screening/brats_nnunet/` is used only to construct the external cohort. It is not the downstream segmentation model and should not be interpreted as a synthetic-image quality evaluator.

## Reproducibility

Downstream experiments should consume frozen manifests rather than rediscovering cases. Each experiment should record:

- input manifest checksum;
- model checkpoint;
- configuration;
- Git commit;
- random seed;
- environment;
- output location;
- evaluation summary.
