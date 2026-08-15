# BraTS nnU-Net External-Cohort Screening

This subtree contains the nnU-Net workflow used to construct a screened
external BraTS 2020 validation cohort for localized medical image
synthesis evaluation.

The screening workflow is intentionally separate from BR-LoRA training,
inference, posterior analysis, and reliability assessment. Its sole
scientific purpose is to identify BraTS 2020 validation slices with no
predicted whole-tumor involvement so that they can serve as candidate
external base images.

## Role in the Project

``` text
BraTS 2020 training release
        │
        ▼
Dataset500_BraTS2020Screening
        │
        ▼
Five-fold nnU-Net training
        │
        ▼
Frozen screening model
        │
        ▼
BraTS 2020 validation release
        │
        ▼
Five-fold ensemble prediction
        │
        ▼
Predicted whole-tumor masks
        │
        ▼
Slice-level screening
        │
        ▼
Compatibility analysis
        │
        ▼
Donor-morphology audit
        │
        ▼
Nested candidate cohorts
        │
        ▼
Global one-to-one matching
        │
        ▼
Compatibility-conditioned audit
        │
        ▼
Frozen 250-case manifest
        │
        ▼
10,000-case BR-LoRA library design
        │
        ▼
Batch-wise synthetic library production
        │
        ▼
Downstream segmentation evaluation
```

The nnU-Net model is a cohort-screening tool only. It is not trained
jointly with BR-LoRA, used during BR-LoRA optimization, used to modify
BR-LoRA predictions, or treated as ground truth for synthesized-image
quality.

The frozen screening outputs also define the admissible external base-image
pool used by the BR-LoRA synthetic library production pipeline. All library
design manifests are generated downstream of this screening stage and do not
modify the screening results.

## Slice-Level Screening

The slice-level screening workflow has been fully implemented and
audited.

The primary tumor-free criterion mirrors the historical composition
pipeline:

``` text
predicted_tumor_pixels == 0
```

Screening additionally verifies compatibility with the donor pool using
the established brain-overlap and eligibility criteria before candidate
external bases are admitted to the evaluation cohort.

The resulting screened cohort is subjected to compatibility-space,
donor-morphology, compatibility-conditioned donor-selection,
deterministic matching, and definitive-manifest audits before external
BR-LoRA evaluation.

## Final External Evaluation Cohort

The completed screening workflow produced

-   125 validation subjects
-   11,414 tumor-free validation slices
-   8,632 compatibility-eligible external bases
-   nested candidate cohorts containing 125, 250, and 625 external cases
-   a definitive frozen evaluation cohort of 250 external cases
-   exactly two external bases per validation subject
-   250 unique donor slices
-   deterministic one-to-one compatibility-constrained donor matching
-   complete compatibility-conditioned donor-selection audits
-   complete provenance and reproducibility metadata


## Relationship to the BR-LoRA Library Pipeline

The screening workflow is executed once to produce a fixed, reproducible pool
of compatibility-eligible external base slices.

Subsequent BR-LoRA library generation consumes these frozen manifests but does
not rerun nnU-Net inference or alter the screened cohort. This separation
ensures that cohort selection and synthetic image generation remain
independently reproducible.


## Repository Layout

``` text
screening/brats_nnunet/
├── README.md
├── configs/
├── inference/
│   ├── README.md
│   └── predict_validation.slurm
├── manifests/
├── scripts/
│   ├── analyze_external_pair_space.py
│   ├── audit_compatibility_conditioned_donor_selection.py
│   ├── audit_definitive_compatibility_conditioned_donor_selection.py
│   ├── audit_donor_morphology.py
│   ├── audit_external_manifest_design.py
│   ├── audit_external_matching_diagnostics.py
│   ├── audit_external_pair_space.py
│   ├── finalize_external_manifest.py
│   ├── prepare_nnunet_dataset.py
│   └── screen_validation_slices.py
└── training/
    ├── README.md
    ├── train_all_folds.slurm
    └── train_fold.sh
```

## Reproducibility Requirements

Record the nnU-Net version, dataset identifier, trainer/plans, fold
checkpoints, Git commit, Falcon partition, prediction provenance,
screening thresholds, compatibility-space audit, donor-selection audit,
definitive manifest checksum, and all generated provenance metadata for
production runs.

Generated datasets, checkpoints, predictions, and runtime logs are not
stored in the Git repository.
