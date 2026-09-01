# BraTS nnU-Net External-Cohort Screening

This subtree contains the nnU-Net workflow used to construct the screened
BraTS 2020 validation cohort used by the BR-LoRA external-cohort and synthetic
library pipelines.

Its scientific role is limited to identifying validation slices with no
predicted whole-tumor involvement and auditing their compatibility with the
eligible donor pool. nnU-Net is not trained jointly with BR-LoRA, used during
BR-LoRA optimization, used to modify BR-LoRA predictions, or treated as ground
truth for synthesized-image quality.

## Role in the Project

```text
Registered BraTS 2020 Training + Validation
        │
        ▼
prepare_nnunet_dataset.py
        │
        ▼
Dataset500_BraTS2020Screening
        │
        ▼
Five-fold nnU-Net training
        │
        ▼
Five-fold validation ensemble prediction
        │
        ▼
screen_validation_slices.py
        │
        ▼
Compatibility and donor-morphology audits
        │
        ▼
Candidate-cohort design and matching
        │
        ▼
Frozen 250-case external cohort
        │
        ▼
Frozen 10,000-case BR-LoRA library design
        │
        ▼
Synthetic-library production
        │
        ▼
Downstream segmentation evaluation
```

The screening stage is an operational dependency of the implemented synthetic
library workflow because the official BraTS 2020 validation release does not
contain segmentation labels.

## 1. Prepare the nnU-Net Dataset

Both registered BraTS 2020 releases are required.

```bash
python screening/brats_nnunet/scripts/prepare_nnunet_dataset.py --help
python screening/brats_nnunet/scripts/prepare_nnunet_dataset.py --validate-only
```

The preparation script resolves the registered Training and Validation dataset
specifications from `data/folders.yaml` unless explicit CLI paths are supplied.

See [`../../data/README.md`](../../data/README.md) for BraTS acquisition and
registration.

## 2. Train the Five nnU-Net Folds

Production nnU-Net training is intended for a **CUDA-capable GPU**. Ordinary
CPU execution and Apple MPS are not the supported production path for this
stage.

The provided Slurm array workflow trains folds 0-4:

```text
training/train_all_folds.slurm
```

The five folds are independent and can run concurrently as cluster resources
permit.

See [`training/README.md`](training/README.md) for the repository-specific
training workflow and the
[official nnU-Net documentation](https://github.com/MIC-DKFZ/nnUNet) for
nnU-Net installation and general usage.

## 3. Predict the BraTS Validation Cohort

Production inference also requires a **CUDA-capable GPU**.

The provided five-fold ensemble workflow is:

```text
inference/predict_validation.slurm
```

It produces one whole-tumor prediction for each of the 125 registered BraTS
validation subjects.

See [`inference/README.md`](inference/README.md).

## 4. Screen Validation Slices

Inspect the user-facing interface with:

```bash
python screening/brats_nnunet/scripts/screen_validation_slices.py --help
python screening/brats_nnunet/scripts/screen_validation_slices.py --validate-only
```

The primary tumor-free criterion is:

```text
predicted_tumor_pixels == 0
```

The completed screening workflow identified 11,414 tumor-free validation
slices and 8,632 compatibility-eligible external bases across all 125
validation subjects.

## 5. Audit Compatibility and Freeze the External Cohort

After slice-level screening, the implemented workflow proceeds through:

```text
audit_external_pair_space.py
        ├──► analyze_external_pair_space.py
        │
        ▼
audit_donor_morphology.py
        │
        ▼
audit_external_manifest_design.py
        ├──► audit_external_matching_diagnostics.py
        ├──► audit_compatibility_conditioned_donor_selection.py
        │
        ▼
finalize_external_manifest.py
        │
        ▼
audit_definitive_compatibility_conditioned_donor_selection.py
```

These scripts resolve deterministic workflow locations under
`nnunet_run_root` when explicit paths are omitted. Their scientific matching
and audit parameters remain explicit CLI options.

The finalized external cohort contains:

- 250 cases,
- exactly two selected bases per validation subject,
- 250 unique donor slices, and
- deterministic compatibility-constrained donor assignments.

Use `--help` on any script to inspect its supported arguments.

## 6. Construct the Frozen 10,000-Case Library Design

The 10,000-case BR-LoRA design is downstream of the frozen screening and
matching outputs.

```bash
python screening/brats_nnunet/scripts/design_br_lora_library_10000.py --help
```

The canonical design is tracked under:

```text
downstream_evaluation/manifests/br_lora_library_design_10000/
```

`--output-dir` is intentionally required when reconstructing the design so the
tracked canonical artifacts cannot be overwritten accidentally.

Synthetic-library production itself is documented in
[`../../docs/synthetic_library.md`](../../docs/synthetic_library.md).

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
│   ├── analyze_external_pair_space.py
│   ├── audit_compatibility_conditioned_donor_selection.py
│   ├── audit_definitive_compatibility_conditioned_donor_selection.py
│   ├── audit_donor_morphology.py
│   ├── audit_external_manifest_design.py
│   ├── audit_external_matching_diagnostics.py
│   ├── audit_external_pair_space.py
│   ├── design_br_lora_library_10000.py
│   ├── finalize_external_manifest.py
│   ├── prepare_nnunet_dataset.py
│   └── screen_validation_slices.py
└── training/
    ├── README.md
    ├── train_all_folds.slurm
    └── train_fold.sh
```

## Reproducibility Requirements

For production runs, record the nnU-Net version, dataset identifier,
trainer/plans, fold checkpoints, Git commit, Slurm partition, prediction
provenance, screening thresholds, compatibility audits, definitive-manifest
checksum, and generated provenance metadata.

Generated nnU-Net datasets, checkpoints, and prediction masks remain outside
Git. Selected production and reproducibility-relevant execution logs are
retained under the canonical repository-level `logs/screening/brats_nnunet/`
tree.
