# Localized Medical Image Synthesis

A modular research framework for localized medical image synthesis, regional
composition, parameter-efficient adaptation, Bayesian Regional LoRA (BR-LoRA),
and image-level reliability assessment.

> **Project Status:** Active development. The original baseline notebook has
> been refactored into modular Python components and validated against the
> reference implementation. Current development focuses on extending the
> validated baseline without changing its default behavior.

---

## Overview

This project investigates localized lesion synthesis using conditional diffusion
models together with parameter-efficient adaptation methods.

The framework explicitly separates

- tumor-free base anatomy,
- donor pathological appearance,
- prescribed lesion masks, and
- regional image composition.

The long-term research goals include

- localized lesion synthesis,
- regional image composition,
- parameter-efficient fine-tuning,
- Bayesian Regional LoRA (BR-LoRA),
- predictive uncertainty estimation,
- topology-aware structural analysis,
- image-level reliability assessment, and
- reproducible benchmarking of adaptation strategies.

---

## Current Status

The repository now contains a validated modular implementation of the baseline
patch-conditioned x0 diffusion workflow.

Completed infrastructure includes

- BraTS 2020 training-dataset registration,
- BraTS 2020 validation-dataset registration,
- machine-readable dataset specifications,
- reproducible reconstruction of the historical H5 dataset,
- slice-level manifest generation,
- modular data loading and sampling,
- modular diffusion scheduling,
- modular baseline U-Net implementation,
- modular training and checkpointing,
- reconstruction inference,
- tumor-free base / donor-mask pair selection, and
- localized regional composition.

The modular implementation has been checked directly against the original
notebook for exact numerical equivalence across the core workflow.

---

## Baseline Equivalence Validation

Explicit equivalence checks have passed for

- dataset filtering and train/validation splitting,
- diffusion schedule construction and q-sampling,
- model parameterization and seeded initialization,
- forward inference,
- masked training loss,
- gradients,
- AdamW parameter updates,
- optimizer state,
- epoch-level training and validation losses,
- best and final checkpoint contents,
- reconstruction inference,
- candidate discovery and seeded pair selection, and
- final localized composition.

The baseline notebook remains the scientific reference implementation, while
the modular code is now the primary development surface.

---

## Repository Organization

```text
localized-medical-image-synthesis/
│
├── checkpoints/          Saved model checkpoints
├── configs/              Experiment configuration files
├── data/                 Dataset documentation
├── docs/                 Extended project documentation
├── notebooks/            Original research notebooks
├── outputs/              Generated figures and synthesized images
├── scripts/              Executable workflows and utilities
├── src/                  Modular Python implementation
│   ├── data/
│   ├── diffusion/
│   ├── inference/
│   ├── models/
│   └── training/
│
├── README.md
├── PROJECT_PLAN.md
├── LICENSE
└── .gitignore
```

---

## Dataset Pipeline

The repository operates from the official BraTS 2020 NIfTI releases.

Training-data preparation:

```text
Raw BraTS Training NIfTI
        │
        ▼
register_dataset.py
        │
        ▼
dataset.yaml
        │
        ▼
build_h5_dataset.py
        │
        ▼
57,195 reconstructed H5 slices
        │
        ▼
create_dataset_manifest.py
        │
        ▼
manifest.csv
```

The official BraTS 2020 validation release is registered independently:

```text
Raw BraTS Validation NIfTI
        │
        ▼
register_validation_dataset.py
        │
        ▼
validation_dataset.yaml
```

The official validation release does not contain segmentation masks. It
therefore cannot directly replace the baseline masked validation loss used for
checkpoint selection. It is reserved for downstream inference and evaluation
protocols that do not require ground-truth segmentation.

See `data/README.md` for dataset setup details.

---

## Training Split Modes

The baseline configuration preserves the notebook's original behavior:

```yaml
data:
  split_mode: internal
  train_fraction: 0.9
```

This uses 90% of eligible tumor-containing training slices for optimization and
10% for internal validation.

A second mode is available for fixed-epoch training on all eligible training
slices:

```yaml
data:
  split_mode: full_train
```

`full_train` uses 100% of the eligible BraTS training slices and does not compute
the notebook's masked validation loss. This mode is intended for final
fixed-epoch refits after model-selection settings have already been established.

The official BraTS validation release remains separate from both modes.

---

## Repository Design Principles

- Preserve the original notebook as the reference implementation.
- Preserve notebook behavior as the default configuration.
- Expose legitimate experiment parameters rather than hard-coding them.
- Refactor incrementally and validate every extracted component.
- Avoid implicit dataset discovery and hidden preprocessing assumptions.
- Separate reusable library code from executable scripts.
- Perform expensive dataset validation once and reuse generated metadata.
- Do not silently change scientific behavior during engineering refactors.

---

## Current User-Facing Scripts

```text
scripts/register_dataset.py
scripts/register_validation_dataset.py
scripts/build_h5_dataset.py
scripts/create_dataset_manifest.py
scripts/train_patch_x0.py
scripts/synthesize_patch_x0.py
```

The historical reverse-engineering utility

```text
scripts/verify_h5_conversion.py
```

is retained for preprocessing audit purposes.

---

## Development Roadmap

See `PROJECT_PLAN.md` for the detailed engineering and research roadmap.

---

## Citation

If you use this repository in academic work, please cite

- the BraTS challenge and dataset, and
- this repository once publicly released.

---

## License

This repository is distributed under the terms of the license provided in
`LICENSE`.
