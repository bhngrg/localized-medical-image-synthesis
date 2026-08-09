# Localized Medical Image Synthesis

A modular research framework for localized medical image synthesis, regional
composition, parameter-efficient adaptation, Bayesian Regional LoRA (BR-LoRA),
and image-level reliability assessment.

The repository is being developed through incremental refactoring of the
original research notebook into a reproducible, modular Python codebase while
preserving identical functionality at every stage.

---

# Overview

This project investigates **localized lesion synthesis** using conditional
diffusion models together with parameter-efficient adaptation methods.

Unlike conventional image synthesis pipelines, the framework explicitly
separates

- tumor-free base anatomy,
- donor pathological appearance,
- prescribed lesion masks, and
- regional image composition,

allowing only the specified lesion region to be synthesized while preserving the
remaining anatomy by construction.

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

# Current Status

**Current development phase**

✅ Repository foundation complete

Current implemented infrastructure includes

- BraTS 2020 dataset registration
- automatic dataset validation
- historical H5 reconstruction
- dataset manifest generation
- reconstruction verification against the historical preprocessing

The next development phase focuses on refactoring the original notebook into a
modular Python implementation.

---

# Repository Organization

```text
localized-medical-image-synthesis/
│
├── checkpoints/          Saved model checkpoints
├── configs/              Experiment configuration files
├── data/                 Dataset documentation
├── notebooks/            Original research notebooks
├── outputs/              Generated figures and synthesized images
├── scripts/              Executable utilities and pipelines
├── src/                  Modular Python implementation
│
├── README.md
├── PROJECT_PLAN.md
└── .gitignore
```

---

# Dataset Pipeline

The repository operates directly from the official BraTS 2020 NIfTI releases.

The preprocessing workflow is

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
57,195 historical H5 slices
        │
        ▼
create_dataset_manifest.py
        │
        ▼
manifest.csv
```

Validation data are registered independently using

```text
register_validation_dataset.py
```

Further details are provided in

```text
data/README.md
```

---

# Repository Design Principles

Development follows several guiding principles.

- Preserve the original notebook as the reference implementation.
- Refactor incrementally while maintaining identical behavior.
- Validate every extracted module before introducing new functionality.
- Assign every module a single, well-defined responsibility.
- Prefer explicit, readable implementations over unnecessary abstractions.
- Separate reusable library code from executable scripts.
- Reproduce historical preprocessing exactly before extending the framework.

---

# Development Roadmap

Repository development proceeds through six major phases.

1. Repository foundation
2. Baseline implementation refactoring
3. Regional composition framework
4. Parameter-efficient adaptation
5. Bayesian Regional LoRA (BR-LoRA)
6. Image-level reliability assessment and benchmarking

A detailed roadmap is available in

```text
PROJECT_PLAN.md
```

---

# Current Verification Status

The repository currently includes verification procedures for

- raw dataset registration,
- H5 reconstruction,
- MRI channel ordering,
- segmentation channel ordering,
- slice-level manifest generation, and
- historical preprocessing reproducibility.

These tests ensure that the modular implementation reproduces the original
research pipeline before additional functionality is introduced.

---

# Planned Components

The completed framework will include reusable modules for

- dataset management,
- diffusion utilities,
- localized regional composition,
- parameter-efficient adaptation methods,
- Bayesian Regional LoRA (BR-LoRA),
- training and inference pipelines,
- image-level reliability assessment, and
- reproducible benchmarking.

---

# Citation

If you use this repository in academic work, please cite

- the BraTS challenge and dataset, and
- this repository (once publicly released).

---

# License

This repository is distributed under the terms of the license provided in
`LICENSE`.