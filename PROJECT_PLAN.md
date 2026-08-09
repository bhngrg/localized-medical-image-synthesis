# Localized Medical Image Synthesis
## Repository Development Roadmap

This document describes the engineering roadmap for the repository.

The objective is to transform the original research notebook into a modular,
reproducible, and extensible Python framework while preserving identical
behavior throughout the refactoring process.

The notebook under `notebooks/` remains the reference implementation until each
component has been independently validated.

---

# Repository Development Philosophy

Development follows a staged approach.

Each phase must satisfy two requirements before progressing:

1. Functional equivalence with the corresponding notebook implementation.
2. Independent validation using dedicated verification utilities.

New functionality is introduced **only after** the baseline implementation has
been reproduced.

---

# Guiding Principles

The repository is developed according to the following principles.

- Preserve the original notebook as the reference implementation.
- Refactor incrementally rather than rewriting everything at once.
- Validate every extracted module independently.
- Assign each module a single, well-defined responsibility.
- Separate reusable library code from executable scripts.
- Prefer explicit implementations over unnecessary abstractions.
- Maintain reproducibility throughout development.

---

# Current Status

**Current phase**

✅ Phase 0 — Repository Foundation

Completed infrastructure

- Repository organization
- Documentation
- Dataset registration
- Dataset validation
- Historical H5 reconstruction
- Slice-level manifest generation
- Reconstruction verification
- Validation dataset registration

The repository now has a fully reproducible dataset pipeline that recreates the
historical preprocessing directly from the official BraTS 2020 NIfTI releases.

---

# Phase 0 — Repository Foundation

**Status:** ✅ Complete

## Goal

Establish the repository structure and reproducible dataset pipeline before
refactoring model code.

## Completed

- Git repository initialized
- Project directory structure established
- Repository documentation
- Dataset documentation
- Training dataset registration
- Validation dataset registration
- Historical H5 reconstruction
- Dataset manifest generation
- Historical reconstruction verification

---

# Phase 1 — Baseline Notebook Refactoring

**Status:** 🚧 Active

## Goal

Refactor the original notebook into reusable Python modules while preserving
identical behavior.

## Planned Components

- Data loading
- Dataset utilities
- Image preprocessing
- Diffusion utilities
- Model implementation
- Training pipeline
- Inference pipeline

## Validation

Every extracted component must reproduce the corresponding notebook behavior
before the notebook implementation is retired.

---

# Phase 2 — Regional Composition Framework

**Status:** ⏳ Planned

## Goal

Separate regional composition from the diffusion backbone so that localized
composition becomes an independent component usable by any adaptation strategy.

## Planned Components

- Base-image handling
- Donor-image handling
- Lesion-mask handling
- Regional composition utilities
- Common synthesis interface

---

# Phase 3 — Parameter-Efficient Adaptation

**Status:** ⏳ Planned

## Goal

Provide a unified implementation of multiple parameter-efficient fine-tuning
strategies using a common training and evaluation framework.

## Planned Methods

- Frozen backbone
- Full fine-tuning
- BitFit
- LoRA
- DoRA
- LoKr
- Regional LoRA

---

# Phase 4 — Bayesian Regional LoRA (BR-LoRA)

**Status:** ⏳ Planned

## Goal

Develop Bayesian Regional LoRA for uncertainty-aware localized medical image
synthesis.

## Planned Components

- Bayesian low-rank adapters
- Variational optimization
- Posterior sampling
- Predictive uncertainty estimation

---

# Phase 5 — Reliability Framework

**Status:** ⏳ Planned

## Goal

Develop an image-level reliability assessment framework for localized synthesis.

## Planned Components

- Predictive uncertainty
- Structural consistency analysis
- Topological analysis
- Repeat stability
- Controlled perturbation analysis
- Continuous review prioritization

---

# Phase 6 — Benchmarking and Evaluation

**Status:** ⏳ Planned

## Goal

Provide a unified evaluation framework for comparing localized synthesis methods
under identical experimental conditions.

## Planned Components

- Accuracy metrics
- Efficiency metrics
- Reliability metrics
- Visualization
- Statistical summaries
- Reproducible experiment configurations

---

# Validation Strategy

Validation accompanies every development phase.

Current verification utilities include

- raw dataset validation
- validation dataset registration
- historical H5 reconstruction verification
- MRI channel verification
- segmentation verification
- slice manifest verification

Additional verification procedures will be added as new modules are
implemented.

---

# Notebook-to-Module Mapping

This section will be updated as notebook functionality is migrated into the
modular implementation.

| Notebook Component | Modular Component | Status |
|--------------------|------------------|--------|
| Dataset preparation | `src/data/` | 🚧 |
| Image preprocessing | `src/preprocessing/` | ⏳ |
| Diffusion utilities | `src/diffusion/` | ⏳ |
| Model implementation | `src/models/` | ⏳ |
| Training | `src/training/` | ⏳ |
| Inference | `src/inference/` | ⏳ |
| Evaluation | `src/evaluation/` | ⏳ |

---

# Long-Term Vision

The completed repository will provide a reproducible research framework for

- localized medical image synthesis,
- regional image composition,
- parameter-efficient adaptation,
- Bayesian Regional LoRA,
- uncertainty estimation,
- topology-aware reliability assessment,
- reproducible benchmarking, and
- future extension to additional medical imaging datasets.

Every component will be independently testable, reusable, and validated against
the original research implementation before new functionality is introduced.