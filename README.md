# Localized Medical Image Synthesis

A modular research framework for localized medical image synthesis, regional
composition, parameter-efficient adaptation, Bayesian Regional LoRA (BR-LoRA),
and image-level reliability assessment.

> **Project Status**: Active development. The original baseline notebook has been
> refactored into modular Python components and validated against the reference
> implementation. The repository now contains validated end-to-end BR-LoRA
> training, posterior mean inference, and posterior sampling infrastructure
> supporting both internal (90/10) model-selection workflows and full-training
> refits. Internal posterior-sampling analyses, including Monte Carlo
> convergence and Monte Carlo standard error (MCSE) validation, have been
> implemented. Current development focuses on external validation,
> reliability assessment, and benchmarking across adaptation strategies.

---

## Overview

This project investigates donor-conditioned localized medical image synthesis
using conditional diffusion models together with parameter-efficient adaptation
methods.

The framework explicitly separates

- tumor-free base anatomy,
- donor pathological appearance,
- prescribed lesion masks, and
- hard regional image composition.

The repository currently supports

- localized lesion synthesis,
- hard regional image composition,
- parameter-efficient adaptation using BitFit, LoRA, DoRA, LoKr, and Bayesian Regional LoRA (BR-LoRA),
- mean-field Gaussian variational posteriors,
- reparameterized posterior sampling,
- posterior mean and posterior variance inference,
- parameter-normalized KL regularization,
- multi-epoch BR-LoRA training,
- checkpointing and resume support,
- Monte Carlo convergence analysis,
- Monte Carlo standard error (MCSE) analysis, and
- internal (90/10) and full-training workflows.

Ongoing development is focused on

- external validation of trained BR-LoRA models,
- topology-aware structural analysis,
- image-level reliability assessment,
- benchmarking across adaptation strategies, and
- AAAI manuscript experiments and figures.

---

## Current Status

The repository contains a validated modular implementation of the baseline
patch-conditioned x0 diffusion workflow together with Bayesian Regional LoRA
(BR-LoRA) training, inference, and posterior analysis infrastructure.

Completed infrastructure includes

- BraTS 2020 dataset registration and validation,
- machine-readable dataset specifications,
- reproducible reconstruction of the historical H5 dataset,
- slice-level manifest generation,
- modular data loading and sampling,
- modular diffusion scheduling,
- modular baseline U-Net implementation,
- modular training and checkpointing,
- reconstruction inference,
- tumor-free base / donor-mask pair selection,
- localized hard regional composition,
- fixed-epoch training on all eligible training slices,
- parameter-efficient adaptation using BitFit, LoRA, DoRA, LoKr, and BR-LoRA,
- mean-field Gaussian variational posteriors,
- analytic KL divergence,
- reparameterized variational optimization,
- posterior mean and posterior sampling inference,
- Monte Carlo convergence analysis,
- Monte Carlo standard error (MCSE) analysis,
- internal (90/10) BR-LoRA training,
- full-training BR-LoRA workflows,
- checkpoint management and resume support, and
- experiment logging and training-history serialization.

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
- candidate discovery and seeded pair selection,
- localized hard regional composition, and
- BR-LoRA pair-preparation consistency across synthesis and posterior inference workflows.

The baseline notebook remains the scientific reference implementation, while
the modular repository serves as the primary development and experimentation
platform.

---

## Bayesian Regional LoRA Validation

The BR-LoRA infrastructure has been validated independently on the same local
`AppearanceX0UNet` backbone used by the baseline workflow.

Completed validation includes

- exact seven-layer target selection,
- deterministic LoRA parameter accounting,
- variational posterior parameter accounting,
- frozen-backbone preservation,
- exact zero-update initialization of fresh LoRA,
- exact posterior-mean equivalence at initialization,
- seeded posterior sampling reproducibility,
- distinct realizations under distinct posterior seeds,
- analytic KL divergence,
- KL gradients through all posterior tensors,
- exact reconstruction-loss equivalence to the validated baseline,
- parameter-normalized KL regularization,
- linear KL warmup,
- complete BR-LoRA variational objective,
- real-H5 single-step optimization,
- gradient clipping,
- optimizer updates restricted to posterior parameters,
- epoch-level BR-LoRA training,
- global-step bookkeeping,
- sample-weighted metric aggregation,
- validated multi-epoch training,
- checkpoint save/load,
- resume-training restoration,
- posterior mean inference,
- posterior sampling inference,
- posterior convergence analysis,
- Monte Carlo standard error (MCSE) analysis,
- shared pair-preparation consistency across synthesis and posterior inference,
- full-training workflows, and
- split-mode-aware training orchestration.

For the currently validated rank-4, alpha-8 configuration, the seven adapted
convolutional layers contain 18,052 deterministic LoRA parameters and 36,104
variational posterior parameters distributed across 28 trainable posterior
tensors.

---

## Repository Organization

```text
localized-medical-image-synthesis/
│
├── checkpoints/          Saved model checkpoints
├── configs/              Experiment configuration files
│   ├── baseline_patch_x0.yaml
│   ├── baseline_patch_x0_full_train.yaml
│   └── br_lora.yaml
│
├── data/                 Dataset documentation
├── docs/                 Extended project documentation
├── logs/                 Training and runtime logs
├── notebooks/            Original research notebooks
├── outputs/              Generated figures, analyses, and synthesized images
├── scripts/              Executable workflows
│   ├── train_baseline.py
│   ├── train_br_lora.py
│   ├── synthesize_br_lora.py
│   ├── audit_br_lora_posterior.py
│   ├── analyze_br_lora_posterior_convergence.py
│   └── analyze_br_lora_posterior_mcse.py
│
├── src/                  Modular Python implementation
│   ├── data/
│   ├── diffusion/
│   ├── inference/
│   │   └── br_lora_pairs.py
│   ├── models/
│   │   └── adapters/
│   │       ├── base.py
│   │       ├── lora.py
│   │       ├── selection.py
│   │       ├── variational.py
│   │       └── variational_lora.py
│   └── training/
│       ├── losses.py
│       ├── trainer.py
│       ├── br_lora_objectives.py
│       ├── br_lora_step.py
│       ├── br_lora_trainer.py
│       ├── br_lora_fit.py
│       └── br_lora_checkpoint.py
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
Raw BraTS 2020 Training NIfTI
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

The reconstructed training dataset contains 57,195 H5 slices, of which 19,941
tumor-containing slices satisfy the validated slice-selection criteria used by
the synthesis pipeline.

The official BraTS 2020 validation release is registered independently:

```text
Raw BraTS 2020 Validation NIfTI
        │
        ▼
register_validation_dataset.py
        │
        ▼
validation_dataset.yaml
```

The official validation release does not include tumor segmentation masks.
Consequently, it is not used for masked reconstruction losses or checkpoint
selection. Instead, it serves as an independent dataset for downstream
inference, uncertainty estimation, and reliability evaluation.

See `data/README.md` for dataset setup and registration details.

---

## Training Split Modes

The baseline configuration preserves the original notebook behavior:

```yaml
data:
  split_mode: internal
  train_fraction: 0.9
```

This mode uses 90% of the eligible tumor-containing training slices for
optimization and reserves the remaining 10% for internal model selection.

A second mode supports fixed-epoch training on the complete eligible training
dataset:

```yaml
data:
  split_mode: full_train
```

`full_train` uses 100% of the eligible BraTS training slices and omits the
masked validation loss used during internal model selection. It is intended for
final model refits after hyperparameters and training settings have been
established.

The repository currently includes validated reference runs for

- 50-epoch internal (90/10) baseline training,
- 50-epoch internal (90/10) BR-LoRA training,
- 50-epoch full-training baseline, and
- 50-epoch full-training BR-LoRA.

Training logs for these reference experiments are retained under `logs/`.

The official BraTS 2020 validation release remains independent of both training
modes and is reserved for downstream inference and evaluation.

---

## Bayesian Regional LoRA

Bayesian Regional LoRA (BR-LoRA) extends the validated local
`AppearanceX0UNet` through Bayesian low-rank adaptation while leaving the
backbone architecture unchanged.

The current configuration adapts seven convolutional layers:

```yaml
br_lora:
  target_layers:
    - enc1.conv2
    - enc2.conv2
    - enc3.conv2
    - mid.conv2
    - dec2.conv2
    - dec1.conv2
    - out

  rank: 4
  alpha: 8.0
  dropout: 0.0

  initial_std: 0.01
  prior_mean: 0.0
  prior_std: 1.0
  minimum_std: 1.0e-8
```

Each LoRA factor is represented by a trainable mean-field Gaussian posterior.
During inference, the model can operate in either of two modes:

```yaml
sample_posterior: true
```

to generate reparameterized posterior realizations, or

```yaml
sample_posterior: false
```

to perform deterministic posterior-mean inference.

Training checkpoints store the posterior mean and posterior standard-deviation
parameters for every adapted LoRA factor. Individual posterior realizations are
generated on demand during inference and therefore do not need to be stored.

The repository additionally provides utilities for posterior-sampling audits,
Monte Carlo convergence analysis, and Monte Carlo standard error (MCSE)
analysis to validate BR-LoRA posterior inference.

---

## BR-LoRA Variational Objective

BR-LoRA preserves the validated regional reconstruction objective while adding
parameter-normalized KL regularization over the variational posterior.

Conceptually,

```text
reconstruction
    = inside-mask L1
    + outside_weight * outside-mask L1

normalized_kl
    = KL(q || p) / number_of_variational_parameters

total
    = reconstruction
    + kl_weight
      * warmup_multiplier
      * normalized_kl
```

The default configuration performs posterior sampling during variational
training together with linear KL warmup. Validation is performed in
posterior-mean mode so that checkpoint selection is based on a deterministic
criterion rather than a single posterior realization.

The same optimization pipeline is used for both supported training modes:

- **internal (90/10)** for model selection using the masked validation loss, and
- **full_train** for fixed-epoch refitting on all eligible training slices after
  model-selection settings have been established.

The two modes differ only in checkpoint-selection behavior; all BR-LoRA
optimization, posterior parameterization, and variational training components
are shared.

---

## Repository Design Principles

- Preserve the original notebook as the scientific reference implementation.
- Preserve notebook behavior as the default baseline configuration.
- Expose legitimate experiment parameters rather than hard-coding them.
- Refactor incrementally and validate every extracted component.
- Validate Bayesian components independently before end-to-end integration.
- Avoid implicit dataset discovery and hidden preprocessing assumptions.
- Separate reusable library code from executable workflows.
- Perform expensive dataset validation once and reuse generated metadata.
- Separate training, inference, posterior analysis, and evaluation into modular components.
- Share common functionality across workflows rather than duplicating implementations.
- Do not silently change scientific behavior during engineering refactors.

---

## Current User-Facing Scripts

```text
Dataset preparation
-------------------
scripts/register_dataset.py
scripts/register_validation_dataset.py
scripts/build_h5_dataset.py
scripts/create_dataset_manifest.py

Model training
--------------
scripts/train_patch_x0.py
scripts/train_br_lora.py

Image synthesis
---------------
scripts/synthesize_patch_x0.py
scripts/synthesize_br_lora.py

Posterior analysis
------------------
scripts/audit_br_lora_posterior.py
scripts/analyze_br_lora_posterior_convergence.py
scripts/analyze_br_lora_posterior_mcse.py
```

The historical reverse-engineering utility

```text
scripts/verify_h5_conversion.py
```

is retained for preprocessing audit purposes.

The BR-LoRA training and inference workflows support

- notebook-equivalent internal (90/10) training,
- fixed-epoch full-training refits,
- checkpoint save/load and resume,
- posterior mean inference,
- posterior sampling inference,
- posterior convergence analysis, and
- Monte Carlo standard error (MCSE) analysis.

---

## Development Roadmap

See `PROJECT_PLAN.md` for the detailed engineering and research roadmap.

---

## Citation

If you use this repository in academic work, please cite

- the BraTS challenge and dataset,
- any third-party software incorporated into your work, and
- this repository once it is publicly released.

---

## License

This repository is distributed under the terms of the license provided in
`LICENSE`.
