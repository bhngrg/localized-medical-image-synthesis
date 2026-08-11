# Localized Medical Image Synthesis

A modular research framework for localized medical image synthesis, regional
composition, parameter-efficient adaptation, Bayesian Regional LoRA (BR-LoRA),
and image-level reliability assessment.

> **Project Status**: Active development. The original baseline notebook has been 
> refactored into modular Python components and validated against the reference 
> implementation. The repository now contains validated end-to-end BR-LoRA training 
> and inference infrastructure supporting both internal (90/10) model-selection 
> workflows and full-training refits. Current development focuses on evaluation, 
> uncertainty estimation, reliability assessment, and benchmarking across adaptation 
> strategies.

---

## Overview

This project investigates localized lesion synthesis using conditional diffusion
models together with parameter-efficient adaptation methods.

The framework explicitly separates

- tumor-free base anatomy,
- donor pathological appearance,
- prescribed lesion masks, and
- regional image composition.

The repository currently supports

- localized lesion synthesis,
- hard regional image composition,
- deterministic convolutional LoRA,
- Bayesian Regional LoRA (BR-LoRA),
- mean-field Gaussian variational posteriors,
- reparameterized posterior sampling,
- posterior-mean evaluation,
- parameter-normalized KL regularization,
- multi-epoch BR-LoRA training,
- checkpointing and resume support,
- posterior inference, and
- internal and full-training workflows.

Ongoing development is focused on

- predictive uncertainty estimation,
- topology-aware structural analysis,
- image-level reliability assessment,
- common benchmarking across adaptation strategies, and
- external evaluation protocols.

---

## Current Status

The repository contains a validated modular implementation of the baseline
patch-conditioned x0 diffusion workflow together with independently validated
BR-LoRA learning infrastructure.

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
- tumor-free base / donor-mask pair selection,
- localized regional composition,
- fixed-epoch training on all eligible training slices,
- deterministic convolutional LoRA,
- mean-field Gaussian variational parameters,
- Bayesian Regional LoRA adapters,
- posterior sampling and posterior-mean modes,
- analytic KL divergence,
- BR-LoRA variational optimization,
- multi-epoch BR-LoRA training,
- checkpoint orchestration,
- resume-training support,
- training-history serialization,
- posterior-mean inference,
- posterior-sampled inference,
- internal (90/10) BR-LoRA training,
- full-training BR-LoRA workflow, and
- retained experiment logs.

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

## Bayesian Regional LoRA Validation

The BR-LoRA infrastructure has been validated independently on the same local
`AppearanceX0UNet` backbone used by the baseline workflow.

Completed checks include

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
- sample-weighted metric aggregation, and
- deterministic posterior-mean validation with zero parameter changes.
- validated multi-epoch training,
- checkpoint save/load,
- resume-training restoration,
- posterior inference reconstruction,
- full-training workflow, and
- split-mode-aware training orchestration.

For the currently validated rank-4, alpha-8 configuration, the seven adapted
convolutional layers contain 18,052 deterministic LoRA parameters and 36,104
variational posterior parameters across 28 trainable posterior tensors.

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
├── data/                 Dataset documentation
├── docs/                 Extended project documentation
├── logs/                 Retained training/runtime logs
├── notebooks/            Original research notebooks
├── outputs/              Generated figures and synthesized images
├── scripts/              Executable workflows and utilities
├── src/                  Modular Python implementation
│   ├── data/
│   ├── diffusion/
│   ├── inference/
│   ├── models/
│   │   └── adapters/
│   │       ├── base.py
│   │       ├── lora.py
│   │       ├── selection.py
│   │       ├── variational.py
│   │       └── variational_lora.py
│   └── training/
│       ├── br_lora_objectives.py
│       ├── br_lora_step.py
│       ├── br_lora_trainer.py
│       ├── losses.py
│       ├──trainer.py
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

The current baseline dataset contains 19,941 eligible tumor-containing H5
slices after applying the validated slice-selection criteria.

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

The repository has completed

- a 50-epoch internal (90/10) baseline run,
- a 50-epoch internal BR-LoRA run,
- a 50-epoch full-training baseline run, and
- a 50-epoch full-training BR-LoRA run.

Training logs for these reference experiments are retained under `logs/`.

The official BraTS validation release remains separate from both modes.

---

## Bayesian Regional LoRA

BR-LoRA is applied on top of the validated local `AppearanceX0UNet` rather than
replacing the backbone implementation.

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

Each LoRA factor is represented by a trainable diagonal-Gaussian posterior. The
model can therefore operate in either of two modes:

```yaml
sample_posterior: true
```

for reparameterized Bayesian realizations, or

```yaml
sample_posterior: false
```

for deterministic posterior-mean evaluation.

The fitted posterior is represented by the posterior mean and posterior
standard-deviation parameterization; individual posterior realizations do not
need to be stored in the training checkpoint to reproduce future sampling.

---

## BR-LoRA Variational Objective

The BR-LoRA training objective preserves the validated regional reconstruction
loss and adds normalized KL regularization.

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

The current default configuration uses linear KL warmup and posterior sampling
during variational training. Validation defaults to posterior-mean mode so
future checkpoint selection can use a stable deterministic criterion rather
than a single Monte Carlo realization.

The training infrastructure supports both notebook-equivalent internal
train/validation workflows for model selection and fixed-epoch full-training
workflows for final model fitting. Both modes share the same validated BR-LoRA
optimization pipeline while differing only in checkpoint-selection behavior.

---

## Repository Design Principles

- Preserve the original notebook as the reference implementation.
- Preserve notebook behavior as the default baseline configuration.
- Expose legitimate experiment parameters rather than hard-coding them.
- Refactor incrementally and validate every extracted component.
- Validate Bayesian components independently before end-to-end integration.
- Avoid implicit dataset discovery and hidden preprocessing assumptions.
- Separate reusable library code from executable scripts.
- Perform expensive dataset validation once and reuse generated metadata.
- Keep training, inference, composition, uncertainty, and evaluation concerns
  separate.
- Do not silently change scientific behavior during engineering refactors.

---

## Current User-Facing Scripts

```text
scripts/register_dataset.py
scripts/register_validation_dataset.py
scripts/build_h5_dataset.py
scripts/create_dataset_manifest.py
scripts/train_patch_x0.py
scripts/train_br_lora.py
scripts/synthesize_patch_x0.py
```

The historical reverse-engineering utility

```text
scripts/verify_h5_conversion.py
```

is retained for preprocessing audit purposes.

The BR-LoRA training CLI supports both notebook-equivalent internal training
and full-training workflows, including checkpointing, resume support, and
posterior-mean validation.

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
