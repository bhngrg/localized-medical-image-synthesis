# Repository Architecture

## Purpose

The repository is organized around localized medical image synthesis using a validated patch-conditioned diffusion backbone, parameter-efficient adaptation, Bayesian Regional LoRA (BR-LoRA), external cohort screening, and downstream reliability-aware evaluation.

The original notebook remains the scientific reference implementation for the baseline workflow. The modular Python code is the primary implementation used for training, inference, synthetic library construction, and downstream experiments.

## Major Components

```text
localized-medical-image-synthesis/
├── configs/                  Experiment configuration files
├── data/                     Dataset setup documentation
├── docs/                     Extended technical documentation
├── logs/                     Versioned training/evaluation logs where appropriate
├── notebooks/                Historical/reference notebooks
├── outputs/                  Generated local outputs (not generally versioned)
├── screening/                External-cohort screening workflows
│   └── brats_nnunet/
├── downstream_evaluation/    Frozen manifests and downstream study infrastructure
├── scripts/                  Executable entry points and orchestration scripts
├── src/                      Reusable Python modules
├── README.md
├── PROJECT_PLAN.md
└── LICENSE
```

## Scientific Separation of Responsibilities

The repository keeps the following stages conceptually separate:

1. **Dataset registration and reconstruction**
   - BraTS 2020 training data are registered and reconstructed into the validated H5 representation.
   - The official BraTS 2020 validation release is registered independently.

2. **Baseline diffusion training**
   - The baseline patch-conditioned x0 diffusion model is trained under either internal 90/10 model-selection mode or fixed-epoch full-training mode.

3. **BR-LoRA adaptation**
   - Bayesian low-rank adapters are attached to the validated backbone.
   - The backbone remains frozen while variational LoRA parameters are optimized.

4. **External screening**
   - A separate five-fold nnU-Net model is trained and frozen.
   - It is used only to screen the official BraTS 2020 validation cohort for candidate tumor-free base slices.

5. **External BR-LoRA inference**
   - Fixed manifests define external bases and training donors.
   - Posterior realizations are generated with deterministic per-case seeding.

6. **Synthetic library production**
   - A frozen 10,000-case design determines base/donor pairings.
   - Cases are generated in 40 batches of 250.

7. **Downstream evaluation**
   - The completed synthetic library is used for segmentation and reliability-oriented analyses.

## Design Principles

- Preserve notebook-equivalent scientific behavior unless a change is explicit.
- Separate fixed data contracts from experiment settings.
- Prefer manifest-driven execution over hidden data discovery.
- Preserve provenance for production runs.
- Use deterministic seeds where case-level reproducibility is required.
- Keep screening independent of synthesis and model fitting.
- Keep large generated artifacts outside Git.
