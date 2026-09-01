# Repository Architecture

## Purpose

The repository is organized around localized medical image synthesis using a
validated patch-conditioned diffusion backbone, Bayesian Regional LoRA
(BR-LoRA), nnU-Net-based external-cohort screening, frozen synthetic-library
construction, and downstream segmentation evaluation.

The original notebook remains the scientific reference implementation for the
baseline synthesis workflow. The modular Python code is the primary executable
implementation used for training, inference, screening, synthetic-library
production, and downstream experiments.

## Major Components

```text
localized-medical-image-synthesis/
├── checkpoints/              Persistent model state; ignored by Git
├── configs/                  Experiment configuration files
├── data/                     Dataset setup and machine-path documentation
├── docs/                     Extended technical documentation
├── downstream_evaluation/    Frozen manifests and downstream evaluation code
├── logs/                     Selected execution provenance logs
├── notebooks/                Historical/reference notebooks
├── outputs/                  Raw and generated workflow products
├── results/                  Curated scientific results
├── screening/
│   └── brats_nnunet/         nnU-Net screening and compatibility workflow
├── scripts/                  User-facing executable workflows
├── src/                      Reusable Python implementation
├── README.md
└── LICENSE
```

Large generated datasets, checkpoints, predictions, posterior tensors, and
synthetic images are stored outside Git.

## Scientific Separation of Responsibilities

The implemented workflow keeps the following stages separate.

1. **BraTS registration and reconstruction**
   - BraTS 2020 Training and Validation are registered independently.
   - Training data are reconstructed into the validated H5 representation.
   - Frozen manifests replace repeated dataset discovery and rescanning.

2. **Baseline diffusion training**
   - The modular patch-conditioned x0 diffusion model preserves the validated
     notebook behavior.
   - The production generator used for downstream synthetic-data experiments
     uses the fixed full-training configuration.

3. **BR-LoRA adaptation**
   - Bayesian low-rank adapters are attached to the frozen diffusion backbone.
   - Only variational adapter parameters are optimized.
   - Posterior-mean and posterior-sampled inference are supported.

4. **nnU-Net screening**
   - Registered BraTS Training and Validation are converted to the screening
     Dataset500 representation.
   - A separate five-fold nnU-Net model is trained and frozen.
   - Its ensemble predictions are used only to identify candidate tumor-free
     validation slices.

5. **Compatibility-constrained cohort construction**
   - Screened bases are audited against the eligible training-donor pool.
   - Compatibility, morphology, matching, and donor-selection audits produce
     the frozen 250-case external cohort.

6. **Frozen synthetic-library design and production**
   - The 250-case cohort is preserved within a frozen 10,000-case design.
   - BR-LoRA inference generates the library in audited batches.
   - Batch acceptance verifies provenance and integrity before promotion into
     the permanent library.

7. **Downstream segmentation**
   - Frozen manifests define real-only, real + posterior-mean, and
     real + posterior-sampled training regimes.
   - All regimes use the same downstream segmentation implementation and
     frozen internal validation split.

8. **External downstream evaluation**
   - The completed downstream checkpoints are evaluated on the frozen
     UCSF-PDGM external cohort.

## Design Principles

- Preserve notebook-equivalent scientific behavior unless a change is explicit.
- Separate fixed data contracts from user-tunable experiment parameters.
- Prefer manifest-driven execution over hidden data discovery.
- Keep nnU-Net screening independent of BR-LoRA optimization.
- Freeze cohort and library designs before production.
- Use deterministic seeds and explicit provenance where reproducibility is
  required.
- Keep large generated artifacts outside Git.
