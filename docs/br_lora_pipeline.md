# BR-LoRA End-to-End Pipeline

## Overview

The BR-LoRA workflow extends a validated patch-conditioned diffusion model with Bayesian low-rank adapters for localized medical image synthesis.

The pipeline is organized so that model fitting, cohort screening, library design, production, and downstream evaluation can be reproduced independently.

## End-to-End Workflow

```text
BraTS 2020 training NIfTI
        │
        ▼
Dataset registration / H5 reconstruction
        │
        ▼
Validated patch-conditioned x0 diffusion backbone
        │
        ▼
BR-LoRA training
        │
        ▼
Frozen BR-LoRA checkpoint
        │
        ├───────────────────────────────┐
        │                               │
        ▼                               ▼
Posterior inference              nnU-Net screening
        │                               │
        │                               ▼
        │                    Tumor-free validation bases
        │                               │
        └──────────────┬────────────────┘
                       ▼
             Compatibility-constrained
               library design
                       │
                       ▼
             10,000-case frozen design
                       │
                       ▼
             40 batches × 250 cases
                       │
                       ▼
             Production + transfer +
                acceptance audits
                       │
                       ▼
             Frozen synthetic library
                       │
                       ▼
            Downstream evaluation
```

## BR-LoRA Model

The validated BR-LoRA configuration adapts seven convolutional layers of the local `AppearanceX0UNet` backbone using rank-4, alpha-8 low-rank adapters.

Each adapted LoRA factor is represented by a mean-field Gaussian variational posterior. Training uses reparameterized sampling and a variational objective combining the validated regional reconstruction loss with parameter-normalized KL regularization.

Two inference modes are supported:

- **posterior mean mode** for deterministic inference;
- **posterior sampling mode** for uncertainty-aware repeated realizations.

## External Evaluation Contract

External evaluation is manifest-driven. Each case specifies:

- a unique case identifier;
- an external BraTS validation subject;
- an external axial slice index;
- an MRI modality; and
- a donor H5 slice from the labeled BraTS training dataset.

Case-specific seeds are derived deterministically from the global evaluation seed and case identifier so that posterior sampling remains stable under manifest reordering or subsetting.

## Posterior Artifacts

Each completed external-evaluation case stores:

```text
posterior_samples.pt
posterior_mean.pt
posterior_variance.pt
posterior_std.pt
composite_mean.pt
metadata.json
```

The retained posterior stack includes the prediction samples together with the base image, transferred mask, donor image/patch information, diffusion state, and case metadata required for reproducible downstream analysis.

## Production Philosophy

The production workflow intentionally separates:

- scientific design,
- generation,
- integrity checking,
- transfer,
- Falcon acceptance, and
- cleanup.

No local batch is deleted until its Falcon-side transfer and acceptance checks pass.
