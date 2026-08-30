# Data Flow

This document summarizes the implemented data dependencies across the project.

```text
BraTS 2020 Training NIfTI                  BraTS 2020 Validation NIfTI
        │                                             │
        ▼                                             ▼
register_dataset.py                    register_validation_dataset.py
        │                                             │
        ├──────────────────────┐        ┌─────────────┘
        │                      │        │
        ▼                      │        ▼
H5 reconstruction             └──► prepare_nnunet_dataset.py
        │                              │
        ▼                              ▼
training manifest                Dataset500 screening data
        │                              │
        ├──► baseline training         ▼
        │         │              five-fold nnU-Net training
        │         ▼                    │
        │   diffusion checkpoint       ▼
        │         │              validation ensemble prediction
        │         ▼                    │
        └──► BR-LoRA training          ▼
                  │              validation slice screening
                  │                    │
                  ▼                    ▼
          BR-LoRA checkpoint     tumor-free base candidates
                  │                    │
                  │              compatibility + morphology audits
                  │                    │
                  │                    ▼
                  │              frozen 250-case cohort
                  │                    │
                  └──────────┬─────────┘
                             ▼
                  frozen 10,000-case design
                             │
                             ▼
                   BR-LoRA library batches
                             │
                             ▼
                    accepted synthetic library
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
          real only      real + mean    real + sampling
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                 downstream segmentation
                             │
                             ▼
                  frozen UCSF-PDGM cohort
                             │
                             ▼
                    external evaluation
```

## Key Boundaries

- BraTS Training and Validation are registered separately.
- nnU-Net screening uses both registered BraTS releases but does not modify
  BR-LoRA or participate in BR-LoRA optimization.
- BR-LoRA does not discover external cases internally.
- Compatibility-constrained cohort construction is frozen before synthetic
  library design.
- The 10,000-case design is frozen before batch production.
- Batch production does not modify the scientific design.
- Batch acceptance validates staged artifacts; it does not regenerate cases.
- Downstream experiments consume frozen manifests and accepted library
  artifacts.
- UCSF-PDGM evaluation is downstream of the trained segmentation checkpoints
  and is independent of the BraTS screening cohort.
