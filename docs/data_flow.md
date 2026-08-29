# Data Flow

This document summarizes the main data flow through the project.

```text
BraTS 2020 Training NIfTI
        │
        ├──────────────► nnU-Net training
        │                    │
        │                    ▼
        │              frozen 5-fold model
        │                    │
        │                    ▼
        │             validation screening
        │                    │
        │                    ▼
        │           tumor-free base candidates
        │                    │
        ▼                    │
H5 reconstruction           │
        │                    │
        ▼                    │
training manifest            │
        │                    │
        ├────► baseline training
        │            │
        │            ▼
        │      diffusion checkpoint
        │            │
        └────► BR-LoRA training
                     │
                     ▼
               BR-LoRA checkpoint
                     │
                     ├──────────────────┐
                     │                  │
                     ▼                  ▼
             donor pool           screened bases
                     │                  │
                     └────────┬─────────┘
                              ▼
                   compatibility graph
                              │
                              ▼
                     exact-flow matching
                              │
                              ▼
                  10,000-case design
                              │
                              ▼
                    40 × 250 batches
                              │
                              ▼
                  posterior generation
                              │
                              ▼
                  synthetic library
                              │
                              ▼
                  downstream evaluation
```

## Key Boundaries

- Screening does not modify BR-LoRA.
- BR-LoRA does not discover external cases internally.
- Library design is frozen before batch production.
- Batch production does not modify the scientific design.
- Batch acceptance does not regenerate cases; it validates staged artifacts and promotes them into the permanent library.
- Downstream experiments consume frozen library artifacts and manifests.
