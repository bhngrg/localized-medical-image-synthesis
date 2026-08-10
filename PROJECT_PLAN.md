# Localized Medical Image Synthesis
## Repository Development Roadmap

This document tracks the engineering and research roadmap for the repository.

The original notebook under `notebooks/` remains the scientific reference
implementation. The modular implementation is validated against it before new
functionality is introduced.

---

# Current Status

**Active milestone:** Full-training baseline refit and external-evaluation
infrastructure.

The repository foundation, data infrastructure, and baseline notebook refactor
are complete.

---

# Guiding Principles

- Preserve the original notebook as the reference implementation.
- Preserve original settings as default values.
- Expose legitimate user-tunable experiment parameters.
- Do not silently change scientific behavior.
- Validate every extracted module before extending it.
- Separate fixed data contracts from experiment settings.
- Avoid repeated raw-data scans when validated metadata already exist.
- Keep training, inference, composition, and evaluation responsibilities
  separate.

---

# Phase 0 — Repository Foundation

**Status:** ✅ Complete

- [x] Initialize Git repository
- [x] Create repository structure
- [x] Configure `.gitignore`
- [x] Create root documentation
- [x] Create dataset documentation
- [x] Create initial repository commit

---

# Phase 1 — Dataset Infrastructure

**Status:** ✅ Complete

- [x] Register BraTS 2020 training NIfTI release
- [x] Register BraTS 2020 validation NIfTI release
- [x] Write machine-readable dataset specifications
- [x] Reconstruct historical NIfTI-to-H5 preprocessing
- [x] Generate all 57,195 training H5 slices
- [x] Verify representative reconstructed H5 slices
- [x] Generate `manifest.csv`
- [x] Verify manifest against historical metadata
- [x] Preserve historical preprocessing conventions explicitly

---

# Phase 2 — Baseline Notebook Refactor

**Status:** ✅ Complete

## Data

- [x] H5 preprocessing utilities
- [x] Manifest-backed dataset
- [x] Dataset splitting
- [x] DataLoader construction

## Diffusion

- [x] Linear beta schedule
- [x] Forward q-sampling
- [x] Schedule encapsulation without notebook-global state

## Model

- [x] Sinusoidal timestep embedding
- [x] Conditioning MLP
- [x] Conditional residual blocks
- [x] Patch-conditioned x0 U-Net

## Training

- [x] Masked x0 loss
- [x] Batch preparation
- [x] One-epoch training
- [x] Stochastic internal validation
- [x] Best-checkpoint saving
- [x] Final-checkpoint saving
- [x] Executable training script

## Inference and Composition

- [x] Reconstruction inference
- [x] Manifest-backed candidate discovery
- [x] Seeded tumor-free base / donor-mask pair selection
- [x] Tumor-free regional insertion
- [x] Hard outside-mask preservation
- [x] Executable synthesis script

---

# Phase 2 Validation Record

**Status:** ✅ Complete

Exact equivalence has been demonstrated for

| Component | Result |
|---|---|
| Eligible training slice count | ✅ |
| Internal 90/10 split | ✅ |
| Diffusion tensors | ✅ |
| q-sampling | ✅ |
| Model parameter count | ✅ |
| State-dict structure | ✅ |
| Seeded initialization | ✅ |
| Forward output | ✅ |
| Model input preparation | ✅ |
| Masked loss | ✅ |
| Gradients | ✅ |
| AdamW update | ✅ |
| Optimizer state | ✅ |
| Epoch train loss | ✅ |
| Epoch validation loss | ✅ |
| Best checkpoint | ✅ |
| Final checkpoint | ✅ |
| Reconstruction inference | ✅ |
| Candidate membership | ✅ |
| Candidate ordering | ✅ |
| Seeded pair selection | ✅ |
| Regional synthesis tensors | ✅ |
| Outside-mask preservation | ✅ |

---

# Phase 3 — Full-Training Baseline and External Evaluation

**Status:** 🚧 In Progress

## Goal

Extend the validated baseline without altering its default notebook-compatible
behavior.

## Training Modes

- [x] Preserve `internal` mode as the default:
  - 90% optimization
  - 10% internal validation
  - best-checkpoint selection by masked validation loss

- [ ] Add `full_train` mode:
  - 100% of eligible training slices used for optimization
  - fixed number of epochs
  - final checkpoint saved
  - no fabricated masked validation loss

## Official BraTS Validation Release

The official BraTS 2020 validation release contains MRI modalities but no
segmentation masks.

Therefore:

- it cannot directly provide the baseline masked validation loss;
- it must not be used as if ground-truth lesion masks were available;
- it will be used only through an explicitly defined mask-free or
  externally-conditioned evaluation protocol.

## Deliverables

- [ ] Full-training DataLoader path
- [ ] Full-training trainer path
- [ ] Backward-compatible training CLI
- [ ] Overnight baseline full-training run
- [ ] External-validation inference loader
- [ ] Explicit evaluation protocol for unlabeled validation subjects

---

# Phase 4 — Regional Composition Framework

**Status:** ⏳ Planned

## Goal

Generalize localized composition into reusable components independent of the
adaptation strategy.

- [ ] Base-image interface
- [ ] Donor-image interface
- [ ] Lesion-mask interface
- [ ] Regional composition utilities
- [ ] Common synthesis interface
- [ ] True regional-composition ablations

---

# Phase 5 — Parameter-Efficient Adaptation

**Status:** ⏳ Planned

- [ ] Frozen backbone
- [ ] Full fine-tuning
- [ ] BitFit
- [ ] LoRA
- [ ] DoRA
- [ ] LoKr
- [ ] Regional LoRA

---

# Phase 6 — Bayesian Regional LoRA

**Status:** ⏳ Planned

- [ ] Bayesian low-rank adapters
- [ ] Variational optimization
- [ ] Posterior sampling
- [ ] Predictive uncertainty estimation

---

# Phase 7 — Image-Level Reliability Framework

**Status:** ⏳ Planned

- [ ] Predictive uncertainty
- [ ] Structural consistency
- [ ] Topology-aware summaries
- [ ] Repeat stability
- [ ] Controlled perturbation analysis
- [ ] Review prioritization

---

# Phase 8 — Benchmarking and Reproducibility

**Status:** ⏳ Planned

- [ ] Common evaluation protocol
- [ ] Accuracy metrics
- [ ] Efficiency metrics
- [ ] Reliability metrics
- [ ] Statistical summaries
- [ ] Visualization
- [ ] Reproducible experiment configurations

---

# Current Next Actions

1. Add backward-compatible `full_train` mode.
2. Validate that `internal` mode remains unchanged.
3. Validate that `full_train` uses all 19,941 currently eligible slices.
4. Launch the fixed-epoch full-training baseline.
5. Define the appropriate mask-free use of the official BraTS validation
   release before treating it as an evaluation dataset.

---

# Long-Term Vision

The completed repository will provide independently testable and reusable
components for

- localized medical image synthesis,
- regional composition,
- parameter-efficient adaptation,
- Bayesian Regional LoRA,
- uncertainty estimation,
- topology-aware reliability assessment, and
- reproducible benchmarking.
