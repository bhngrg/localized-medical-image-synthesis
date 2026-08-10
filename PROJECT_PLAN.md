# Localized Medical Image Synthesis
## Repository Development Roadmap

This document tracks the engineering and research roadmap for the repository.

The original notebook under `notebooks/` remains the scientific reference
implementation. The modular implementation is validated against it before new
functionality is introduced.

---

# Current Status

**Active milestone:** Integration of Bayesian Regional LoRA (BR-LoRA) into the
validated localized medical image synthesis framework.

The repository foundation, data infrastructure, baseline notebook refactor,
full-training baseline workflow, and core BR-LoRA learning infrastructure are
complete or independently validated. Current development focuses on integrating
BR-LoRA into the end-to-end experiment workflow while preserving the validated
baseline implementation.

---

# Guiding Principles

- Preserve the original notebook as the reference implementation.
- Preserve original settings as default values.
- Expose legitimate user-tunable experiment parameters.
- Do not silently change scientific behavior.
- Validate every extracted module before extending it.
- Validate every Bayesian component independently before integrating it into
  the end-to-end training pipeline.
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

- [x] Preserve `internal` mode:
  - 90% optimization
  - 10% internal validation
  - best-checkpoint selection by masked validation loss

- [x] Add `full_train` mode:
  - 100% of eligible training slices used for optimization
  - fixed number of epochs
  - final checkpoint saved
  - no fabricated masked validation loss

- [x] Validate `full_train` on all 19,941 eligible training slices
- [x] Complete 30-epoch internal and full-training baseline runs
- [x] Record baseline training logs

## Official BraTS Validation Release

The official BraTS 2020 validation release contains MRI modalities but no
segmentation masks.

Therefore:

- it cannot directly provide the baseline masked validation loss;
- it must not be used as if ground-truth lesion masks were available;
- it will be used only through an explicitly defined mask-free or
  externally-conditioned evaluation protocol.

## Remaining Deliverables

- [ ] External-validation inference loader
- [ ] Explicit evaluation protocol for unlabeled validation subjects

---

# Phase 4 — Regional Composition Framework

**Status:** 🚧 In Progress

## Goal

Generalize localized composition into reusable components independent of the
adaptation strategy.

- [x] Base-image interface through the current synthesis workflow
- [x] Donor-image interface through the current synthesis workflow
- [x] Lesion-mask interface through the current synthesis workflow
- [x] Regional composition utilities
- [x] Exact hard outside-mask preservation
- [ ] Common synthesis interface across adaptation strategies
- [ ] True regional-composition ablations

---

# Phase 5 — Parameter-Efficient Adaptation

**Status:** 🚧 In Progress

- [x] Frozen backbone support
- [x] Full fine-tuning baseline support
- [x] Deterministic convolutional LoRA infrastructure
- [ ] BitFit
- [ ] DoRA
- [ ] LoKr
- [ ] Unified Regional LoRA experiment integration

---

# Phase 6 — Bayesian Regional LoRA

**Status:** 🚧 In Progress

## Core Adapter Infrastructure

- [x] Deterministic LoRA initialization bridge
- [x] Bayesian low-rank convolutional adapters
- [x] Mean-field Gaussian variational posteriors
- [x] Reparameterized posterior sampling
- [x] Posterior-mean mode
- [x] Analytic KL divergence
- [x] BR-LoRA parameter accounting
- [x] Frozen-backbone preservation

## Variational Optimization

- [x] Mask-aware reconstruction objective
- [x] Exact reconstruction equivalence to the validated baseline loss
- [x] Parameter-normalized KL regularization
- [x] Linear KL warmup
- [x] Complete BR-LoRA variational objective
- [x] Single BR-LoRA optimization step
- [x] Gradient clipping
- [x] BR-LoRA training epoch runner
- [x] Posterior-mean validation epoch runner
- [x] Global-step bookkeeping
- [x] Sample-weighted epoch metrics

## Validation Record

The BR-LoRA implementation has been independently audited for

- [x] Exact seven-layer target selection
- [x] 18,052 deterministic LoRA parameters
- [x] 36,104 variational posterior parameters
- [x] 28 trainable posterior tensors
- [x] Exact frozen-backbone preservation at fresh LoRA initialization
- [x] Exact posterior-mean equivalence to deterministic LoRA initialization
- [x] Reproducible seeded posterior realizations
- [x] Distinct realizations under distinct posterior seeds
- [x] Finite, nonnegative KL divergence
- [x] KL gradients reaching all posterior parameters
- [x] Real-H5 one-step optimization
- [x] Updates restricted to posterior parameters
- [x] Tiny train/validation epoch audit
- [x] Validation with zero parameter changes

## Remaining Deliverables

- [ ] Multi-epoch BR-LoRA experiment runner
- [ ] Best/latest/final checkpoint orchestration
- [ ] Resume-training support
- [ ] Training-history serialization
- [ ] Posterior inference utilities
- [ ] Posterior realization management
- [ ] Predictive uncertainty estimation

---

# Phase 7 — Reliability Assessment Framework

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

1. Add multi-epoch BR-LoRA training orchestration.
2. Add BR-LoRA checkpoint and training-history serialization.
3. Validate checkpoint save/load and resume behavior.
4. Implement posterior-mean and posterior-sampled inference utilities.
5. Preserve configurable Bayesian realizations for downstream uncertainty
   analysis.
6. Integrate BR-LoRA into the common benchmarking framework.
7. Add remaining PEFT baselines and reliability analyses after the BR-LoRA
   training path is complete.

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
