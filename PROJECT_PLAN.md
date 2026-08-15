# Localized Medical Image Synthesis
## Repository Development Roadmap

This document tracks the engineering and research roadmap for the repository.

The original notebook under `notebooks/` remains the scientific reference
implementation. The modular implementation is validated against it before new
functionality is introduced.

---

# Current Status

**Active milestone:** Large-scale BR-LoRA synthetic library production, downstream evaluation infrastructure, and reliability-aware benchmarking.

The repository foundation, data infrastructure, baseline notebook refactor,
full-training baseline workflow, core BR-LoRA learning infrastructure,
posterior-sampling analyses, and manifest-driven external BR-LoRA evaluation
infrastructure are complete or independently validated. Current development focuses on large-scale BR-LoRA synthetic library production, frozen manifest generation, automated batch orchestration, downstream evaluation infrastructure, predictive uncertainty estimation, image-level reliability assessment, and comparative benchmarking.

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
- Keep training, inference, composition, external-cohort screening, posterior
  analysis, and evaluation responsibilities separate.
- Treat external case selection as a fixed manifest-generation problem rather
  than embedding case discovery inside model evaluation.

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

**Status:** ✅ Complete

## Goal

Extend the validated baseline without altering its default notebook-compatible
behavior while supporting both model-selection and final-training workflows.

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
- [x] Complete 50-epoch internal and full-training baseline runs
- [x] Record baseline training logs
- [x] Generalize the training infrastructure to support both split modes through a common training interface

## Official BraTS Validation Release

The official BraTS 2020 validation release contains MRI modalities but no
segmentation masks.

Therefore:

- it cannot directly provide the baseline masked validation loss;
- it must not be used as if ground-truth lesion masks were available;
- it is reserved for downstream inference and evaluation protocols that do not
  require ground-truth segmentations.

## Deliverables

- [x] Internal-validation training workflow
- [x] Full-training workflow
- [x] Shared split-mode infrastructure
- [x] Fixed-epoch checkpoint generation
- [x] Reproducible training logs
- [x] Registered BraTS validation-slice loader
- [x] Training-compatible raw-NIfTI preprocessing path
- [x] Numerical preprocessing-equivalence audit against reconstructed training H5 slices
- [x] Fixed external-evaluation manifest contract
- [x] Manifest-driven BR-LoRA external evaluator
- [x] Retained posterior realization and posterior-summary artifact contract
- [x] Case-specific deterministic seeding independent of manifest order/subsetting

---

# Phase 4 — Regional Composition Framework

**Status:** 🚧 Active

## Goal

Generalize localized composition into reusable components independent of the
adaptation strategy.

- [x] Base-image interface through the current synthesis workflow
- [x] Donor-image interface through the current synthesis workflow
- [x] Lesion-mask interface through the current synthesis workflow
- [x] Regional composition utilities
- [x] Exact hard outside-mask preservation
- [x] Integration with the validated baseline synthesis workflow
- [ ] Common synthesis interface across all adaptation strategies
- [ ] True regional-composition ablations

---

# Phase 5 — Parameter-Efficient Adaptation

**Status:** 🚧 In Progress

## Goal

Establish a common parameter-efficient adaptation framework for localized
medical image synthesis while enabling direct comparison across adaptation
strategies.

- [x] Frozen backbone support
- [x] Full fine-tuning baseline support
- [x] Deterministic convolutional LoRA infrastructure
- [x] Bayesian Regional LoRA (BR-LoRA)
- [x] Variational low-rank adapter infrastructure
- [x] Internal (90/10) BR-LoRA training workflow
- [x] Full-training BR-LoRA workflow
- [ ] BitFit
- [ ] DoRA
- [ ] LoKr
- [ ] Unified adaptation interface across all PEFT methods
- [ ] Common benchmarking pipeline across adaptation strategies

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

## Experiment Infrastructure

- [x] Multi-epoch BR-LoRA training orchestration
- [x] Internal (90/10) training workflow
- [x] Full-training workflow
- [x] Best/latest/final checkpoint orchestration
- [x] Resume-training support
- [x] Training-history serialization
- [x] Split-mode-aware training interface
- [x] Training-log preservation

## Inference Infrastructure

- [x] Posterior-mean inference
- [x] Posterior-sampled inference
- [x] Shared diffusion-input preparation
- [x] Strict checkpoint reconstruction
- [x] Posterior realization management
- [x] Registered external BraTS validation-slice loading
- [x] Fixed-manifest external base / training donor pair preparation
- [x] Manifest-driven external BR-LoRA evaluation
- [x] Per-case deterministic seeding independent of manifest order/subsetting
- [x] Retained posterior realization stacks
- [x] CPU-derived posterior mean, variance, and standard deviation products
- [x] Deterministic hard-composite reconstruction from retained artifacts

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
- [x] Multi-epoch training audit
- [x] Resume-training audit
- [x] Full-training audit
- [x] Posterior inference audit
- [x] External validation-slice preprocessing equivalence audit
- [x] External posterior artifact contract audit
- [x] Exact outside-mask reconstruction audit
- [x] Manifest-order/subsetting reproducibility audit

## Remaining Deliverables

- [ ] Run 100-realization external evaluation for the internal (90/10) BR-LoRA model
- [ ] Run 100-realization external evaluation for the full-training BR-LoRA model
- [ ] Predictive uncertainty summaries across the definitive external cohort
- [ ] Uncertainty calibration analyses
- [ ] Reliability-score construction

---

# Phase 7 — External Cohort Screening with nnU-Net

**Status:** ✅ Completed

## Goal

Construct a reproducible external BraTS validation base-image cohort without
using unavailable ground-truth validation segmentations.

The screening workflow is intentionally isolated from BR-LoRA training and
inference. Its sole role is to generate a fixed external base manifest consumed
by `scripts/evaluate_br_lora_external.py`.

## Screening Workflow

- [x] Create `screening/brats_nnunet/` project subtree
- [x] Configure nnU-Net dataset conversion from the registered BraTS 2020 training release
- [x] Train nnU-Net using the four BraTS MRI modalities and training segmentation masks
- [x] Freeze the selected nnU-Net screening model
- [x] Validation inference launcher implemented (Falcon)
- [x] Run frozen nnU-Net inference on all 125 official BraTS 2020 validation subjects
- [x] Derive predicted whole-tumor masks
- [x] Identify tumor-free external base slices from the nnU-Net predictions
- [x] Audit the external base–donor compatibility space
- [x] Audit eligible donor morphology
- [x] Construct and compare nested 125/250/625 candidate evaluation cohorts
- [x] Audit compatibility-conditioned donor-selection distributions
- [x] Diagnose the global one-to-one matching algorithm
- [x] Freeze the definitive 250-case external evaluation manifest
- [x] Audit the definitive frozen donor assignments
- [x] Preserve complete screening and manifest-construction provenance

## External BR-LoRA Evaluation

- [x] Registered validation-dataset interface
- [x] Training-compatible validation-slice preprocessing
- [x] Fixed external evaluation manifest contract
- [x] External base / training donor pair preparation
- [x] Manifest-driven BR-LoRA evaluator
- [x] Retained posterior realization stack
- [x] Posterior mean, variance, and standard deviation products
- [x] Hard-composite reconstruction from retained artifacts
- [x] Case-specific deterministic seeding
- [x] Plug the definitive screened external manifest into the evaluator
- [ ] Run internal (90/10) BR-LoRA with 100 posterior realizations per case
- [ ] Run full-training BR-LoRA with 100 posterior realizations per case


---

# Phase 7.5 — BR-LoRA Synthetic Library Production

**Status:** ✅ Complete (production infrastructure)

## Goal

Produce a deterministic, reproducible synthetic image library suitable for downstream segmentation experiments.

## Library Design

- [x] Compatibility-constrained library design
- [x] Deterministic donor/base assignment
- [x] Exact-flow assignment optimization
- [x] Frozen 10,000-case design manifest
- [x] Batch manifest generation (40 × 250 cases)

## Production Pipeline

- [x] Automated batch orchestration
- [x] External evaluation manifest generation
- [x] Production logging
- [x] Checksum inventories
- [x] Production audits
- [x] Falcon synchronization
- [x] Acceptance audits
- [x] Resumable production workflow

## Remaining

- [ ] Complete production of all 40 batches
- [ ] Final library integrity audit
- [ ] Freeze master synthetic library manifest


---

# Phase 8 — Reliability Assessment Framework

**Status:** ⏳ Planned

## Goal

Develop an image-level reliability framework for localized medical image
synthesis that combines predictive uncertainty, structural analysis, repeat
stability, and robustness under controlled perturbations.

## Predictive Uncertainty

- [ ] Posterior predictive uncertainty estimation
- [ ] Pixel-level uncertainty maps
- [ ] Image-level uncertainty summaries

## Structural Reliability

- [ ] Structural consistency analysis
- [ ] Topology-aware summaries
- [ ] Connected-component analysis
- [ ] Shape and size consistency metrics

## Stability and Robustness

- [ ] Repeat synthesis stability
- [ ] Source-image perturbation analysis
- [ ] Base-image perturbation analysis
- [ ] Lesion-mask perturbation analysis

## Reliability Assessment

- [ ] Composite image-level reliability measures
- [ ] Continuous review prioritization
- [ ] Reliability visualization
- [ ] Reliability reporting utilities

---

# Phase 9 — Benchmarking and Reproducibility

**Status:** ⏳ Planned

## Goal

Establish a common benchmarking framework for localized medical image
synthesis that enables fair comparison of adaptation strategies across
accuracy, computational efficiency, and image-level reliability.

## Adaptation Benchmarking

- [ ] Common evaluation protocol
- [ ] Frozen backbone benchmark
- [ ] Full fine-tuning benchmark
- [ ] LoRA benchmark
- [ ] BR-LoRA benchmark
- [ ] BitFit benchmark
- [ ] DoRA benchmark
- [ ] LoKr benchmark

## Performance Evaluation

- [ ] Accuracy metrics
- [ ] Efficiency metrics
- [ ] Reliability metrics
- [ ] Statistical summaries
- [ ] Comparative visualizations

## Reproducibility

- [ ] Reproducible experiment configurations
- [ ] Experiment manifests
- [ ] Versioned evaluation outputs
- [ ] End-to-end reproducibility documentation

---

# Current Next Actions

1. Finish production of the remaining BR-LoRA synthetic-library batches.
2. Complete the final production audit and freeze the master library manifest.
3. Launch downstream segmentation experiments using the completed synthetic library.
4. Complete image-level reliability analyses.
5. Finish the remaining PEFT baselines.
6. Perform comparative benchmarking.
7. Prepare the AAAI manuscript experiments.

---

# Long-Term Vision

The completed repository will provide independently testable, reusable, and
reproducible components for

- localized medical image synthesis,
- regional image composition,
- parameter-efficient adaptation,
- Bayesian Regional LoRA,
- predictive uncertainty estimation,
- image-level reliability assessment,
- topology-aware structural analysis, and
- comprehensive benchmarking of adaptation strategies.
