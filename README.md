# Localized Medical Image Synthesis

A modular research framework for donor-conditioned localized medical image synthesis, regional composition, parameter-efficient adaptation, Bayesian Regional LoRA (BR-LoRA), and image-level reliability assessment.

> **Project status:** Active development. The baseline patch-conditioned x0 diffusion workflow has been refactored from the original notebook into modular Python components and validated against the reference implementation. The repository now supports validated BR-LoRA training, posterior-mean and posterior-sampling inference, screened external evaluation, and reproducible construction of a frozen 10,000-case BR-LoRA synthetic library. The current production workflow generates the library in 40 audited batches of 250 cases, with checksum verification and acceptance into a master library manifest.

---

## Overview

This project investigates localized medical image synthesis in which pathological appearance from a donor image is synthesized within a prescribed region of a tumor-free base image while preserving unaffected anatomy outside that region.

The framework explicitly separates:

- tumor-free base anatomy,
- donor pathological appearance,
- prescribed lesion masks,
- regional synthesis, and
- hard regional composition.

The primary Bayesian adaptation method is **Bayesian Regional LoRA (BR-LoRA)**. BR-LoRA places mean-field Gaussian distributions over low-rank adapter parameters while keeping the conditional diffusion backbone frozen. The repository also supports deterministic parameter-efficient alternatives including BitFit, LoRA, DoRA, and LoKr.

The broader research workflow evaluates localized synthesis through complementary measures of predictive uncertainty, structural consistency, repeat stability, and robustness to controlled perturbations.

---

## Current Capabilities

The repository currently supports:

- BraTS 2020 dataset registration and validation,
- reconstruction of the historical H5 training representation,
- reproducible slice-level manifest generation,
- modular diffusion scheduling,
- modular `AppearanceX0UNet` training and inference,
- localized lesion synthesis,
- hard regional image composition,
- internal 90/10 model-selection workflows,
- fixed-epoch full-training refits,
- BitFit, LoRA, DoRA, LoKr, and BR-LoRA adaptation,
- mean-field Gaussian variational posteriors,
- analytic KL divergence,
- reparameterized posterior sampling,
- posterior-mean inference,
- posterior variance and standard-deviation summaries,
- checkpoint save/load and resume support,
- Monte Carlo convergence analysis,
- Monte Carlo standard error (MCSE) analysis,
- registered external BraTS validation-slice loading,
- external preprocessing equivalence validation,
- frozen-manifest external BR-LoRA evaluation,
- deterministic per-case posterior sampling,
- nnU-Net-based external-cohort screening,
- compatibility-constrained base/donor pairing,
- frozen 10,000-case library design,
- batch-wise synthetic-library production,
- SHA-256 transfer verification, and
- batch acceptance into a master library manifest.

Ongoing work focuses on downstream reliability analyses, topology-aware structural evaluation, robustness experiments, benchmarking across adaptation strategies, and manuscript figures and tables.

---

## Repository Organization

```text
localized-medical-image-synthesis/
│
├── checkpoints/          Model checkpoints (generated artifacts)
├── configs/              Baseline and BR-LoRA configuration files
├── data/                 Dataset setup documentation
├── docs/                 Extended project documentation
├── downstream_evaluation/
│   └── manifests/        Frozen downstream and synthetic-library manifests
├── logs/                 Training, evaluation, screening, and production logs
├── notebooks/            Original research notebooks / reference implementation
├── outputs/              Generated analyses, figures, and synthesis outputs
├── screening/
│   └── brats_nnunet/     nnU-Net screening and compatibility workflow
├── scripts/              Executable training, inference, analysis, and production workflows
├── src/                  Modular Python implementation
├── README.md
├── PROJECT_PLAN.md
├── LICENSE
└── .gitignore
```

The repository is still under active development. Script locations are intentionally being kept stable while the 10,000-case production run is in progress; structural cleanup will be performed after production completes.

---

## Dataset Pipeline

The repository operates from the official BraTS 2020 NIfTI releases.

### Training data

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

The reconstructed training dataset contains 57,195 H5 slices. Of these, 19,941 tumor-containing slices satisfy the validated slice-selection criteria used by the synthesis workflow.

### Validation data

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

The official validation release does not contain tumor segmentation masks, so it is not used for masked reconstruction loss or checkpoint selection. Instead, it is reserved for downstream external inference, screening, predictive uncertainty estimation, and reliability evaluation.

A registered validation-slice loader reproduces the effective training image representation from raw validation NIfTI data. This preprocessing path has been numerically validated against the reconstructed H5 training representation.

See [`data/README.md`](data/README.md) for dataset setup and registration details.

---

## External Cohort Screening

A frozen five-fold nnU-Net ensemble is used to screen the official BraTS 2020 validation release for tumor-free candidate slices.

The screening workflow then audits candidate bases for anatomical compatibility with eligible donor lesions. This produced a reproducible nested 125/250/625 external cohort design and a definitive 250-case external evaluation cohort.

The definitive external cohort contains two cases per validation subject. Each base is paired with a unique donor slice through deterministic compatibility-constrained matching and is consumed directly by the external BR-LoRA evaluation workflow.

Screening code and documentation are located under:

```text
screening/brats_nnunet/
```

---

## Training Split Modes

The baseline configuration preserves the original notebook behavior.

### Internal model selection

```yaml
data:
  split_mode: internal
  train_fraction: 0.9
```

This uses 90% of eligible tumor-containing training slices for optimization and reserves 10% for internal model selection.

### Full training

```yaml
data:
  split_mode: full_train
```

This uses 100% of eligible BraTS training slices and omits the masked validation loss used during internal model selection. It is intended for fixed-epoch final refits after model settings have been established.

Validated reference workflows include:

- 50-epoch internal baseline training,
- 50-epoch internal BR-LoRA training,
- 50-epoch full-training baseline, and
- 50-epoch full-training BR-LoRA.

---

## Bayesian Regional LoRA

BR-LoRA extends the validated local `AppearanceX0UNet` through Bayesian low-rank adaptation while preserving the frozen backbone.

The validated rank-4, alpha-8 configuration targets seven convolutional layers:

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

This configuration contains 18,052 deterministic LoRA parameters and 36,104 variational posterior parameters distributed across 28 trainable posterior tensors.

Each LoRA factor is represented by a trainable mean-field Gaussian posterior. Inference can use either posterior samples or posterior means:

```yaml
sample_posterior: true
```

or

```yaml
sample_posterior: false
```

Posterior realizations are generated on demand. The evaluation pipeline can retain the exact posterior realization stack for reproducible downstream analyses.

---

## BR-LoRA Variational Objective

BR-LoRA preserves the validated regional reconstruction objective while adding parameter-normalized KL regularization.

Conceptually:

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

Training uses reparameterized posterior sampling with linear KL warmup. Internal validation is performed in posterior-mean mode so checkpoint selection is based on a deterministic criterion.

---

## Baseline and BR-LoRA Validation

The modular implementation has been checked against the reference notebook and independently audited across baseline and Bayesian components.

Validated baseline behavior includes:

- dataset filtering and train/validation splitting,
- diffusion schedule construction and q-sampling,
- model parameterization and seeded initialization,
- forward inference,
- masked training loss,
- gradients,
- AdamW updates,
- optimizer state,
- epoch-level losses,
- checkpoint contents,
- reconstruction inference,
- candidate discovery and seeded pair selection, and
- localized hard regional composition.

Validated BR-LoRA behavior includes:

- exact seven-layer target selection,
- deterministic and variational parameter accounting,
- frozen-backbone preservation,
- zero-update initialization,
- posterior-mean equivalence at initialization,
- seeded posterior reproducibility,
- analytic KL divergence and gradients,
- reconstruction-loss equivalence,
- KL warmup and normalization,
- real-H5 optimization,
- gradient clipping,
- checkpoint save/load and resume,
- posterior mean and posterior sampling inference,
- posterior convergence analysis,
- MCSE analysis,
- external preprocessing equivalence,
- manifest-driven external inference,
- retained posterior realization validation, and
- case-specific reproducibility independent of manifest order.

The original notebook remains the scientific reference implementation. The modular repository is the primary development and experimentation platform.

---

## External BR-LoRA Evaluation

External evaluation consumes a fixed manifest rather than selecting cases internally. This ensures that trained models are evaluated on exactly the same external bases and donor lesions.

The evaluation workflow supports:

- deterministic per-case seeds,
- fixed diffusion noise across posterior draws,
- configurable posterior sample counts,
- resume-safe case validation,
- retained posterior realization stacks,
- posterior mean,
- posterior variance,
- posterior standard deviation,
- composite mean reconstruction, and
- per-case metadata and run-level summaries.

Primary entry point:

```text
scripts/evaluate_br_lora_external.py
```

---

## Frozen 10,000-Case Synthetic Library

The repository includes a prespecified library design for 10,000 BR-LoRA synthetic cases.

The final design contains:

- 10,000 total cases,
- 125 external subjects,
- 80 cases per external subject,
- 40 batches of 250 cases,
- globally unique donor slices,
- compatibility-constrained base/donor assignments,
- a donor-subject cap of 31 cases,
- a maximum final base reuse of 10, and
- exact preservation of the original 250-case Batch 0001 cohort.

The frozen design artifacts are stored under:

```text
downstream_evaluation/manifests/br_lora_library_design_10000/
```

The compatibility cache used during library construction is intentionally not version controlled because it is recomputable from the frozen inputs.

---

## Library Production Workflow

The 10,000-case library is produced sequentially in 40 batches of 250 cases.

The production workflow is:

```text
Frozen batch manifest
        │
        ▼
BR-LoRA posterior inference
        │
        ▼
Local production audit
        │
        ▼
Local SHA-256 inventory
        │
        ▼
Transfer to persistent compute storage
        │
        ▼
Remote SHA-256 recomputation
        │
        ▼
Exact checksum comparison
        │
        ▼
Batch acceptance
        │
        ▼
Master library manifest update
        │
        ▼
Local staging cleanup
```

Primary orchestration scripts are:

```text
scripts/run_br_lora_library_batch.py
scripts/run_br_lora_library_pipeline.py
scripts/run_br_lora_library_all_remaining.sh
screening/brats_nnunet/scripts/design_br_lora_library_10000.py
```

The orchestrator is fail-safe: later batches are not started if a preceding batch fails, and local production data are not deleted unless transfer verification and remote acceptance succeed.

Machine-specific storage paths used for local production are implementation details of the current research environment and will be parameterized further as the repository is prepared for public release.

---

## Downstream Evaluation Manifests

Version-controlled downstream manifests include:

- real BraTS slice catalogs,
- subject-level summaries,
- downstream subject splits,
- training-only donor pools,
- external cohort manifests,
- the frozen 10,000-case library design,
- per-batch library manifests, and
- per-batch external evaluation manifests.

These artifacts provide a frozen record of case selection and experimental composition without requiring generated posterior tensors or large image outputs to be stored in Git.

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

External evaluation
-------------------
scripts/evaluate_br_lora_external.py

Posterior analysis
------------------
scripts/audit_br_lora_posterior.py
scripts/analyze_br_lora_posterior_convergence.py
scripts/analyze_br_lora_posterior_mcse.py

Synthetic-library production
----------------------------
scripts/run_br_lora_library_batch.py
scripts/run_br_lora_library_pipeline.py
scripts/run_br_lora_library_all_remaining.sh
```

The historical preprocessing audit utility `scripts/verify_h5_conversion.py` is retained for equivalence and provenance checks.

---

## Repository Design Principles

- Preserve the original notebook as the scientific reference implementation.
- Preserve notebook behavior as the default baseline configuration.
- Expose legitimate experiment parameters rather than silently changing behavior.
- Refactor incrementally and validate extracted components.
- Validate Bayesian components independently before end-to-end integration.
- Avoid implicit dataset discovery and hidden preprocessing assumptions.
- Separate reusable library code from executable workflows.
- Perform expensive dataset validation once and reuse generated metadata.
- Share common functionality across training, inference, and evaluation workflows.
- Keep frozen experimental manifests under version control.
- Keep large generated tensors, checkpoints, and recomputable caches out of Git.
- Preserve provenance through logs, hashes, manifests, and deterministic seeds.

---

## Reproducibility

The repository uses deterministic manifests, fixed seeds, explicit dataset specifications, frozen case assignments, checkpoint provenance, and SHA-256 inventories to make research artifacts traceable.

Large generated datasets and posterior tensors are not distributed through Git. Instead, the repository retains the code, configuration, manifests, audit outputs, and provenance needed to reconstruct the workflows.

Before public release, remaining machine-specific paths in operational scripts and historical logs will be reviewed and documented or parameterized where appropriate.

---

## Development Roadmap

See [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for the detailed engineering and research roadmap.

Near-term work includes:

- completion and final audit of the 10,000-case BR-LoRA library,
- downstream segmentation experiments,
- predictive uncertainty analyses,
- topology-aware structural analyses,
- repeat-stability analyses,
- controlled source/base/mask perturbation experiments,
- PEFT benchmarking, and
- manuscript tables and figures.

---

## Citation

If you use this repository in academic work, please cite:

- the BraTS challenge and dataset,
- relevant third-party software used in your workflow, and
- the associated BR-LoRA manuscript or repository release once available.

A formal citation entry will be added when the corresponding manuscript and public release are finalized.

---

## License

This repository is distributed under the terms of the license provided in [`LICENSE`](LICENSE).
