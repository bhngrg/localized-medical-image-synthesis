# Localized Medical Image Synthesis

A modular research framework for donor-conditioned localized medical image
synthesis with Bayesian Regional LoRA (BR-LoRA), compatibility-constrained
regional composition, synthetic-data generation, and downstream evaluation.

BR-LoRA adapts a frozen conditional diffusion model by placing mean-field
Gaussian distributions over low-rank adapter parameters. The implemented
workflow uses tumor-free base images, tumor-containing donor images, and donor
lesion masks to synthesize localized pathology while preserving unaffected
base anatomy through hard regional composition.

The repository includes the complete implemented workflow from BraTS 2020 data
registration through BR-LoRA training, nnU-Net screening, frozen synthetic
library construction, downstream segmentation, and external validation on
UCSF-PDGM.

---

## Workflow at a Glance

The main execution order is:

```text
BraTS 2020 Training                     BraTS 2020 Validation
        │                                       │
        ▼                                       ▼
register_dataset.py             register_validation_dataset.py
        │                                       │
        ├───────────────┐               ┌───────┘
        │               │               │
        ▼               │               ▼
build_h5_dataset.py     │       prepare_nnunet_dataset.py
        │               │               │
        ▼               │               ▼
create_dataset_manifest.py      nnU-Net five-fold training
        │                               │
        ▼                               ▼
baseline diffusion              validation prediction
        │                               │
        ▼                               ▼
BR-LoRA training                validation slice screening
        │                               │
        │                               ▼
        │                      compatibility audits
        │                               │
        │                               ▼
        │                      frozen 250-case cohort
        │                               │
        └───────────────┬───────────────┘
                        ▼
              frozen 10,000-case design
                        │
                        ▼
                synthetic library
                        │
                        ▼
            downstream segmentation
              ┌─────────┼─────────┐
              ▼         ▼         ▼
          real only   real +    real +
                      posterior posterior
                        mean    sampling
              └─────────┼─────────┘
                        ▼
              UCSF-PDGM evaluation
```

BraTS training and validation registration are independent and can be run in
parallel. The baseline/BR-LoRA training branch can proceed once the registered
training data and H5 representation are available. The nnU-Net screening branch
requires both registered BraTS releases.

Once their prerequisites are available, the three downstream segmentation
regimes are independent. The real-only regime does not require the synthetic
library and can therefore start earlier. UCSF-PDGM evaluation requires all
three downstream checkpoints.

---

## 1. Configure Machine-Specific Paths

Copy the example path configuration:

```bash
cp data/folders.example.yaml data/folders.yaml
```

`data/folders.yaml` is machine-specific and is not committed to Git.

Important path keys used by the implemented workflow include:

```yaml
data_root: null
yaml_dataset_path: null
validation_data_root: null
yaml_validation_dataset_path: null

h5_root: null

nnunet_archive_root: null
nnunet_run_root: null

br_lora_library_root: null
downstream_real_training_manifest: null
downstream_synthetic_manifest: null
downstream_validation_manifest: null

ucsf_pdgm_root: null
ucsf_pdgm_metadata_root: null
ucsf_pdgm_manifest: null
```

See [`data/README.md`](data/README.md) for the complete BraTS training and
validation setup and [`data/folders.example.yaml`](data/folders.example.yaml)
for the supported machine-specific path configuration.

---

## 2. Register BraTS 2020 Training and Validation Data

Both the official BraTS 2020 training and validation releases are required by
the complete workflow.

Training registration:

```bash
python scripts/register_dataset.py --help
```

Validation registration:

```bash
python scripts/register_validation_dataset.py --help
```

These two registration steps are independent and may run in parallel.

The resulting registered dataset specifications are:

```text
dataset.yaml
validation_dataset.yaml
```

Dataset registration performs the expensive source-data validation once.
Downstream workflows reuse the registered specifications rather than repeating
the full scan.

See [`data/README.md`](data/README.md).

---

## 3. Reconstruct the Training H5 Representation

The synthesis model uses the validated H5 representation of the BraTS training
release.

```bash
python scripts/build_h5_dataset.py --help
python scripts/create_dataset_manifest.py --help
```

The complete reconstructed training representation contains 57,195 axial H5
slices from 369 BraTS training subjects.

The generated manifest is subsequently used by baseline training, BR-LoRA
training, donor-pool construction, and downstream data preparation.

---

## 4. Train the Baseline Conditional Diffusion Model

The modular baseline preserves the original notebook behavior as the default
configuration.

```bash
python scripts/train_patch_x0.py --help
```

The original notebook remains the scientific reference implementation; the
modular Python implementation is the primary executable workflow.

For the production generator used in downstream synthetic-data experiments,
the final model uses the full BraTS training configuration:

```yaml
data:
  split_mode: full_train
```

The earlier internal 90/10 split remains supported in the code for development
and provenance but is not part of the main downstream production path.

---

## 5. Train BR-LoRA

BR-LoRA adapts the frozen baseline diffusion backbone using Bayesian low-rank
adapters.

```bash
python scripts/train_br_lora.py --help
```

The production configuration is defined in the tracked configuration files
under [`configs/`](configs/).

BR-LoRA supports posterior-mean and posterior-sampled inference while retaining
the frozen diffusion backbone.

For implementation and training details, see
[`docs/br_lora_pipeline.md`](docs/br_lora_pipeline.md).

Posterior products can be generated and audited with:

```bash
python scripts/audit_br_lora_posterior.py --help
python scripts/analyze_br_lora_posterior_convergence.py --help
python scripts/analyze_br_lora_posterior_mcse.py --help
```

The convergence and MCSE analyses are independent once the posterior
realization artifact has been generated.

---

## 6. Prepare and Run nnU-Net Screening

nnU-Net screening is required because the BraTS 2020 validation release has no
tumor labels. A five-fold nnU-Net ensemble is used to identify tumor-free
candidate base slices before compatibility-constrained pairing.

```bash
python screening/brats_nnunet/scripts/prepare_nnunet_dataset.py --help
python screening/brats_nnunet/scripts/prepare_nnunet_dataset.py --validate-only
```

Production nnU-Net training and inference require a **CUDA-capable GPU**.
Five folds can be trained concurrently using:

```text
screening/brats_nnunet/training/train_all_folds.slurm
```

Validation prediction uses:

```text
screening/brats_nnunet/inference/predict_validation.slurm
```

See [`screening/brats_nnunet/README.md`](screening/brats_nnunet/README.md)
and the [official nnU-Net documentation](https://github.com/MIC-DKFZ/nnUNet).

---

## 7. Screen Validation Slices and Freeze the External Cohort

After nnU-Net prediction:

```bash
python screening/brats_nnunet/scripts/screen_validation_slices.py --help
python screening/brats_nnunet/scripts/screen_validation_slices.py --validate-only
```

The screening and compatibility scripts then identify tumor-free bases,
evaluate donor compatibility, and freeze the definitive 250-case external
cohort.

All scripts use deterministic locations under `nnunet_run_root` when explicit
paths are omitted.

See [`screening/brats_nnunet/README.md`](screening/brats_nnunet/README.md)
for the complete audit and matching sequence.

---

## 8. Construct the Frozen 10,000-Case Design

The canonical 10,000-case BR-LoRA design is tracked under:

```text
downstream_evaluation/manifests/br_lora_library_design_10000/
```

It contains 125 validation subjects, 80 cases per subject, 40 batches of 250,
and compatibility-constrained donor assignments.

For reproducible reconstruction:

```bash
python screening/brats_nnunet/scripts/design_br_lora_library_10000.py --help
```

`--output-dir` is intentionally required to prevent accidental overwrite of the
tracked canonical design.

See [`docs/synthetic_library.md`](docs/synthetic_library.md).

---

## 9. Produce the Synthetic Library

Synthetic-library production uses the frozen design and trained BR-LoRA
checkpoint.

```bash
python scripts/run_br_lora_library_batch.py --help
python scripts/accept_br_lora_library_batch.py --help
```

Generated batches are audited, hashed, staged, and accepted into the permanent
library. Independent batches can run in parallel when they use distinct batch
and staging locations.

See [`docs/synthetic_library.md`](docs/synthetic_library.md).

---

## 10. Train the Downstream Segmentation Models

The downstream comparison uses three regimes:

```text
real_only
real_plus_br_lora_mean
real_plus_br_lora_posterior
```

The frozen split contains 332 BraTS subjects for training and 37 held-out
subjects for internal validation.

```bash
python scripts/train_downstream_segmentation.py --help
```

The real-only experiment can run before synthetic-library production finishes.
The two augmented regimes can run independently once the library is available.

The downstream U-Net is adapted from
[Low-Grade-Glioma-Segmentation](https://github.com/edaaydinea/Low-Grade-Glioma-Segmentation).

See [`downstream_evaluation/README.md`](downstream_evaluation/README.md) and
[`docs/downstream_evaluation.md`](docs/downstream_evaluation.md).

---

## 11. Evaluate on UCSF-PDGM

The three downstream checkpoints are externally evaluated on the frozen
202-subject UCSF-PDGM cohort.

Official dataset source:

[UCSF-PDGM on The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/)

Validate cohort provenance:

```bash
python downstream_evaluation/scripts/validate_ucsf_pdgm_external_cohort.py --help
```

Run evaluation:

```bash
python -m downstream_evaluation.segmentation.evaluate_ucsf_pdgm --help
```

See [`docs/ucsf_pdgm_external_validation.md`](docs/ucsf_pdgm_external_validation.md)
for cohort derivation, preprocessing, and evaluation details.

---

## Additional BR-LoRA Evaluation

BR-LoRA itself can also be evaluated on the fixed official BraTS validation
cohort independently of the downstream segmentation experiment:

```bash
python scripts/evaluate_br_lora_external.py --help
```

This is distinct from the downstream UCSF-PDGM segmentation evaluation.

---

## Documentation

Focused documentation is organized as follows:

- [`data/README.md`](data/README.md) — BraTS training and validation setup
- [`screening/brats_nnunet/README.md`](screening/brats_nnunet/README.md) — nnU-Net screening and compatibility workflow
- [`docs/br_lora_pipeline.md`](docs/br_lora_pipeline.md) — BR-LoRA implementation workflow
- [`docs/synthetic_library.md`](docs/synthetic_library.md) — frozen synthetic-library design and production
- [`downstream_evaluation/README.md`](downstream_evaluation/README.md) — downstream segmentation workflow
- [`docs/downstream_evaluation.md`](docs/downstream_evaluation.md) — downstream scientific design
- [`docs/ucsf_pdgm_external_validation.md`](docs/ucsf_pdgm_external_validation.md) — UCSF-PDGM external validation
- [`docs/reproducibility.md`](docs/reproducibility.md) — reproducibility conventions
- [`docs/architecture.md`](docs/architecture.md) — repository architecture
- [`docs/data_flow.md`](docs/data_flow.md) — data-flow summary
- [`docs/README.md`](docs/README.md) — documentation index

---

## Reproducibility

The repository uses registered dataset specifications, deterministic manifests,
fixed seeds, frozen case assignments, checkpoint provenance, strict downstream
training controls, and SHA-256 inventories to make experimental artifacts
traceable.

Large generated datasets, checkpoints, synthetic images, and posterior tensors
are not stored in Git. The repository instead retains the code, configuration,
frozen manifests, audit outputs, and provenance required to reconstruct the
implemented workflows.

See [`docs/reproducibility.md`](docs/reproducibility.md).

---

## Citation

If you use this repository in academic work, please cite the BraTS dataset,
relevant third-party software used in the workflow, and the corresponding
BR-LoRA manuscript or repository release when available.

Specific dataset and software attribution is documented in the relevant
workflow documentation.

---

## License

This repository is distributed under the terms of the license provided in
[`LICENSE`](LICENSE).
