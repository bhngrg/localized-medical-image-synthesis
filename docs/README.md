# Documentation

This directory contains the technical documentation for the localized medical
image synthesis repository.

Start with the root [`README.md`](../README.md) for the recommended workflow
and the order in which the main scripts should be run. The documents below
provide details for individual parts of that workflow.

## Documentation Guide

- [`../data/README.md`](../data/README.md) — BraTS 2020 acquisition,
  registration, H5 reconstruction, manifests, and machine-specific paths.
- [`architecture.md`](architecture.md) — repository structure and separation of
  responsibilities.
- [`data_flow.md`](data_flow.md) — compact data flow from BraTS inputs through
  synthesis and downstream evaluation.
- [`br_lora_pipeline.md`](br_lora_pipeline.md) — BR-LoRA training, inference,
  posterior artifacts, and regional composition.
- [`synthetic_library.md`](synthetic_library.md) — frozen 10,000-case BR-LoRA
  synthetic-library design, production, auditing, and acceptance.
- [`../downstream_evaluation/README.md`](../downstream_evaluation/README.md) —
  operational instructions for the downstream segmentation experiments.
- [`downstream_evaluation.md`](downstream_evaluation.md) — scientific design and
  data contracts for downstream evaluation.
- [`ucsf_pdgm_external_validation.md`](ucsf_pdgm_external_validation.md) —
  UCSF-PDGM acquisition, frozen 202-subject cohort, cohort validation,
  preprocessing, and external segmentation evaluation.
- [`reproducibility.md`](reproducibility.md) — seeds, manifests, checksums,
  audits, and provenance.

## Scope

The documentation describes workflows and artifacts that are currently
implemented in the repository.

Generated images, checkpoints, large caches, local datasets, and other
machine-specific runtime artifacts are not distributed through Git.
