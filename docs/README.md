# Documentation

This directory contains extended technical documentation for the localized medical image synthesis repository.

The root `README.md` provides the high-level project overview. The documents here describe the engineering and reproducibility details of the BR-LoRA workflow, external screening pipeline, synthetic library construction, and downstream evaluation infrastructure.

## Documents

- [`architecture.md`](architecture.md) — repository and system architecture
- [`br_lora_pipeline.md`](br_lora_pipeline.md) — end-to-end BR-LoRA workflow
- [`synthetic_library.md`](synthetic_library.md) — 10,000-case synthetic library design and production
- [`downstream_evaluation.md`](downstream_evaluation.md) — downstream evaluation contracts and planned analyses
- [`reproducibility.md`](reproducibility.md) — seeds, manifests, checksums, audits, and provenance
- [`data_flow.md`](data_flow.md) — compact data-flow overview from BraTS inputs to downstream evaluation

## Scope

These documents are intended to explain how the repository works without duplicating the shorter directory-level READMEs. Generated images, checkpoints, large caches, and cluster-side production artifacts are intentionally not stored in Git.
