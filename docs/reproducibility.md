# Reproducibility

## Overview

Reproducibility is treated as a first-class requirement throughout the repository.

The project relies on frozen manifests, deterministic seeding, checksums, explicit audits, and preserved provenance rather than implicit data discovery.

## Frozen Inputs

Important frozen inputs include:

- registered dataset specifications;
- training manifests;
- downstream subject splits;
- downstream donor pools;
- screened external-cohort manifests;
- BR-LoRA checkpoints;
- the 10,000-case library design;
- per-batch design manifests;
- per-batch external-evaluation manifests.

## Random Seeds

The repository uses explicit seeds for training, design, and evaluation.

External BR-LoRA evaluation derives a stable per-case seed from:

```text
evaluation_seed + case_id
```

using SHA-256 rather than Python's process-dependent hash function. This ensures that case-level posterior sampling does not depend on manifest order.

## Checksums

SHA-256 inventories are used for:

- frozen design artifacts;
- local batch production files;
- Falcon batch production files;
- master-library manifests.

A batch is accepted only after the Mac-side and Falcon-side inventories match exactly.

## Production Audits

The library workflow records:

- local production audit;
- Mac checksum inventory;
- Falcon checksum inventory;
- Falcon acceptance audit;
- pre-promotion master-manifest snapshot.

These artifacts provide an audit trail from generation through acceptance.

## Git Provenance

Training and evaluation metadata include the current Git commit when available.

Generated artifacts should also record:

- script path;
- script checksum where appropriate;
- configuration paths;
- source-manifest paths;
- dataset specification paths;
- checkpoint path;
- timestamps.

## Large Artifacts

The following should remain outside Git unless there is a specific reason to version them:

- model checkpoints;
- posterior realization stacks;
- synthetic image batches;
- nnU-Net predictions;
- large compatibility caches;
- cluster scratch outputs.

Git should contain the code, configuration, small manifests, audit summaries, and documentation required to reproduce those artifacts.

## Public-Repository Hygiene

Before public release, verify that the repository does not contain:

- private SSH keys;
- passwords or API tokens;
- secrets;
- private credentials;
- unnecessary machine-specific caches;
- large generated binaries.

Machine-specific example paths may appear in historical logs or cluster scripts, but user-facing documentation should prefer placeholders or configurable paths.
