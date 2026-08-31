# BR-LoRA Synthetic Library

## Purpose

The BR-LoRA synthetic library provides a fixed, reproducible image set for downstream segmentation and reliability experiments.

The production target is:

```text
10,000 synthetic cases
40 batches
250 cases per batch
125 external BraTS validation subjects
80 final cases per external subject
```

The first 250 cases were preserved as the original frozen Batch 0001. The remaining 9,750 cases were designed under the same scientific constraints and then divided into Batches 0002–0040.

## Frozen Design

The main design artifact is:

```text
downstream_evaluation/manifests/br_lora_library_design_10000/
    br_lora_library_design_10000.csv
```

Associated artifacts include:

```text
br_lora_library_design_10000_audit.csv
br_lora_library_design_10000_summary.json
br_lora_library_design_10000_sha256.txt
batches/
```

A recomputable training-only compatibility cache is intentionally excluded from Git.

## Design Constraints

The frozen design enforces:

- 10,000 total library cases;
- 80 cases per external subject;
- compatibility between each external base and donor lesion mask;
- minimum mask-inside-brain fraction of 0.80;
- no donor-slice reuse;
- donor-subject cap of 31 cases;
- preservation of the original Batch 0001;
- deterministic library identifiers and pair keys.

The final design audit verifies:

- exact library size;
- unique library case IDs;
- unique pair keys;
- unique donor slices;
- exactly 125 external subjects;
- exactly 80 cases per external subject;
- donor-subject cap compliance;
- 40 batches of 250 cases;
- exact preservation of Batch 0001.

## Compatibility and Matching

External base masks and downstream-training donor masks are compared under the established anatomical compatibility criterion.

The production design uses a sparse exact max-flow formulation for donor assignment. Candidate donor lists are limited per slot for computational tractability while preserving feasibility.

The matching enforces:

- one donor slice per planned slot;
- global donor-slice uniqueness;
- frozen Batch 0001 donor occupancy;
- donor-subject capacity limits.

## Batch Artifacts

Each batch has a frozen design manifest:

```text
batch_0007_manifest.csv
```

For production inference, a corresponding external-evaluation manifest is generated:

```text
batch_0007_external_evaluation_manifest.csv
```

The external-evaluation manifest contains only the fields required by the BR-LoRA evaluator.

## Production Pipeline

Primary scripts:

```text
scripts/run_br_lora_library_batch.py
scripts/accept_br_lora_library_batch.py
screening/brats_nnunet/scripts/design_br_lora_library_10000.py
```

The batch production sequence is:

```text
frozen batch manifest
        │
        ▼
BR-LoRA posterior generation
        │
        ▼
production audit
        │
        ▼
canonical SHA-256 inventory
        │
        ▼
staging area
        │
        ▼
batch integrity verification
        │
        ▼
batch acceptance
        │
        ▼
permanent library
        │
        ▼
master-library manifest update
```

## Production Audits

A normal completed batch contains 250 case directories and 1,501 files:

```text
250 × 6 per-case artifacts = 1,500 files
1 × evaluation_summary.json
```

The production audit verifies the expected case count, posterior-sampling contract, metadata consistency, and artifact presence.

The production checksum inventory provides the canonical integrity record for the completed staged batch. Acceptance independently recomputes the staged file hashes and requires an exact match before promotion.

## Batch Acceptance

Batch acceptance verifies:

- staged file count;
- exact agreement with the production SHA-256 inventory;
- production audit status;
- execution-manifest checksum;
- current master-library state;
- uniqueness constraints;
- batch order;
- artifact references.

After these checks pass, the batch is copied into the permanent library and verified again against the production checksum inventory. The supporting design, execution-manifest, audit, and checksum files are copied separately and verified by exact source/destination SHA-256 comparison.

On success, the master manifest is promoted and a pre-promotion snapshot is retained.

## Failure Recovery

Production and acceptance are deliberately separate. If either stage fails:

- the staged batch remains available for inspection;
- the failed batch can be corrected or rerun;
- the permanent master manifest is not promoted unless acceptance succeeds.

Acceptance preserves the staging copy after successful promotion; staging cleanup is an explicit user-controlled operation.

Infrastructure-specific Mac-to-Falcon orchestration used for the original library build is retained under `scripts/historical/` for provenance only and is not part of the supported public workflow.

## Downstream Posterior Shard Cache

The accepted synthetic library is the canonical source artifact for downstream
BR-LoRA augmentation. In particular, the accepted per-case
`posterior_samples.pt` files remain the source of truth for posterior-sampling
experiments.

For downstream segmentation, the repository can derive an optional
epoch-specific shard cache from those accepted posterior files to reduce
shared-filesystem file-open and metadata overhead. The cache is not part of
synthetic-library generation or batch acceptance, and it is not promoted into
the permanent library.

Cache construction preserves the exact deterministic downstream
case-to-realization schedule and copies the selected posterior tensors without
numerical transformation. Written shards are reloaded and checked with exact
`torch.equal` comparison, and per-shard SHA-256 hashes plus source provenance
are recorded in `cache_manifest.json`.

The cache builder is:

```text
downstream_evaluation/segmentation/build_posterior_shard_cache.py
```

Operational and methodological details are documented in:

```text
downstream_evaluation/README.md
docs/downstream_evaluation.md
```
