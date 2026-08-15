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
scripts/run_br_lora_library_pipeline.py
scripts/run_br_lora_library_all_remaining.sh
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
local production audit
        │
        ▼
Mac SHA-256 inventory
        │
        ▼
rsync transfer to Falcon
        │
        ▼
Falcon SHA-256 inventory
        │
        ▼
exact checksum comparison
        │
        ▼
Falcon acceptance
        │
        ▼
master-library manifest update
        │
        ▼
local staging cleanup
```

## Production Audits

A normal completed batch contains 250 case directories and 1,501 files:

```text
250 × 6 per-case artifacts = 1,500 files
1 × evaluation_summary.json
```

The production audit verifies the expected case count, posterior-sampling contract, metadata consistency, and artifact presence.

The transfer audit independently recomputes SHA-256 hashes on Falcon and requires an exact match with the Mac-side inventory before acceptance.

## Falcon Acceptance

Falcon-side acceptance verifies:

- transferred file count;
- Mac/Falcon checksum identity;
- production audit status;
- execution-manifest checksum;
- current master-library state;
- uniqueness constraints;
- batch order;
- artifact references.

On success, the master manifest is promoted and a pre-promotion snapshot is retained.

## Failure Recovery

The orchestration scripts are fail-fast. If any stage fails:

- later batches are not started;
- local staging is not deleted;
- the failed batch can be inspected and rerun;
- the master manifest is not promoted unless acceptance succeeds.

This behavior is intentional and should be preserved during future refactoring.
