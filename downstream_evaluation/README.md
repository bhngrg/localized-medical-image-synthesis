# Downstream Segmentation Evaluation

This directory contains the downstream tumor-segmentation evaluation used to
assess whether BR-LoRA synthetic images improve the utility of the real BraTS
training data.

The evaluation compares three training regimes using the same segmentation
architecture, preprocessing contract, validation cohort, loss, optimizer, and
evaluation metrics:

1. `real_only`
   - Real BraTS training data only.

2. `real_plus_br_lora_mean`
   - Real BraTS training data plus 10,000 fixed BR-LoRA posterior-mean
     synthetic cases.

3. `real_plus_br_lora_posterior`
   - Real BraTS training data plus 10,000 BR-LoRA synthetic cases.
   - One deterministic posterior realization is selected per synthetic case
     per epoch.
   - The current 20-epoch training run therefore uses 20 distinct posterior
     realizations per case from the 100 stored realizations.

## Scientific Defaults

The tracked configuration is:

```text
downstream_evaluation/configs/segmentation.yaml
```

The current defaults preserve the preliminary downstream experiments:

- seed: 42
- image channel: 0 (FLAIR)
- batch size: 26
- workers: 4
- epochs: 20
- learning rate: 0.001
- segmentation threshold: 0.5
- optimizer: Adamax
- loss: BCE-with-logits + soft Dice
- model: vanilla 2D U-Net

Expected frozen data counts are:

- real training slices: 41,460
- synthetic cases: 10,000
- combined augmented training samples: 51,460
- validation slices: 5,735

## Machine-Specific Paths

Machine-specific paths are stored in:

```text
data/folders.yaml
```

This file is ignored by Git.

Users may provide paths explicitly on the command line. Explicit CLI paths are
saved to `data/folders.yaml`, allowing `--validate-only` to serve as a safe
setup step before launching a large training run.

See:

```text
data/folders.example.yaml
```

for the available path keys.

## Posterior-Sampling Shard Cache

The posterior-sampling regime is defined by the original BR-LoRA posterior
library and its deterministic case-specific realization schedule. The original
posterior library remains the source of truth and is not modified by the cache
workflow.

Each synthetic case stores 100 retained posterior realizations. For training
seed `42`, the realization order for a synthetic case with `library_index` is
defined by:

```text
numpy.default_rng(42 + library_index).permutation(100)
```

The current 20-epoch downstream experiment uses the first 20 entries of that
case-specific permutation, one realization per epoch.

Reading one realization at a time from the original per-case posterior files
creates substantial shared-filesystem and file-open overhead. The optional
posterior shard cache reorganizes the exact selected tensors into larger,
epoch-specific shard files. This is a storage/I/O optimization only: it does
not change the frozen synthetic-library design, posterior samples, seed,
case-to-realization assignment, epoch schedule, masks, preprocessing, or
training objective.

The cache builder is:

```text
downstream_evaluation/segmentation/build_posterior_shard_cache.py
```

The Falcon Slurm launcher is:

```text
downstream_evaluation/segmentation/build_posterior_shard_cache.slurm
```

The default cache-build contract is:

```text
seed: 42
epochs: 20
posterior realizations available per case: 100
synthetic cases: 10,000
shard size: 500 cases
```

For 10,000 cases and a shard size of 500, this produces 20 shards per epoch
and 400 shard files across 20 epochs.

The builder is deliberately non-overwriting. The requested output directory
must not already exist. The original posterior library is read-only from the
builder's perspective.

Each written shard is immediately reloaded and checked using exact
`torch.equal` comparison against the source-derived selected tensor. The
builder also verifies seed, epoch, library-index, and original
posterior-realization metadata. A SHA-256 hash is recorded for every verified
shard.

After all shards have been built successfully, the cache root contains a
`cache_manifest.json` recording the source manifest and its SHA-256 hash, the
source library root, the deterministic schedule rule, shard layout, shard
hashes, and aggregate verification status.

A machine-specific cache location may be recorded with:

```text
downstream_posterior_shard_cache_root
```

in `data/folders.yaml`.

Cache-backed posterior training is implemented as an explicit opt-in path.
The original per-case posterior loader remains the repository fallback when no
cache root is configured. A machine can opt in persistently by setting
`downstream_posterior_shard_cache_root` in its ignored `data/folders.yaml`, or
for a single invocation by passing `--posterior-shard-cache-root`.

The resolution order is:

```text
--posterior-shard-cache-root
    >
data/folders.yaml: downstream_posterior_shard_cache_root
    >
original per-case posterior loader
```

No tracked configuration hard-codes a machine-specific shard-cache location.

Before launching a cache-backed training run, validate the complete downstream
data contract with:

```bash
python scripts/train_downstream_segmentation.py \
  --regime real_plus_br_lora_posterior \
  --posterior-shard-cache-root /path/to/verified/cache \
  --device cpu \
  --validate-only
```

The shared A30 launcher forwards the same explicit trainer option:

```bash
sbatch downstream_evaluation/segmentation/train_downstream_segmentation_a30.slurm \
  real_plus_br_lora_posterior \
  --posterior-shard-cache-root /path/to/verified/cache
```

When a cache is used, run metadata and checkpoints record the cache root,
cache-manifest path and SHA-256, and the posterior loader mode. This preserves
an explicit provenance distinction between the canonical per-case posterior
library and its derived shard-cache representation.

## Validate Before Training

The unified user-facing entry point is:

```text
scripts/train_downstream_segmentation.py
```

A configuration/data-contract check can be run without starting training:

```bash
python scripts/train_downstream_segmentation.py \
  --regime real_only \
  --device cpu \
  --validate-only
```

For BR-LoRA augmentation:

```bash
python scripts/train_downstream_segmentation.py \
  --regime real_plus_br_lora_mean \
  --device cpu \
  --validate-only
```

or:

```bash
python scripts/train_downstream_segmentation.py \
  --regime real_plus_br_lora_posterior \
  --device cpu \
  --validate-only
```

Validation checks the configured paths, manifests, regime-specific dataset
contracts, and expected sample counts. It does not start training or create a
training-run output directory.

## Reproducibility

The hardened downstream implementation uses explicit reproducibility controls:

- Python seed
- NumPy seed
- PyTorch CPU seed
- PyTorch CUDA seeds
- deterministic cuDNN
- cuDNN benchmarking disabled
- `torch.use_deterministic_algorithms(True)`
- explicit DataLoader generators
- deterministic worker and Albumentations transform seeding
- separate training and validation DataLoader generators
- deterministic posterior-sample schedules for the posterior-sampling regime

CUDA jobs additionally require:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

The strict CUDA configuration was tested on an NVIDIA A30 using a dedicated
reproducibility diagnostic. Two independently reinitialized runs within the
same Slurm job and GPU allocation produced identical loss sequences and
identical final model-state hashes.

Cross-job reproducibility is evaluated separately through the hardened
production reruns.

## Slurm

The shared A30 launcher is:

```text
downstream_evaluation/segmentation/train_downstream_segmentation_a30.slurm
```

It uses the `fdtbiotech` Slurm account and the `a30_normal_q` partition.

Example:

```bash
sbatch downstream_evaluation/segmentation/train_downstream_segmentation_a30.slurm \
  real_only \
  --validate-only
```

The same launcher supports all three regimes.

## Outputs and Provenance

Persistent downstream segmentation training bundles are written under:

```text
checkpoints/downstream_segmentation/
```

The canonical directory is organized by training condition and seed. Each
bundle contains the best checkpoint together with the resolved configuration,
run metadata, and training history.

When a default canonical training destination already contains a previous
bundle, that bundle is preserved before the new run begins under:

```text
checkpoints/historical/downstream_segmentation/
```

Explicit user-specified output directories remain overwrite-protected rather
than being moved automatically.

Each training run records:

- resolved experiment configuration
- manifest paths and SHA-256 hashes
- Git commit
- Python/PyTorch/CUDA environment
- GPU and Slurm metadata
- reproducibility settings
- training history
- best checkpoint and checkpoint hash

Retained execution logs that provide useful scientific or reproducibility
provenance are organized by workflow under the repository-level:

```text
logs/
```

Raw generated workflow products, including external-evaluation outputs, are
written under:

```text
outputs/
```

Curated scientific results intended for preservation and reporting are stored
separately under:

```text
results/
```

For the hardened UCSF-PDGM analysis, rerunning the default locked analysis
archives only the analysis artifacts owned by that script under
`results/historical/` before regenerating them. Other curated files sharing
the result directory are left untouched.

## External Evaluation

The independent external validation cohort is derived from UCSF-PDGM after
excluding records that overlap the BraTS 2021 segmentation cohort and removing
the single follow-up examination from the independent cohort. The frozen
external cohort contains 202 baseline subjects.

Validate the frozen cohort provenance with:

```bash
python downstream_evaluation/scripts/validate_ucsf_pdgm_external_cohort.py --help
```

Run external evaluation with:

```bash
python -m downstream_evaluation.segmentation.evaluate_ucsf_pdgm --help
```

External evaluation uses the repository's established FLAIR preprocessing
contract and reports slice-level metrics as well as subject-level volumetric
Dice and IoU.

See [`../docs/ucsf_pdgm_external_validation.md`](../docs/ucsf_pdgm_external_validation.md)
for acquisition, cohort derivation, preprocessing, validation, and evaluator
details.

## Reference Implementation and Attribution

The downstream segmentation idea and vanilla U-Net structure were adapted in
part from:

**edaaydinea/Low-Grade-Glioma-Segmentation**

https://github.com/edaaydinea/Low-Grade-Glioma-Segmentation

In particular, that repository provided the reference point for using a
U-Net-style segmentation task and Dice-based evaluation to assess downstream
utility.

The implementation in this repository was rewritten for the present
BraTS/UCSF-PDGM and BR-LoRA workflow. It uses this repository's own
preprocessing, manifests, synthetic-data loaders, loss and metric code,
reproducibility controls, and external-validation pipeline.
