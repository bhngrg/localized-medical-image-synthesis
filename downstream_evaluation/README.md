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

New hardened runs are written under:

```text
outputs/downstream_segmentation/runs/
```

Run directories are non-overwriting and include the regime, seed, partition,
and Slurm job ID when available.

Each training run records:

- resolved experiment configuration
- manifest paths and SHA-256 hashes
- Git commit
- Python/PyTorch/CUDA environment
- GPU and Slurm metadata
- reproducibility settings
- training history
- best checkpoint and checkpoint hash

Large runtime outputs remain outside version control. Selected logs that are
useful for scientific or reproducibility provenance are curated under:

```text
downstream_evaluation/logs/
```

The files under `downstream_evaluation/results/` currently summarize the
preliminary pre-hardening runs and are retained as historical evidence rather
than overwritten.

## External Evaluation

The independent external validation cohort is derived from UCSF-PDGM after
removing patients overlapping with BraTS. The frozen external cohort contains
202 subjects.

External evaluation uses the repository's established FLAIR preprocessing
contract and reports slice-level metrics as well as subject-level volumetric
Dice and IoU.

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
