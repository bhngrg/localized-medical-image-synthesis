# Preliminary External Validation Results

## UCSF-PDGM External Segmentation Evaluation

These results are preliminary downstream results obtained before the planned
code generalization and reproducibility hardening. They are preserved here as
a scientific checkpoint and should not yet be treated as the final
reproducibility-verified results.

## External Cohort

- Dataset: UCSF-PDGM Version 5
- Independent baseline subjects: 202
- Follow-up examinations excluded
- BraTS-overlapping subjects excluded
- Total axial slices evaluated: 31,310
- Tumor-positive slices: 11,572
- FLAIR input
- Whole-tumor target: segmentation > 0
- Evaluation threshold: 0.5
- Image geometry: 240 x 240 x 155
- External preprocessing matched the BraTS model-facing preprocessing:
  per-slice standardization followed by positive-pixel percentile normalization

## Training Regimes

1. Real BraTS training data only
2. Real BraTS + 10,000 BR-LoRA posterior-mean synthetic images
3. Real BraTS + 10,000 BR-LoRA posterior-sampling synthetic cases

Each model was evaluated on the same 202 UCSF-PDGM subjects.

## Primary External Metric

The primary external comparison is subject-level volumetric Dice. A single
3D Dice value was computed for each subject across the complete 155-slice
volume. Results are summarized across the 202 subjects as mean +/- standard
deviation.

| Training regime | Mean volumetric Dice +/- SD | 95% bootstrap CI |
|---|---:|---:|
| Real only | 0.682 +/- 0.206 | [0.653, 0.709] |
| Real + BR-LoRA posterior mean | 0.798 +/- 0.164 | [0.774, 0.819] |
| Real + BR-LoRA posterior sampling | 0.795 +/- 0.173 | [0.771, 0.818] |

## Paired Improvements in Volumetric Dice

Because all three models were evaluated on the same 202 subjects, model
comparisons were performed using paired subject-level volumetric Dice
differences.

| Comparison | Mean paired difference | 95% bootstrap CI |
|---|---:|---:|
| Posterior mean - Real only | +0.116 | [+0.102, +0.131] |
| Posterior sampling - Real only | +0.113 | [+0.099, +0.128] |
| Posterior sampling - Posterior mean | -0.0025 | [-0.0099, +0.0046] |

Both BR-LoRA synthetic augmentation strategies improved external volumetric
Dice relative to real-only training. The posterior-mean and
posterior-sampling augmentation strategies performed similarly; the paired
confidence interval for their difference includes zero.

## Secondary Subject-Level Metrics

### Volumetric IoU

| Training regime | Mean volumetric IoU +/- SD |
|---|---:|
| Real only | 0.550 +/- 0.214 |
| Real + BR-LoRA posterior mean | 0.688 +/- 0.183 |
| Real + BR-LoRA posterior sampling | 0.687 +/- 0.193 |

### Mean Tumor-Positive-Slice Dice Per Subject

| Training regime | Mean +/- SD |
|---|---:|
| Real only | 0.656 +/- 0.180 |
| Real + BR-LoRA posterior mean | 0.703 +/- 0.165 |
| Real + BR-LoRA posterior sampling | 0.701 +/- 0.178 |

### Mean Tumor-Positive-Slice IoU Per Subject

| Training regime | Mean +/- SD |
|---|---:|
| Real only | 0.567 +/- 0.184 |
| Real + BR-LoRA posterior mean | 0.627 +/- 0.169 |
| Real + BR-LoRA posterior sampling | 0.623 +/- 0.182 |

## Slice-Level Metrics

These metrics reproduce the internal-validation style of averaging over
individual axial slices and are retained for direct comparability. They are
secondary to the subject-level volumetric analysis because slices belonging
to the same patient are not independent.

| Training regime | All-slice Dice | Tumor-positive-slice Dice |
|---|---:|---:|
| Real only | 0.7190 | 0.6759 |
| Real + BR-LoRA posterior mean | 0.8224 | 0.7194 |
| Real + BR-LoRA posterior sampling | 0.8250 | 0.7171 |

## Internal Validation Context

The selected checkpoints had similar tumor-positive validation Dice on the
held-out BraTS internal validation split:

| Training regime | Internal tumor-positive Dice |
|---|---:|
| Real only | 0.736496 |
| Real + BR-LoRA posterior mean | 0.732357 |
| Real + BR-LoRA posterior sampling | 0.731261 |

Thus, the substantially stronger performance of the synthetic-augmented
models appeared on the independent UCSF-PDGM cohort rather than on the
internal BraTS validation split.

## Bootstrap Analysis

- Unit of resampling: subject
- Number of subjects: 202
- Bootstrap replicates: 100,000
- Random seed: 2026
- Confidence interval: percentile 95% interval
- Model comparisons used paired subject-level resampling

## Checkpoints

### Real only

`outputs/downstream_segmentation/real_only_seed42_a30_normal_q/best_model.pt`

- Selected epoch: 20
- Internal tumor-positive validation Dice: 0.7364959729295704

### Real + BR-LoRA posterior mean

`outputs/downstream_segmentation/real_plus_br_lora_seed42_a30_normal_q_rerun/best_model.pt`

- Selected epoch: 20
- Internal tumor-positive validation Dice: 0.7323571678980253
- This checkpoint is from the completed rerun selected after the original
  posterior-mean training job reached its walltime.

### Real + BR-LoRA posterior sampling

`outputs/downstream_segmentation/real_plus_br_lora_posterior_seed42_a30_normal_q/best_model.pt`

- Selected epoch: 20
- Internal tumor-positive validation Dice: 0.7312612908345795

## External Evaluation Provenance

- Slurm job: 554517
- State: COMPLETED
- Exit code: 0:0
- Elapsed time: 00:07:07
- Partition: a30_normal_q
- Node: fal015
- GPU: NVIDIA A30
- Conda environment: brats-nnunet
- Python: 3.11.15
- PyTorch: 2.13.0+cu126
- Evaluation stderr: empty

External outputs are stored under:

`outputs/downstream_segmentation/ucsf_pdgm_external_202/`

Each experiment contains:

- `summary.csv`
- `subject_metrics.csv`
- `slice_metrics.csv`

## Interpretation

The current preliminary results support the downstream hypothesis that
augmenting real BraTS training data with BR-LoRA-generated images improves
segmentation generalization to an independent external cohort.

Both posterior-mean and posterior-sampling augmentation produced substantial
improvements over real-only training in subject-level volumetric Dice.
There is currently no evidence from this comparison that one of the two
BR-LoRA augmentation strategies is meaningfully superior to the other.

These results should be retained as preliminary evidence. Final reported
results should be generated again after downstream code generalization,
determinism/reproducibility hardening, and final reproducibility audits.
