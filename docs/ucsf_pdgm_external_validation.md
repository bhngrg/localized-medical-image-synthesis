# UCSF-PDGM External Validation

## Purpose

UCSF-PDGM is used as the independent external cohort for the downstream
segmentation evaluation.

The downstream experiment compares segmentation models trained using:

1. real BraTS training data only;
2. real BraTS data plus BR-LoRA posterior-mean synthetic images; and
3. real BraTS data plus BR-LoRA posterior-sampling synthetic images.

All three models are evaluated on the same frozen UCSF-PDGM cohort.

For instructions for training the downstream segmentation models, see
[`../downstream_evaluation/README.md`](../downstream_evaluation/README.md).

## Dataset Acquisition

UCSF-PDGM is not distributed with this repository.

Obtain the dataset from the official Cancer Imaging Archive (TCIA) collection:

https://www.cancerimagingarchive.net/collection/ucsf-pdgm/

The repository was developed and validated using UCSF-PDGM Version 5.

TCIA is the authoritative source for dataset acquisition, collection
documentation, licensing/access information, and current release details.
This repository therefore does not maintain a separate UCSF-PDGM downloader.

The TCIA collection describes 495 unique subjects and 501 imaging studies.
Some identifiers correspond to follow-up examinations of the same subject.
The collection metadata also provides mappings to the BraTS 2021 cohorts,
which are important when constructing an evaluation cohort that is independent
of BraTS.

## Machine-Specific Dataset Path

After obtaining UCSF-PDGM, configure the local dataset root in:

```text
data/folders.yaml
```

using:

```yaml
ucsf_pdgm_root: /path/to/UCSF-PDGM-v5
```

`data/folders.yaml` is machine-specific and ignored by Git.

The expected root is the directory containing subject directories such as:

```text
UCSF-PDGM-v5/
    UCSF-PDGM-0005_nifti/
        UCSF-PDGM-0005_FLAIR.nii.gz
        UCSF-PDGM-0005_tumor_segmentation.nii.gz
        ...
```

The evaluator also accepts an explicit command-line path. Explicit CLI paths
take precedence over `data/folders.yaml`.

See [`../data/folders.example.yaml`](../data/folders.example.yaml) for the
supported path keys.

## Frozen External Cohort

The external evaluation does not rediscover subjects at runtime.

The frozen subject manifest is:

```text
downstream_evaluation/manifests/ucsf_pdgm_external_202_subjects.csv
```

It contains 202 baseline subjects.

For every subject, the manifest records:

- subject identifier;
- relative FLAIR path;
- relative tumor-segmentation path;
- expected image dimensions;
- expected voxel spacing;
- expected orientation;
- whole-tumor definition;
- source cohort;
- BraTS 2021 segmentation-overlap status; and
- follow-up status.

The frozen cohort uses:

```text
modality: FLAIR
expected shape: 240 x 240 x 155
expected spacing: 1 x 1 x 1 mm
orientation: LPS
whole tumor: segmentation > 0
BraTS segmentation overlap: false
follow-up examination: false
```

The raw image used by this workflow is the subject's `FLAIR.nii.gz` image,
not `FLAIR_bias.nii.gz`.

## Cohort Derivation

The 202-subject cohort was constructed to provide baseline UCSF-PDGM cases
that do not overlap with the BraTS 2021 segmentation cohort.

The retained cohort-provenance tables are external data artifacts and are not
copied into this repository because they contain clinical metadata that is not
required by the segmentation workflow.

The derivation represented by those retained tables is:

```text
501 UCSF-PDGM records represented in the provenance tables
        |
        +-- 298 records overlapping the BraTS 2021 segmentation cohort
        |       |
        |       +-- 262 BraTS Training
        |       +--  36 BraTS Validation
        |
        +-- 203 records independent of that segmentation cohort
                |
                +-- remove one follow-up examination
                    UCSF-PDGM-0391_FU016d
                |
                +-- 202 independent baseline subjects
```

The 202 baseline identifiers match the version-controlled frozen subject
manifest exactly.

This repository-specific derivation should not be interpreted as a replacement
for TCIA's description of the complete UCSF-PDGM collection. TCIA remains the
authoritative source for collection-level subject and study accounting.

## Cohort-Provenance Validation

If the cohort-provenance CSVs are available locally, their directory can be
configured in:

```text
data/folders.yaml
```

as:

```yaml
ucsf_pdgm_metadata_root: /path/to/ucsf_pdgm/metadata
```

The expected provenance files are:

```text
brats21_segmentation_subjects_to_exclude.csv
ucsf_pdgm_independent_subjects.csv
ucsf_pdgm_independent_baseline_subjects.csv
```

Validate the derivation with:

```bash
python downstream_evaluation/scripts/validate_ucsf_pdgm_external_cohort.py
```

The validator checks, among other contracts:

- the expected 298/203/202 record counts;
- the 262/36 BraTS Training/Validation split among overlap exclusions;
- disjoint overlap and independent sets;
- absence of BraTS segmentation-cohort identifiers in the independent set;
- removal of the single follow-up record;
- exact agreement between the 202 baseline identifiers and the frozen
  repository manifest; and
- frozen-manifest overlap and follow-up flags.

The provenance CSVs are used to audit how the frozen cohort was derived.
They are not required for routine model inference once the frozen manifest has
been established.

## Image and Segmentation Contract

For each frozen subject, evaluation loads:

```text
<subject>_FLAIR.nii.gz
<subject>_tumor_segmentation.nii.gz
```

The whole-tumor target is defined as:

```text
segmentation > 0
```

The frozen cohort was audited for:

- expected 240 x 240 x 155 image geometry;
- 1 mm isotropic voxel spacing;
- LPS orientation;
- matching image and segmentation shapes;
- matching image and segmentation affine geometry;
- finite image values;
- valid segmentation labels; and
- presence of tumor in every retained subject.

The evaluator also verifies that all files referenced by the frozen manifest
exist before inference begins.

## FLAIR Preprocessing

UCSF-PDGM FLAIR slices are transformed to the same effective model-facing
representation used for the reconstructed BraTS H5 data.

For each axial slice:

1. load the FLAIR data as floating point;
2. standardize the complete 2D slice using its mean and population standard
   deviation;
3. identify positive standardized pixels as the brain region;
4. when sufficient positive pixels are present, calculate the 1st and 99th
   percentiles over those pixels;
5. clip to that range; and
6. scale the result to `[0, 1]` as `float32`.

Constant slices are handled explicitly.

This preprocessing path was numerically cross-checked against the established
BraTS H5 preprocessing contract before external evaluation.

## Validate the Evaluator Before Inference

The hardened evaluator is:

```text
downstream_evaluation/segmentation/evaluate_ucsf_pdgm.py
```

It requires the three downstream segmentation checkpoints explicitly.

A safe validation-only call is:

```bash
python -m downstream_evaluation.segmentation.evaluate_ucsf_pdgm \
  --real-only-checkpoint /path/to/real_only/best_model.pt \
  --posterior-mean-checkpoint /path/to/real_plus_br_lora_mean/best_model.pt \
  --posterior-sampling-checkpoint /path/to/real_plus_br_lora_posterior/best_model.pt \
  --validate-only
```

Validation checks the configured UCSF-PDGM root, frozen cohort manifest,
subject files, checkpoint structure, checkpoint epoch, and segmentation
threshold.

It does not run inference and does not create an evaluation output directory.

## External Evaluation

Run the evaluator without `--validate-only` after all paths and checkpoints
have passed validation:

```bash
python -m downstream_evaluation.segmentation.evaluate_ucsf_pdgm \
  --real-only-checkpoint /path/to/real_only/best_model.pt \
  --posterior-mean-checkpoint /path/to/real_plus_br_lora_mean/best_model.pt \
  --posterior-sampling-checkpoint /path/to/real_plus_br_lora_posterior/best_model.pt
```

The default batch size is 26 and can be changed with `--batch-size`.

The evaluator supports:

```text
--device auto
--device cpu
--device mps
--device cuda
```

## Slurm Evaluation

The A30 launcher is:

```text
downstream_evaluation/segmentation/evaluate_ucsf_pdgm_a30.slurm
```

It uses the `fdtbiotech` account and `a30_normal_q` partition.

Example:

```bash
sbatch downstream_evaluation/segmentation/evaluate_ucsf_pdgm_a30.slurm \
  --real-only-checkpoint /path/to/real_only/best_model.pt \
  --posterior-mean-checkpoint /path/to/real_plus_br_lora_mean/best_model.pt \
  --posterior-sampling-checkpoint /path/to/real_plus_br_lora_posterior/best_model.pt
```

The same launcher can be used with `--validate-only` before production
evaluation.

## Metrics

Evaluation is performed over all 202 frozen subjects.

The evaluator reports slice-level metrics across:

- all axial slices; and
- tumor-positive axial slices only.

The primary overlap metrics are:

- Dice; and
- intersection over union (IoU).

For each subject, predictions are additionally accumulated across the complete
3D volume. This provides true subject-level volumetric Dice and IoU rather
than an average of independent slice scores.

Subject-level mean slice metrics are also retained.

Empty-prediction/empty-target slices receive Dice and IoU equal to 1.

## Outputs and Provenance

New hardened evaluations are written under:

```text
outputs/downstream_segmentation/evaluations/ucsf_pdgm_external_202/
```

Run directories are non-overwriting.

Evaluation provenance records include:

- frozen manifest path and SHA-256 hash;
- checkpoint paths and SHA-256 hashes;
- checkpoint epochs and thresholds;
- UCSF-PDGM root;
- expected modality, geometry, and whole-tumor rule;
- batch size;
- Python, PyTorch, and CUDA environment;
- GPU and Slurm metadata when applicable;
- Git commit;
- evaluation summaries; and
- reference-implementation attribution.

Large evaluation outputs are not stored in Git.

Selected logs and repository-controlled summaries may be retained under:

```text
downstream_evaluation/logs/
downstream_evaluation/results/
```

Existing files identified as preliminary or pre-hardening results are
historical provenance records and should not be overwritten.

## Reproducibility Boundary

The version-controlled frozen subject manifest is the evaluation contract.

Routine evaluation should consume that manifest rather than reconstructing the
202-subject cohort from clinical metadata each time. The separate cohort
validator exists so that the derivation can be audited when the provenance
tables are available.

This separation keeps clinical metadata out of the model-facing evaluation
workflow while preserving a reproducible record of how the external cohort was
defined.

## Citation

Users of UCSF-PDGM should cite the dataset according to the instructions on
the official TCIA collection page:

https://www.cancerimagingarchive.net/collection/ucsf-pdgm/

The current TCIA collection page lists the dataset DOI as:

```text
10.7937/tcia.bdgf-8v37
```
