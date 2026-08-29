# BraTS 2020 Dataset

This repository operates directly from the official **BraTS 2020 Training** and
**BraTS 2020 Validation** NIfTI releases.

Rather than relying on the historical preprocessed H5 dataset, the repository
provides a fully reproducible data-preparation pipeline that reconstructs the
historical H5 files directly from the raw BraTS NIfTI volumes.

The raw datasets are **not distributed** with this repository.

---

# Dataset Download

The BraTS 2020 datasets are publicly available from Kaggle:

https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation

A Kaggle account is required to download the data.

---

# Required Datasets

The repository expects two independent datasets.

## Training Dataset

```text
MICCAI_BraTS2020_TrainingData/
├── BraTS20_Training_001/
├── BraTS20_Training_002/
├── ...
└── BraTS20_Training_369/
```

The training release contains

- four MRI modalities
- expert segmentation masks

and is used to construct the historical H5 dataset.

---

## Validation Dataset

```text
MICCAI_BraTS2020_ValidationData/
├── BraTS20_Validation_001/
├── BraTS20_Validation_002/
├── ...
└── BraTS20_Validation_125/
```

The validation release contains

- four MRI modalities

but **does not include segmentation masks**.

---

# Dataset Registration

Before any preprocessing is performed, both datasets are registered exactly
once.

Training dataset:

```text
scripts/register_dataset.py
```

Validation dataset:

```text
scripts/register_validation_dataset.py
```

These scripts

- validate the raw NIfTI datasets,
- verify directory structure,
- verify subject completeness,
- verify image geometry,
- verify affine consistency,
- verify segmentation labels (training only),
- record observed MRI and segmentation dtypes, and
- generate machine-readable dataset specifications.

The resulting files are

```text
dataset.yaml
validation_dataset.yaml
```

Downstream workflows consume these specifications rather than rescanning the raw
datasets.

---

# Machine-Specific Paths

Machine-specific paths can be supplied explicitly on the command line or
stored in a machine-specific YAML file. An example is provided at

```text
data/folders.example.yaml
```

Copy it to

```text
data/folders.yaml
```

and populate the paths needed on the current machine. The real
`data/folders.yaml` is excluded from version control so local filesystem paths
are not committed to the repository.

Path resolution follows this precedence:

```text
explicit CLI argument
        │
        ▼
saved data/folders.yaml value
        │
        ▼
interactive selector
```

The interactive fallback applies only to scripts that provide an interactive
selector. Model training and Falcon BR-LoRA batch acceptance are
noninteractive: required paths must be supplied by CLI or available in the
folders configuration.

Paths selected explicitly by CLI or through an interactive selector are saved
to the folders configuration for reuse.

The configuration supports the following keys:

```yaml
data_root: null
yaml_dataset_path: null

validation_data_root: null
yaml_validation_dataset_path: null

h5_root: null
manifest_path: null

br_lora_library_root: null
```

Explicit CLI arguments always override saved values.

---

# Historical H5 Reconstruction

The historical H5 dataset is reconstructed from the registered training dataset
using

```text
scripts/build_h5_dataset.py
```

This produces

```text
57,195 H5 files
```

with the historical structure

```text
image
mask
```

Each H5 file corresponds to a single axial slice.

The reconstructed dataset has been verified against the historical release and
reproduces

- image values to floating-point precision,
- mask values exactly,
- historical slice ordering,
- historical channel ordering.

---

# Slice Manifest

After H5 reconstruction, slice-level metadata are generated using

```text
scripts/create_dataset_manifest.py
```

which produces

```text
manifest.csv
```

The manifest contains one row per H5 slice and records

- H5 filename
- subject index
- slice index
- tumor/non-tumor target
- pixel counts for each segmentation label

This manifest replaces repeated H5 scanning during downstream experiments.

---

# Verification

The repository includes verification utilities that compare the reconstructed
dataset against the historical preprocessing.

Current verification includes

- raw NIfTI registration
- H5 reconstruction
- H5 image verification
- H5 mask verification
- slice-level manifest verification

These checks ensure that future refactoring preserves identical preprocessing
behavior.

---

# Dataset Workflow

Training dataset:

```text
Raw BraTS Training NIfTI
        │
        ▼
register_dataset.py
        │
        ▼
dataset.yaml
        │
        ▼
build_h5_dataset.py
        │
        ▼
57,195 reconstructed H5 files
        │
        ▼
create_dataset_manifest.py
        │
        ▼
manifest.csv
```

Validation dataset:

```text
Raw BraTS Validation NIfTI
        │
        ▼
register_validation_dataset.py
        │
        ▼
validation_dataset.yaml
```

---

# Repository Philosophy

Expensive dataset validation is performed exactly once.

All subsequent workflows consume the generated

- dataset specifications,
- reconstructed H5 files, and
- manifest,

rather than repeatedly scanning the raw datasets.

This improves reproducibility, simplifies downstream code, and clearly separates
dataset registration from model development.

---

# Citation

If you use this repository in academic work, please cite

- the BraTS challenge and dataset, and
- this repository (once publicly released).