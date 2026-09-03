# UCSF-PDGM External Downstream Segmentation Evaluation

## Evaluation setup

The downstream experiment evaluates whether augmenting the real BraTS training
set with BR-LoRA synthetic images improves tumor segmentation on an independent
external cohort. Three segmentation models were trained under the same
architecture and optimization protocol, differing only in the training data
available to each model:

- **Real only:** trained using the real BraTS downstream-training set.
- **Real + BR-LoRA posterior mean:** trained using the same real data augmented
  with synthetic images generated from the posterior-mean BR-LoRA model.
- **Real + BR-LoRA posterior sampling:** trained using the same real data
  augmented with fixed, reproducibly seeded posterior draws from the accepted
  BR-LoRA synthetic library.

The BraTS downstream split was defined at the subject level, with 332 subjects
assigned to downstream training and 37 subjects assigned to downstream
validation. Within the 332-subject training partition, every exact donor slice
used to construct the frozen 10,000-case BR-LoRA synthetic library was excluded
from the real downstream-training pool before segmentation training. This
produced 41,460 real training slices, with zero residual overlap with the
10,000 frozen donor slices. The augmented regimes therefore used these same
41,460 real slices plus 10,000 BR-LoRA synthetic cases.

This separation is intentionally **slice-level with respect to the synthetic
donor pool**, not subject-level: a BraTS training subject could still contribute
other, non-donor slices to downstream segmentation training. Thus, the design
prevents an exact synthetic donor slice from also appearing as a real
segmentation-training example without claiming that all subjects contributing
to synthetic generation were excluded from downstream training.

All three models were trained with seed 42 and were evaluated on the same
frozen **202-subject UCSF-PDGM external cohort** after subjects overlapping with
the BraTS cohort were excluded. No UCSF-PDGM cases were used for downstream
model training or model selection.

The external evaluation therefore tests generalization to a separate dataset
rather than performance on the BraTS validation split used during downstream
model development.

## External volumetric Dice

The external evaluation uses the FLAIR modality. Reference whole-tumor masks
are defined as all voxels with segmentation label greater than zero. Model
outputs are converted to binary tumor predictions using the frozen threshold
of 0.5.

For each UCSF-PDGM subject, the slice-level segmentation predictions are
reassembled into a three-dimensional predicted tumor mask and compared with
the corresponding three-dimensional reference tumor mask.

Volumetric Dice is

$$
\mathrm{Dice}
=
\frac{2\,|P \cap G|}
{|P| + |G|},
$$

where $P$ is the set of voxels predicted as tumor and $G$ is the set of
reference tumor voxels for the subject.

A Dice value of **1** indicates perfect spatial overlap, whereas **0** indicates
no overlap. Because the calculation is performed after reconstructing the
subject-level 3D volume, this metric is not an average of individual
slice-level Dice scores. It measures how well the complete predicted tumor
volume agrees with the complete reference tumor volume for each external
subject.

The table reports the **arithmetic mean ± sample standard deviation across the
202 subjects**. The 95% confidence intervals and paired differences were
estimated using **10,000 nonparametric subject-level bootstrap resamples with
replacement** using percentile intervals and seed 2026. Resampling was paired
by subject ID, and the same bootstrap subject indices were used for all three
training regimes and comparisons. Slice-level resampling was not used.

## Results

| Training regime                   | External volumetric Dice (mean ± SD)   | 95% CI         | Paired Δ vs. real-only   |
|:----------------------------------|:---------------------------------------|:---------------|:-------------------------|
| Real only                         | 0.664 ± 0.208                          | [0.635, 0.693] | —                        |
| Real + BR-LoRA posterior mean     | 0.812 ± 0.167                          | [0.788, 0.834] | +0.147 [0.129, 0.166]    |
| Real + BR-LoRA posterior sampling | 0.711 ± 0.201                          | [0.683, 0.738] | +0.046 [0.036, 0.057]    |

Both BR-LoRA-augmented training regimes improved external volumetric Dice
relative to real-only training under this frozen evaluation protocol. The
posterior-mean augmentation regime produced the largest improvement
($\Delta=+0.147$), while posterior-sampled augmentation also improved over
real-only training ($\Delta=+0.046$).

These results characterize the downstream performance of the specific
synthetic-data construction and training protocols evaluated here. They should
not be interpreted as evidence that posterior-mean synthesis is intrinsically
superior to posterior sampling in general.
