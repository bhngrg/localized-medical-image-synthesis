# UCSF-PDGM External Downstream Segmentation Evaluation

## Evaluation setup

The downstream experiment evaluates whether augmenting the real BraTS training
set with BR-LoRA synthetic images improves tumor segmentation on an independent
external cohort. The primary analysis compares three segmentation models
trained under the same architecture and optimization protocol:

- **Real only:** trained using the real BraTS downstream-training set.
- **Real + BR-LoRA posterior mean (feathered):** trained using the same real
  data augmented with posterior-mean BR-LoRA synthetic images after the
  configured deterministic inner-only feathering step.
- **Real + BR-LoRA posterior sampling (feathered):** trained using the same real
  data augmented with fixed, reproducibly seeded posterior draws from the
  accepted BR-LoRA synthetic library after the same feathering step.

Earlier **non-feathered posterior-mean** and **non-feathered posterior-sampling**
models are also reported as development and provenance comparators. They use
the corresponding BR-LoRA synthetic images without the downstream feathering
step and are not part of the primary three-model analysis.

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

All reported models were trained with seed 42 and were evaluated on the same
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
\mathrm{Dice} = \frac{2 \lvert P \cap G \rvert}{\lvert P \rvert + \lvert G \rvert},
$$

where $P$ is the set of voxels predicted as tumor and $G$ is the set of
reference tumor voxels for the subject.

A Dice value of **1** indicates perfect spatial overlap, whereas **0** indicates
no overlap. Because the calculation is performed after reconstructing the
subject-level 3D volume, this metric is not an average of individual
slice-level Dice scores. It measures how well the complete predicted tumor
volume agrees with the complete reference tumor volume for each external
subject.

The table reports the **arithmetic mean ± bootstrap standard error (SE) across
the 202 subjects**. The bootstrap SE was calculated as the sample standard
deviation of the 10,000 bootstrap estimates of the mean volumetric Dice. The
95% confidence intervals and paired differences were estimated using the same
**10,000 nonparametric subject-level bootstrap resamples with replacement**,
using percentile intervals and seed 2026. Resampling was paired by subject ID,
and the same bootstrap subject indices were used for all reported conditions
and comparisons. Slice-level resampling was not used.

## Results

| Training regime                   | Composition   | External volumetric Dice (mean ± bootstrap SE) | 95% CI         | Paired Δ vs. real-only |
|:----------------------------------|:--------------|:------------------------------------------------|:---------------|:-----------------------|
| Real only                         | —             | 0.664 ± 0.015                                   | [0.635, 0.693] | —                      |
| Real + BR-LoRA posterior mean     | Non-feathered | 0.812 ± 0.012                                   | [0.788, 0.834] | +0.147 [0.129, 0.166]  |
| Real + BR-LoRA posterior mean     | Feathered     | 0.772 ± 0.012                                   | [0.747, 0.795] | +0.107 [0.092, 0.123]  |
| Real + BR-LoRA posterior sampling | Non-feathered | 0.711 ± 0.014                                   | [0.683, 0.738] | +0.046 [0.036, 0.057]  |
| Real + BR-LoRA posterior sampling | Feathered     | 0.805 ± 0.011                                   | [0.783, 0.826] | +0.141 [0.123, 0.158]  |

All four BR-LoRA-augmented conditions had higher mean external volumetric Dice
than real-only training under the corresponding frozen evaluation protocols.
The non-feathered posterior-mean condition had a paired mean difference of
$\Delta=+0.147$ relative to real-only training, while the feathered
posterior-mean condition had $\Delta=+0.107$. The non-feathered
posterior-sampling condition had $\Delta=+0.046$, while the feathered
posterior-sampling condition had $\Delta=+0.141$.

The primary downstream analysis uses the real-only condition together with the
two feathered BR-LoRA conditions. The non-feathered augmented conditions are
retained here as development and provenance comparators. Accordingly, direct
feathered-versus-non-feathered differences are presented descriptively rather
than as additional primary inferential comparisons.

These results characterize the downstream performance of the specific
synthetic-data construction and training protocols evaluated here and should
not be interpreted as establishing the general superiority of one BR-LoRA
posterior synthesis strategy over another.
