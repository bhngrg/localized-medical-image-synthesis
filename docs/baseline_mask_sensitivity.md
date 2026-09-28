# Baseline Mask-Sensitivity Analysis

This analysis characterizes how the pretrained baseline conditional diffusion generator responds to whole-tumor mask geometry, boundary structure, topology, and controlled mask perturbations. It is a characterization of the baseline generator rather than a BR-LoRA training or downstream segmentation experiment.

The scientific cohort contains 19,941 tumor-containing BraTS 2020 H5 slices from 369 volumes, using the same whole-tumor eligibility threshold of at least 300 tumor pixels used by the baseline full-training workflow.

## Analysis overview

The analysis has three complementary parts:

1. **Observed geometric and topological characteristics.** Each eligible whole-tumor mask is characterized using geometric summaries, basic topological summaries, and persistent-homology summaries.
2. **Observed geometry versus baseline synthesis response.** Those mask characteristics are associated with boundary-response and reconstruction-error measures from the fixed baseline generator.
3. **Controlled signed-distance perturbations.** Each original mask is eroded or dilated while the source image and stochastic synthesis inputs are held fixed, enabling paired assessment of how changes to the conditioning mask affect the generated image.

The third analysis is a controlled geometric perturbation experiment. Topological changes are observed consequences of erosion or dilation; topological structure itself is not directly manipulated.

## Mask geometry and topological characteristics

For every eligible slice, the analysis records mask area, Crofton perimeter, perimeter-to-area ratio, compactness, ordinary Betti numbers, maximum Euclidean distance-transform (EDT) value, and H0/H1 persistent-homology summaries.

The basic topological quantities are:

- `beta_0`: number of connected mask components;
- `beta_1`: number of holes;
- Euler characteristic: `beta_0 - beta_1`.

The persistent-homology summaries include the number of positive finite persistence intervals and their maximum and total persistence for H0 and H1.

Across the 19,941 slices, median mask area was 1,618 pixels and median maximum EDT was 14.56 pixels. The median numbers of connected components and holes were two and one, respectively. H0 positive finite persistence was present in 98.8% of slices, whereas H1 positive finite persistence was present in 44.7%.

These descriptors are strongly interrelated. For example, mask area and maximum EDT had a Spearman correlation of 0.893, while perimeter-to-area ratio and maximum EDT had a correlation of -0.918. The geometry-response analyses should therefore be interpreted as marginal associations rather than independent effects of orthogonal mask characteristics.

Curated outputs are stored under:

`results/baseline_mask_sensitivity/mask_geometry_topology/`

## Boundary-response characterization

The baseline generator is evaluated relative to the original normalized FLAIR image as a function of signed distance from the whole-tumor boundary. Negative signed distances denote pixels inside the original mask and positive distances denote pixels outside it.

The principal boundary-response quantities include:

- signed and absolute intensity jumps across the mask boundary for the real image, baseline prediction, and hard composite;
- excess prediction/composite boundary jump relative to the real-image boundary jump;
- mean image-gradient magnitude near the boundary and the corresponding excess gradient relative to the real image;
- prediction MAE within 1-4 pixels inside the lesion boundary; and
- prediction MAE in the deeper lesion interior.

The geometry-response analysis joins all 19,941 slices exactly by `(volume, slice)`, with H5-basename and mask-pixel equality checks. Primary associations are quantified using Spearman correlation. Because multiple slices originate from the same subject/volume, uncertainty is quantified using 10,000-replicate BraTS-volume cluster-bootstrap percentile 95% confidence intervals with seed 2026. Slice-level p-values are not used. A volume-median sensitivity analysis first aggregates each quantity within volume and then recomputes the correlations across the 369 volumes.

The strongest observed associations involved basic measures of geometric scale and boundary complexity rather than the H1 persistent-homology summaries. For example, maximum EDT was associated with prediction excess boundary jump (`rho = -0.398`, 95% cluster-bootstrap CI `[-0.443, -0.352]`) and composite excess boundary gradient (`rho = 0.354`, 95% CI `[0.304, 0.403]`). Perimeter-to-area ratio was associated with prediction excess boundary jump (`rho = 0.389`, 95% CI `[0.341, 0.436]`) and composite excess boundary gradient (`rho = -0.344`, 95% CI `[-0.394, -0.292]`).

The secondary MAE analysis similarly showed stronger relationships near the lesion boundary than in the deep interior. Mask area was associated with prediction MAE within 1-4 pixels of the boundary (`rho = -0.303`, 95% CI `[-0.358, -0.247]`), whereas its association with deep-interior MAE was small and its interval included zero (`rho = -0.033`, 95% CI `[-0.078, 0.015]`). Deep-interior MAE is unavailable for 202 slices whose masks contain no pixels in the defined deep-interior region.

These are descriptive associations and should not be interpreted as causal effects of individual geometric or topological descriptors.

Curated outputs are stored under:

`results/baseline_mask_sensitivity/geometry_response_associations/`

## Controlled signed-distance perturbations

Controlled perturbations are constructed from the original binary whole-tumor mask using Euclidean distance transforms. Radius zero reproduces the original mask exactly.

For an erosion with radius `-r`, an original-mask pixel is retained only when its interior Euclidean distance is strictly greater than `r`. For a dilation with radius `+r`, exterior pixels whose Euclidean distance to the original mask is less than or equal to `r` are added.

The full perturbation audit covers integer radii from -8 through +8 for every eligible slice, producing 338,997 mask conditions. As an integrity check, reconstructed masks must exactly reproduce the perturbation catalog's recorded mask area.

Even small geometric perturbations frequently alter basic topological structure. At radius -1, 67.3% of masks changed `beta_0` or `beta_1`; at radius +1, 70.1% underwent a topological change. The topological-change fraction increased to 77.1% at radius -4 and 76.3% at radius +4. Strong erosions can destroy the mask completely: at radius -4, 59 eroded masks were empty, increasing to 2,605 at radius -8.

Empty masks are assigned `beta_0 = 0`, `beta_1 = 0`, and Euler characteristic 0 in the topological audit. Empty-mask destruction is reported separately from the general topological-change indicator.

Curated topological-audit outputs are stored under:

`results/baseline_mask_sensitivity/signed_distance/`

## Paired generator-response experiment

Generator response is evaluated at radii -4, -2, -1, 0, +1, +2, and +4. Each nonzero-radius condition is compared with the same slice's radius-zero synthesis while holding the source image, appearance vector, diffusion timestep, and diffusion noise fixed. This pairing isolates the response to the conditioning-mask perturbation from stochastic variation in the synthesis procedure.

The response catalog contains 139,587 slice-radius conditions. Generator-response measurements are available for all conditions except the 59 radius -4 erosions that produce empty masks. Radius zero is an exact identity control and therefore has zero response by construction.

Key response measures include:

- `composite_mae_global`: global mean absolute difference from the radius-zero composite;
- `composite_mae_original_boundary`: mean absolute difference in the original-mask boundary shell;
- `composite_mae_changed_region`: mean absolute difference over pixels whose mask membership changed;
- `composite_abs_response_near_change_fraction`: fraction of total absolute composite response occurring in the neighborhood of changed mask pixels.

The perturbation response is highly localized. Across the nonzero perturbation radii, the median fraction of total absolute composite response occurring near changed mask pixels ranged from approximately 0.859 to 0.917. Response magnitude also increased with stronger dilation: median global composite MAE increased from `1.39e-05` at radius +1 to `6.06e-05` at radius +4, while median changed-region MAE increased from 0.00130 to 0.00285.

For each nonzero radius, response is also stratified by whether the basic topological structure changed. The reported effect is

`median(response | topological change) - median(response | topological structure preserved)`.

Uncertainty is estimated with 10,000-replicate BraTS-volume cluster-bootstrap percentile 95% confidence intervals using seed 2026. P-values and multiplicity corrections are not used. Perturbations that changed basic topology generally showed somewhat larger responses than perturbations that preserved topology at the same radius, but this comparison is observational within each geometric perturbation level: topological change is not an independently randomized intervention.

Curated response outputs are stored under:

`results/baseline_mask_sensitivity/perturbation_topology_response/`

## Interpretation

Together, these analyses characterize three properties of the baseline conditional generator that are relevant to the later adaptation workflow.

First, synthesis behavior near the prescribed lesion boundary varies systematically with mask geometry, particularly measures related to lesion scale, depth, and boundary complexity. Second, controlled changes to the conditioning mask produce spatially localized changes in the synthesized composite, with most absolute response concentrated near the altered mask region. Third, even one- or two-pixel signed-distance perturbations frequently alter basic mask topological structure, demonstrating that geometric mask perturbation and topological change are closely coupled in this dataset.

These results characterize the sensitivity of the fixed baseline generator. They do not by themselves establish that one mask geometry is preferable, that topological changes cause synthesis differences, or that the measured image differences correspond directly to downstream segmentation performance.

## Runtime and curated artifacts

Large per-condition catalogs and intermediate generator outputs are runtime artifacts and are not intended for Git distribution. They are retained locally as reproducibility intermediates, while the repository stores compact scientific summaries and analysis tables under `results/baseline_mask_sensitivity/`.

The principal curated result groups are:

- `mask_geometry_topology/` — original-mask geometric, topological, and persistent-homology catalog and summary;
- `geometry_response_associations/` — geometry/boundary-response associations, cluster-bootstrap intervals, secondary MAE analyses, and volume-median sensitivity analyses;
- `signed_distance/` — controlled perturbation topological summaries;
- `perturbation_topology_response/` — radius-level paired response summaries and topologically stratified response analyses.

Large intermediate catalogs are written under `outputs/baseline_mask_sensitivity/` and are retained outside Git as reproducibility intermediates generated by the documented workflow.
