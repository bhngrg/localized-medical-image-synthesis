# Preliminary Downstream Segmentation Results

These results are preliminary and were obtained before the planned reproducibility-hardening pass. The current checkpoints and outputs are being preserved for collaborator review and will not be overwritten when the downstream code is later generalized and rerun.

## Internal BraTS validation

Checkpoint selection was based on tumor-positive validation Dice.

| Experiment | Training regime | Best epoch | Tumor-positive Dice | Tumor-positive IoU | All-slice Dice | All-slice IoU |
|---|---|---:|---:|---:|---:|---:|
| 1 | Real only | 20 | 0.736496 | 0.661304 | 0.852869 | 0.820786 |
| 2 | Real + BR-LoRA posterior mean | 20 | 0.732357 | 0.657880 | 0.862960 | 0.831183 |
| 3 | Real + BR-LoRA posterior sampling | 20 | 0.731261 | 0.655442 | 0.855692 | 0.823342 |

## Selected checkpoints

- Experiment 1:
  `outputs/downstream_segmentation/real_only_seed42_a30_normal_q/best_model.pt`

- Experiment 2:
  `outputs/downstream_segmentation/real_plus_br_lora_seed42_a30_normal_q_rerun/best_model.pt`

- Experiment 3:
  `outputs/downstream_segmentation/real_plus_br_lora_posterior_seed42_a30_normal_q/best_model.pt`

## Run provenance

- Experiment 1: Slurm job `552448`, completed 20/20 epochs in `02:28:34`.
- Experiment 2: selected rerun Slurm job `553847`, completed 20/20 epochs in `03:09:20`.
  - Original Experiment 2 job `552475` reached 18/20 completed epochs and was terminated only because of the original 12-hour walltime.
  - The original timed-out outputs remain preserved separately.
- Experiment 3: Slurm job `552535`, completed 20/20 epochs in `03:10:30`.
  - Each synthetic case used 20 distinct posterior realizations across the 20 training epochs.

## Interpretation

The three internal-validation results are very close. These values should therefore be treated primarily as checkpoint-selection and optimization diagnostics. The planned external UCSF-PDGM evaluation will provide the more important comparison of downstream generalization across the three training regimes.
