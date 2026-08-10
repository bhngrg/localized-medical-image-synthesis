"""Training utilities for localized medical image synthesis."""

from .losses import (
    DEFAULT_OUTSIDE_LOSS_WEIGHT,
    masked_x0_loss,
)
from .trainer import (
    build_checkpoint_payload,
    fit,
    prepare_model_input,
    save_checkpoint,
    train_one_epoch,
    validate_one_epoch,
)

from .br_lora_objectives import (
    BRLoRAObjectiveError,
    BRLoRAVariationalObjective,
    RegionalReconstructionLoss,
    br_lora_objective,
    linear_kl_warmup_multiplier,
    normalized_variational_objective,
    reconstruction_matches_baseline,
    regional_reconstruction_loss,
)

from .br_lora_step import (
    BRLoRAStepError,
    BRLoRAStepResult,
    train_br_lora_step,
)

__all__ = [
    "DEFAULT_OUTSIDE_LOSS_WEIGHT",
    "build_checkpoint_payload",
    "fit",
    "masked_x0_loss",
    "prepare_model_input",
    "save_checkpoint",
    "train_one_epoch",
    "validate_one_epoch",
    "BRLoRAObjectiveError",
    "BRLoRAVariationalObjective",
    "RegionalReconstructionLoss",
    "br_lora_objective",
    "linear_kl_warmup_multiplier",
    "normalized_variational_objective",
    "reconstruction_matches_baseline",
    "regional_reconstruction_loss",
    "BRLoRAStepError",
    "BRLoRAStepResult",
    "train_br_lora_step",
]
