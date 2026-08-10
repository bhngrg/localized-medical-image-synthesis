"""Inference utilities for localized medical image synthesis."""

from .br_lora import (
    BRLoRAInferenceError,
    BRLoRAPosteriorMeanResult,
    BRLoRAPosteriorSamplesResult,
    LoadedBRLoRA,
    PreparedBRLoRAInference,
    load_fitted_br_lora,
    posterior_mean_inference,
    posterior_sample_inference,
    prepare_br_lora_batch,
)
from .composition import (
    discover_composition_candidates,
    select_clean_insertion_pairs,
    synthesize_insertion_pairs,
)
from .sampling import reconstruct_batch

__all__ = [
    "BRLoRAInferenceError",
    "BRLoRAPosteriorMeanResult",
    "BRLoRAPosteriorSamplesResult",
    "LoadedBRLoRA",
    "PreparedBRLoRAInference",
    "discover_composition_candidates",
    "load_fitted_br_lora",
    "posterior_mean_inference",
    "posterior_sample_inference",
    "prepare_br_lora_batch",
    "reconstruct_batch",
    "select_clean_insertion_pairs",
    "synthesize_insertion_pairs",
]
