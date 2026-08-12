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
from .br_lora_external_pairs import (
    PreparedExternalPair,
    prepare_external_pair,
)
from .br_lora_pairs import (
    prepare_selected_pairs,
)
from .composition import (
    discover_composition_candidates,
    select_clean_insertion_pairs,
    synthesize_insertion_pairs,
)
from .external_manifest import (
    ExternalEvaluationCase,
    ExternalManifestError,
    load_external_evaluation_manifest,
)
from .posterior_products import (
    PosteriorProducts,
    PosteriorProductsError,
    compute_posterior_products,
    reconstruct_composite_mean,
    reconstruct_composites,
)
from .sampling import (
    reconstruct_batch,
)


__all__ = [
    "BRLoRAInferenceError",
    "BRLoRAPosteriorMeanResult",
    "BRLoRAPosteriorSamplesResult",
    "ExternalEvaluationCase",
    "ExternalManifestError",
    "LoadedBRLoRA",
    "PosteriorProducts",
    "PosteriorProductsError",
    "PreparedBRLoRAInference",
    "PreparedExternalPair",
    "compute_posterior_products",
    "discover_composition_candidates",
    "load_external_evaluation_manifest",
    "load_fitted_br_lora",
    "posterior_mean_inference",
    "posterior_sample_inference",
    "prepare_br_lora_batch",
    "prepare_external_pair",
    "prepare_selected_pairs",
    "reconstruct_batch",
    "reconstruct_composite_mean",
    "reconstruct_composites",
    "select_clean_insertion_pairs",
    "synthesize_insertion_pairs",
]
