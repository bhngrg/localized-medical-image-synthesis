"""Inference utilities for localized medical image synthesis."""

from .adaptation import (
    AdaptationInferenceError,
    AdaptationInferenceResult,
    LoadedAdaptation,
    PreparedAdaptationInference,
    SUPPORTED_ADAPTATION_METHODS,
    adaptation_inference,
    load_fitted_adaptation,
    prepare_adaptation_batch,
)
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
    "AdaptationInferenceError",
    "AdaptationInferenceResult",
    "BRLoRAInferenceError",
    "BRLoRAPosteriorMeanResult",
    "BRLoRAPosteriorSamplesResult",
    "ExternalEvaluationCase",
    "ExternalManifestError",
    "LoadedAdaptation",
    "LoadedBRLoRA",
    "PosteriorProducts",
    "PosteriorProductsError",
    "PreparedAdaptationInference",
    "PreparedBRLoRAInference",
    "PreparedExternalPair",
    "SUPPORTED_ADAPTATION_METHODS",
    "adaptation_inference",
    "compute_posterior_products",
    "discover_composition_candidates",
    "load_external_evaluation_manifest",
    "load_fitted_adaptation",
    "load_fitted_br_lora",
    "posterior_mean_inference",
    "posterior_sample_inference",
    "prepare_adaptation_batch",
    "prepare_br_lora_batch",
    "prepare_external_pair",
    "prepare_selected_pairs",
    "reconstruct_batch",
    "reconstruct_composite_mean",
    "reconstruct_composites",
    "select_clean_insertion_pairs",
    "synthesize_insertion_pairs",
]
