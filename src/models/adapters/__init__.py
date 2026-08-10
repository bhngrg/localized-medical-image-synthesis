"""
Adapter infrastructure for deterministic LoRA and Bayesian Regional LoRA.
"""

from .base import (
    AdaptationError,
    AdaptationReport,
    count_parameters,
    freeze_module,
    make_adaptation_report,
    trainable_parameter_names,
    unfreeze_module,
)

from .lora import (
    LoRAConv2d,
    LoRAError,
    deterministic_lora_parameter_count,
    inject_lora,
    iter_lora_modules,
)

from .selection import (
    ModuleSelectionError,
    replace_named_module,
    resolve_parent_module,
    select_named_modules,
)

from .variational import (
    DiagonalGaussianParameter,
    VariationalParameterError,
)

from .variational_lora import (
    VariationalLoRAConv2d,
    VariationalLoRAError,
    convert_lora_to_variational,
    disable_variational_lora,
    disable_variational_sampling,
    enable_variational_lora,
    enable_variational_sampling,
    iter_variational_lora_modules,
    variational_lora_kl_divergence,
    variational_lora_parameter_count,
)


__all__ = [
    "AdaptationError",
    "AdaptationReport",
    "DiagonalGaussianParameter",
    "LoRAConv2d",
    "LoRAError",
    "ModuleSelectionError",
    "VariationalLoRAConv2d",
    "VariationalLoRAError",
    "VariationalParameterError",
    "convert_lora_to_variational",
    "count_parameters",
    "deterministic_lora_parameter_count",
    "disable_variational_lora",
    "disable_variational_sampling",
    "enable_variational_lora",
    "enable_variational_sampling",
    "freeze_module",
    "inject_lora",
    "iter_lora_modules",
    "iter_variational_lora_modules",
    "make_adaptation_report",
    "replace_named_module",
    "resolve_parent_module",
    "select_named_modules",
    "trainable_parameter_names",
    "unfreeze_module",
    "variational_lora_kl_divergence",
    "variational_lora_parameter_count",
]