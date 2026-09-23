"""
Adapter infrastructure for deterministic adaptation and Bayesian Regional LoRA.
"""

from .base import (
    AdaptationError,
    AdaptationReport,
    configure_bitfit,
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

from .dora import (
    DoRAConv2d,
    DoRAError,
    configure_dora,
    deterministic_dora_parameter_count,
    inject_dora,
    iter_dora_modules,
)

from .lokr import (
    LoKrConv2d,
    LoKrError,
    balanced_factor_pair,
    configure_lokr,
    deterministic_lokr_parameter_count,
    inject_lokr,
    iter_lokr_modules,
)

from .regional_lora import (
    configure_regional_lora,
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
    "DoRAConv2d",
    "DoRAError",
    "LoRAConv2d",
    "LoRAError",
    "LoKrConv2d",
    "LoKrError",
    "ModuleSelectionError",
    "VariationalLoRAConv2d",
    "VariationalLoRAError",
    "VariationalParameterError",
    "balanced_factor_pair",
    "configure_bitfit",
    "configure_dora",
    "configure_lokr",
    "configure_regional_lora",
    "convert_lora_to_variational",
    "count_parameters",
    "deterministic_dora_parameter_count",
    "deterministic_lora_parameter_count",
    "deterministic_lokr_parameter_count",
    "disable_variational_lora",
    "disable_variational_sampling",
    "enable_variational_lora",
    "enable_variational_sampling",
    "freeze_module",
    "inject_dora",
    "inject_lora",
    "inject_lokr",
    "iter_dora_modules",
    "iter_lora_modules",
    "iter_lokr_modules",
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