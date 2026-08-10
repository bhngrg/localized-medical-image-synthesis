"""
Deterministic target-module selection and replacement utilities.

These helpers are used by LoRA and BR-LoRA to select and replace layers in the
validated AppearanceX0UNet backbone without changing unrelated modules.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from torch import nn

from .base import AdaptationError


class ModuleSelectionError(AdaptationError):
    """Raised when target-module selection or replacement is invalid."""


def select_named_modules(
    model: nn.Module,
    *,
    exact_names: Iterable[str] = (),
    suffixes: Iterable[str] = (),
    regex_patterns: Iterable[str] = (),
    module_types: tuple[
        type[nn.Module],
        ...,
    ] | None = None,
    exclude_patterns: Iterable[str] = (),
    require_match: bool = True,
) -> tuple[
    tuple[
        str,
        nn.Module,
    ],
    ...,
]:
    """
    Select modules deterministically in ``named_modules()`` order.

    A module is selected when it satisfies at least one supplied name selector
    and passes the optional type and exclusion filters.

    When no name selector is supplied, all named modules are considered before
    type and exclusion filtering.
    """

    exact_name_set = frozenset(
        exact_names
    )

    suffix_tuple = tuple(
        suffixes
    )

    compiled_patterns = tuple(
        re.compile(
            pattern
        )
        for pattern in regex_patterns
    )

    compiled_exclusions = tuple(
        re.compile(
            pattern
        )
        for pattern in exclude_patterns
    )

    has_name_selector = bool(
        exact_name_set
        or suffix_tuple
        or compiled_patterns
    )

    selected: list[
        tuple[
            str,
            nn.Module,
        ]
    ] = []

    for name, module in model.named_modules():

        if not name:
            continue

        name_matches = (
            not has_name_selector

            or name in exact_name_set

            or any(
                name.endswith(
                    suffix
                )
                for suffix in suffix_tuple
            )

            or any(
                pattern.search(
                    name
                )
                for pattern in compiled_patterns
            )
        )

        if not name_matches:
            continue

        if any(
            pattern.search(
                name
            )
            for pattern in compiled_exclusions
        ):
            continue

        if (
            module_types is not None
            and not isinstance(
                module,
                module_types,
            )
        ):
            continue

        selected.append(
            (
                name,
                module,
            )
        )

    if (
        require_match
        and not selected
    ):
        raise ModuleSelectionError(
            "No target modules matched the supplied selectors, "
            "module types, and exclusions."
        )

    return tuple(
        selected
    )


def resolve_parent_module(
    model: nn.Module,
    qualified_name: str,
) -> tuple[
    nn.Module,
    str,
]:
    """
    Resolve the parent module and final path component of a nested module.

    This supports ordinary attributes as well as integer indices inside
    ``nn.Sequential`` and ``nn.ModuleList`` containers.
    """

    if not qualified_name:
        raise ModuleSelectionError(
            "The root module cannot be replaced by an empty name."
        )

    parts = qualified_name.split(
        "."
    )

    parent = model

    for part in parts[
        :-1
    ]:

        if (
            part.isdigit()
            and isinstance(
                parent,
                (
                    nn.Sequential,
                    nn.ModuleList,
                ),
            )
        ):
            parent = parent[
                int(
                    part
                )
            ]

        else:
            if not hasattr(
                parent,
                part,
            ):
                raise ModuleSelectionError(
                    f"Could not resolve module path "
                    f"{qualified_name!r}; "
                    f"missing component {part!r}."
                )

            child = getattr(
                parent,
                part,
            )

            if not isinstance(
                child,
                nn.Module,
            ):
                raise ModuleSelectionError(
                    f"Resolved component {part!r} "
                    f"in {qualified_name!r}, "
                    "but it is not an nn.Module."
                )

            parent = child

    return (
        parent,
        parts[
            -1
        ],
    )


def replace_named_module(
    model: nn.Module,
    qualified_name: str,
    replacement: nn.Module,
) -> None:
    """Replace one nested module by its qualified name."""

    parent, final_component = (
        resolve_parent_module(
            model,
            qualified_name,
        )
    )

    if (
        final_component.isdigit()
        and isinstance(
            parent,
            (
                nn.Sequential,
                nn.ModuleList,
            ),
        )
    ):
        index = int(
            final_component
        )

        if index >= len(
            parent
        ):
            raise ModuleSelectionError(
                f"Module index {index} is out of range "
                f"for {qualified_name!r}."
            )

        parent[
            index
        ] = replacement

        return

    if not hasattr(
        parent,
        final_component,
    ):
        raise ModuleSelectionError(
            f"Could not replace module {qualified_name!r}; "
            f"the final component {final_component!r} does not exist."
        )

    current = getattr(
        parent,
        final_component,
    )

    if not isinstance(
        current,
        nn.Module,
    ):
        raise ModuleSelectionError(
            f"Cannot replace {qualified_name!r} because the current "
            "value is not an nn.Module."
        )

    setattr(
        parent,
        final_component,
        replacement,
    )


__all__ = [
    "ModuleSelectionError",
    "replace_named_module",
    "resolve_parent_module",
    "select_named_modules",
]