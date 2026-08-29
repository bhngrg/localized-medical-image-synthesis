"""
Machine-specific path configuration helpers.

Repository scripts may obtain filesystem paths from three sources, in
descending order of precedence:

1. an explicit command-line argument;
2. a value stored in the machine-specific folders YAML file;
3. an optional interactive selector supplied by the calling script.

Paths resolved from explicit command-line arguments or interactive selectors
are written back to the folders configuration when requested by the caller.
The configuration file is intended to contain machine-specific paths and
should not be committed to version control.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Optional

import yaml


PathSelector = Callable[[], Path]


def load_folders_config(
    folders_file: Optional[Path],
) -> dict[str, str]:
    """
    Load the machine-specific folders configuration.

    A missing configuration file is treated as an empty configuration.
    An empty YAML file is also treated as an empty configuration.
    """
    if folders_file is None:
        return {}

    folders_file = Path(folders_file)

    if not folders_file.exists():
        return {}

    if not folders_file.is_file():
        raise ValueError(
            "Folders configuration path is not a file:\n\n"
            f"{folders_file}"
        )

    try:
        with folders_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            loaded = yaml.safe_load(file)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(
            "Could not read folders configuration:\n\n"
            f"{folders_file}"
        ) from exc

    if loaded is None:
        return {}

    if not isinstance(loaded, Mapping):
        raise ValueError(
            "Folders configuration must contain a YAML mapping:\n\n"
            f"{folders_file}"
        )

    config: dict[str, str] = {}

    for key, value in loaded.items():
        if value is None:
            continue

        if not isinstance(key, str):
            raise ValueError(
                "Folders configuration keys must be strings."
            )

        if not isinstance(value, str):
            raise ValueError(
                "Folders configuration values must be paths encoded "
                f"as strings. Invalid key: {key}"
            )

        config[key] = value

    return config


def save_folders_config(
    folders_file: Optional[Path],
    config: Mapping[str, str],
) -> None:
    """Write the machine-specific folders configuration."""
    if folders_file is None:
        return

    folders_file = Path(folders_file)

    try:
        folders_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with folders_file.open(
            "w",
            encoding="utf-8",
        ) as file:
            yaml.safe_dump(
                dict(config),
                file,
                sort_keys=True,
            )
    except OSError as exc:
        raise ValueError(
            "Could not write folders configuration:\n\n"
            f"{folders_file}"
        ) from exc


def resolve_path(
    *,
    key: str,
    cli_value: Optional[Path],
    config: dict[str, str],
    selector: Optional[PathSelector] = None,
) -> Path:
    """
    Resolve one path using CLI > saved configuration > selector precedence.

    When a CLI value or selector supplies the path, the in-memory
    configuration is updated so the caller may persist it.
    """
    if cli_value is not None:
        path = Path(cli_value)
        config[key] = str(path)
        return path

    configured_value = config.get(key)

    if configured_value:
        return Path(configured_value)

    if selector is not None:
        path = Path(selector())
        config[key] = str(path)
        return path

    raise ValueError(
        "Required path was not provided and is not present in the "
        f"folders configuration: {key}"
    )
