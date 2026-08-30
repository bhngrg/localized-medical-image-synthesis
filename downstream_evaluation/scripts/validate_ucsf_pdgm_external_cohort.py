#!/usr/bin/env python3

"""
Validate the frozen UCSF-PDGM external-validation cohort.

The validator uses the cohort-provenance CSVs produced during preparation of
the UCSF-PDGM external dataset and verifies that they reproduce the frozen
202-subject repository manifest exactly.

Machine-specific paths follow the repository convention:

    CLI argument > data/folders.yaml > tracked repository default, when one
    exists.

The UCSF-PDGM metadata root has no repository default because the external
dataset is not distributed with this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    load_folders_config,
    resolve_path,
    save_folders_config,
)


DEFAULT_FROZEN_MANIFEST = (
    PROJECT_ROOT
    / "downstream_evaluation"
    / "manifests"
    / "ucsf_pdgm_external_202_subjects.csv"
)

BRATS_EXCLUSION_FILENAME = (
    "brats21_segmentation_subjects_to_exclude.csv"
)
INDEPENDENT_FILENAME = "ucsf_pdgm_independent_subjects.csv"
BASELINE_FILENAME = "ucsf_pdgm_independent_baseline_subjects.csv"

EXPECTED_TOTAL_REPRESENTED_SUBJECTS = 501
EXPECTED_BRATS21_OVERLAP = 298
EXPECTED_BRATS21_TRAINING = 262
EXPECTED_BRATS21_VALIDATION = 36
EXPECTED_INDEPENDENT_SUBJECTS = 203
EXPECTED_FOLLOW_UP_SUBJECTS = 1
EXPECTED_BASELINE_SUBJECTS = 202
EXPECTED_FOLLOW_UP_ID = "UCSF-PDGM-0391_FU016d"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the frozen 202-subject UCSF-PDGM "
            "external-validation cohort."
        )
    )

    parser.add_argument(
        "--folders-file",
        type=Path,
        default=Path("data/folders.yaml"),
        help="Machine-specific path configuration YAML.",
    )

    parser.add_argument(
        "--ucsf-pdgm-metadata-root",
        type=Path,
        default=None,
        help=(
            "Directory containing the UCSF-PDGM cohort-provenance "
            "CSVs. Overrides ucsf_pdgm_metadata_root in "
            "--folders-file."
        ),
    )

    parser.add_argument(
        "--ucsf-pdgm-manifest",
        type=Path,
        default=None,
        help=(
            "Frozen UCSF-PDGM external subject manifest. CLI "
            "overrides ucsf_pdgm_manifest in --folders-file; "
            "otherwise the tracked repository manifest is used."
        ),
    )

    return parser.parse_args()


def resolve_optional_repo_path(
    *,
    key: str,
    cli_value: Path | None,
    folders_config: dict[str, str],
    default: Path,
) -> Path:
    """
    Resolve CLI > folders YAML > tracked repository default.

    CLI overrides are persisted to the machine-specific folders mapping.
    """
    if cli_value is not None:
        path = Path(cli_value)
        folders_config[key] = str(path)
        return path

    configured = folders_config.get(key)

    if configured:
        return Path(configured)

    return default


def resolve_paths(
    args: argparse.Namespace,
) -> dict[str, Path]:
    folders_config = load_folders_config(
        args.folders_file
    )

    metadata_root = resolve_path(
        key="ucsf_pdgm_metadata_root",
        cli_value=args.ucsf_pdgm_metadata_root,
        config=folders_config,
        selector=None,
    )

    manifest = resolve_optional_repo_path(
        key="ucsf_pdgm_manifest",
        cli_value=args.ucsf_pdgm_manifest,
        folders_config=folders_config,
        default=DEFAULT_FROZEN_MANIFEST,
    )

    save_folders_config(
        args.folders_file,
        folders_config,
    )

    return {
        "metadata_root": Path(metadata_root),
        "manifest": Path(manifest),
    }


def require_columns(
    frame: pd.DataFrame,
    path: Path,
    columns: list[str],
) -> None:
    missing = sorted(
        set(columns) - set(frame.columns)
    )

    if missing:
        raise RuntimeError(
            f"{path} is missing required columns: {missing}"
        )


def read_csv(
    path: Path,
    required_columns: list[str],
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required CSV not found:\n{path}"
        )

    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
    )

    require_columns(
        frame,
        path,
        required_columns,
    )

    return frame


def clean_metadata(
    frame: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    frame = frame.copy()

    for column in (
        "ID",
        "BraTS21 ID",
        "BraTS21 Segmentation Cohort",
    ):
        frame[column] = frame[column].str.strip()

    if frame["ID"].eq("").any():
        raise RuntimeError(
            f"Metadata contains an empty subject ID:\n{path}"
        )

    if frame["ID"].duplicated().any():
        duplicate_ids = sorted(
            frame.loc[
                frame["ID"].duplicated(keep=False),
                "ID",
            ].unique()
        )

        raise RuntimeError(
            f"Duplicate subject IDs in {path}:\n"
            f"{duplicate_ids}"
        )

    return frame


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)

    metadata_root = (
        paths["metadata_root"]
        .expanduser()
        .resolve()
    )

    manifest_path = (
        paths["manifest"]
        .expanduser()
        .resolve()
    )

    if not metadata_root.is_dir():
        raise NotADirectoryError(
            "UCSF-PDGM metadata root not found or is not "
            f"a directory:\n{metadata_root}"
        )

    exclusion_path = (
        metadata_root
        / BRATS_EXCLUSION_FILENAME
    )
    independent_path = (
        metadata_root
        / INDEPENDENT_FILENAME
    )
    baseline_path = (
        metadata_root
        / BASELINE_FILENAME
    )

    metadata_columns = [
        "ID",
        "BraTS21 ID",
        "BraTS21 Segmentation Cohort",
    ]

    exclusion = clean_metadata(
        read_csv(
            exclusion_path,
            metadata_columns,
        ),
        exclusion_path,
    )

    independent = clean_metadata(
        read_csv(
            independent_path,
            metadata_columns,
        ),
        independent_path,
    )

    baseline = clean_metadata(
        read_csv(
            baseline_path,
            metadata_columns,
        ),
        baseline_path,
    )

    frozen = read_csv(
        manifest_path,
        [
            "subject_id",
            "brats21_segmentation_overlap",
            "follow_up_exam",
        ],
    )

    frozen["subject_id"] = (
        frozen["subject_id"].str.strip()
    )

    if frozen["subject_id"].eq("").any():
        raise RuntimeError(
            "Frozen manifest contains an empty subject ID."
        )

    if frozen["subject_id"].duplicated().any():
        duplicate_ids = sorted(
            frozen.loc[
                frozen["subject_id"].duplicated(keep=False),
                "subject_id",
            ].unique()
        )

        raise RuntimeError(
            "Frozen manifest contains duplicate subject IDs: "
            f"{duplicate_ids}"
        )

    exclusion_ids = set(exclusion["ID"])
    independent_ids = set(independent["ID"])
    baseline_ids = set(baseline["ID"])
    frozen_ids = set(frozen["subject_id"])

    source_overlap = (
        exclusion_ids & independent_ids
    )

    if source_overlap:
        raise RuntimeError(
            "BraTS21-excluded and independent subject lists "
            "overlap:\n"
            f"{sorted(source_overlap)}"
        )

    exclusion_labels = set(
        exclusion["BraTS21 Segmentation Cohort"]
    )

    if exclusion_labels != {
        "Training",
        "Validation",
    }:
        raise RuntimeError(
            "Unexpected BraTS21 Segmentation Cohort labels "
            "in exclusion metadata: "
            f"{sorted(exclusion_labels)}"
        )

    if exclusion["BraTS21 ID"].eq("").any():
        raise RuntimeError(
            "A BraTS21 segmentation-overlap subject has "
            "an empty BraTS21 ID."
        )

    if (
        independent["BraTS21 Segmentation Cohort"]
        .ne("")
        .any()
    ):
        raise RuntimeError(
            "Independent-subject metadata contains a "
            "non-empty BraTS21 Segmentation Cohort."
        )

    if independent["BraTS21 ID"].ne("").any():
        raise RuntimeError(
            "Independent-subject metadata contains a "
            "non-empty BraTS21 ID."
        )

    if (
        baseline["BraTS21 Segmentation Cohort"]
        .ne("")
        .any()
    ):
        raise RuntimeError(
            "Baseline metadata contains a non-empty "
            "BraTS21 Segmentation Cohort."
        )

    if baseline["BraTS21 ID"].ne("").any():
        raise RuntimeError(
            "Baseline metadata contains a non-empty "
            "BraTS21 ID."
        )

    if not baseline_ids.issubset(
        independent_ids
    ):
        raise RuntimeError(
            "Baseline subjects are not a subset of the "
            "independent-subject cohort."
        )

    removed_from_independent = sorted(
        independent_ids - baseline_ids
    )

    follow_up_ids = sorted(
        subject_id
        for subject_id in independent_ids
        if "_FU" in subject_id
    )

    if (
        removed_from_independent
        != [EXPECTED_FOLLOW_UP_ID]
    ):
        raise RuntimeError(
            "Unexpected subjects removed between independent "
            "and baseline cohorts: "
            f"{removed_from_independent}"
        )

    if follow_up_ids != [
        EXPECTED_FOLLOW_UP_ID
    ]:
        raise RuntimeError(
            "Unexpected follow-up IDs in the independent cohort: "
            f"{follow_up_ids}"
        )

    training_count = int(
        (
            exclusion["BraTS21 Segmentation Cohort"]
            == "Training"
        ).sum()
    )

    validation_count = int(
        (
            exclusion["BraTS21 Segmentation Cohort"]
            == "Validation"
        ).sum()
    )

    checks = {
        "total_represented_subjects": (
            len(exclusion) + len(independent)
        ),
        "brats21_segmentation_overlap": len(
            exclusion
        ),
        "brats21_training": training_count,
        "brats21_validation": validation_count,
        "independent_subjects": len(independent),
        "follow_up_subjects": len(
            removed_from_independent
        ),
        "baseline_independent_subjects": len(
            baseline
        ),
        "frozen_manifest_subjects": len(frozen),
        "excluded_independent_overlap": len(
            source_overlap
        ),
    }

    expected = {
        "total_represented_subjects": (
            EXPECTED_TOTAL_REPRESENTED_SUBJECTS
        ),
        "brats21_segmentation_overlap": (
            EXPECTED_BRATS21_OVERLAP
        ),
        "brats21_training": (
            EXPECTED_BRATS21_TRAINING
        ),
        "brats21_validation": (
            EXPECTED_BRATS21_VALIDATION
        ),
        "independent_subjects": (
            EXPECTED_INDEPENDENT_SUBJECTS
        ),
        "follow_up_subjects": (
            EXPECTED_FOLLOW_UP_SUBJECTS
        ),
        "baseline_independent_subjects": (
            EXPECTED_BASELINE_SUBJECTS
        ),
        "frozen_manifest_subjects": (
            EXPECTED_BASELINE_SUBJECTS
        ),
        "excluded_independent_overlap": 0,
    }

    for key, expected_value in expected.items():
        observed_value = checks[key]

        if observed_value != expected_value:
            raise RuntimeError(
                f"{key}: expected {expected_value}, "
                f"observed {observed_value}"
            )

    only_in_baseline = sorted(
        baseline_ids - frozen_ids
    )
    only_in_frozen = sorted(
        frozen_ids - baseline_ids
    )

    if only_in_baseline or only_in_frozen:
        raise RuntimeError(
            "Baseline cohort does not match the frozen "
            "external-validation manifest. "
            f"Only in baseline: {only_in_baseline}; "
            f"only in frozen: {only_in_frozen}"
        )

    overlap_flags = set(
        frozen["brats21_segmentation_overlap"]
        .str.strip()
        .str.lower()
    )

    if overlap_flags != {"false"}:
        raise RuntimeError(
            "Frozen manifest contains unexpected "
            "brats21_segmentation_overlap values: "
            f"{sorted(overlap_flags)}"
        )

    follow_up_flags = set(
        frozen["follow_up_exam"]
        .str.strip()
        .str.lower()
    )

    if follow_up_flags != {"false"}:
        raise RuntimeError(
            "Frozen manifest contains unexpected "
            "follow_up_exam values: "
            f"{sorted(follow_up_flags)}"
        )

    print(
        "UCSF-PDGM external cohort validation: PASS"
    )
    print()
    print("Resolved paths:")
    print("metadata_root:", metadata_root)
    print("frozen_manifest:", manifest_path)
    print()

    for key, value in checks.items():
        print(f"{key}: {value}")

    print()
    print("Excluded independent follow-up:")
    print(EXPECTED_FOLLOW_UP_ID)

    print()
    print(
        "Baseline/frozen subject intersection:",
        len(baseline_ids & frozen_ids),
    )
    print(
        "Only in baseline:",
        len(only_in_baseline),
    )
    print(
        "Only in frozen:",
        len(only_in_frozen),
    )

    print()
    print("SHA256:")
    print(
        BRATS_EXCLUSION_FILENAME + ":",
        sha256_file(exclusion_path),
    )
    print(
        INDEPENDENT_FILENAME + ":",
        sha256_file(independent_path),
    )
    print(
        BASELINE_FILENAME + ":",
        sha256_file(baseline_path),
    )
    print(
        manifest_path.name + ":",
        sha256_file(manifest_path),
    )


if __name__ == "__main__":
    main()
