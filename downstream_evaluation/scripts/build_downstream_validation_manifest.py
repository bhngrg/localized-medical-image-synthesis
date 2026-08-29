#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_ROOT = REPO_ROOT / "downstream_evaluation" / "manifests"

CATALOG_PATH = MANIFEST_ROOT / "brats_real_catalog.csv"
SUBJECT_SPLIT_PATH = MANIFEST_ROOT / "brats_downstream_subject_split.csv"

OUTPUT_MANIFEST_PATH = (
    MANIFEST_ROOT
    / "downstream_validation_manifest.csv"
)

OUTPUT_AUDIT_PATH = (
    MANIFEST_ROOT
    / "downstream_validation_manifest_audit.json"
)


EXPECTED_FULL_CATALOG_ROWS = 57195
EXPECTED_FULL_SUBJECTS = 369
EXPECTED_TRAIN_SUBJECTS = 332
EXPECTED_VALIDATION_SUBJECTS = 37
EXPECTED_VALIDATION_ROWS = 5735
EXPECTED_TUMOR_CONTAINING = 2447
EXPECTED_TUMOR_FREE = 3288


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    catalog = pd.read_csv(CATALOG_PATH)
    subject_split = pd.read_csv(SUBJECT_SPLIT_PATH)

    train_volumes = set(
        subject_split.loc[
            subject_split["downstream_split"] == "train",
            "volume",
        ].astype(int)
    )

    validation_volumes = set(
        subject_split.loc[
            subject_split["downstream_split"] == "validation",
            "volume",
        ].astype(int)
    )

    if train_volumes & validation_volumes:
        raise RuntimeError(
            "Train and validation subject sets overlap."
        )

    manifest = catalog.loc[
        catalog["volume"].astype(int).isin(validation_volumes)
    ].copy()

    manifest["downstream_split"] = "validation"

    manifest = (
        manifest
        .sort_values(["volume", "slice_index"])
        .reset_index(drop=True)
    )

    checks = {
        "full_catalog_rows": len(catalog),
        "full_subjects": catalog["volume"].nunique(),
        "train_subjects": len(train_volumes),
        "validation_subjects": len(validation_volumes),
        "validation_rows": len(manifest),
        "unique_validation_subjects": manifest["volume"].nunique(),
        "unique_validation_keys": len(
            manifest.drop_duplicates(["volume", "slice_index"])
        ),
        "tumor_containing": int(
            manifest["has_tumor"].astype(bool).sum()
        ),
        "tumor_free": int(
            (~manifest["has_tumor"].astype(bool)).sum()
        ),
        "train_validation_subject_overlap": len(
            train_volumes & validation_volumes
        ),
    }

    expected = {
        "full_catalog_rows": EXPECTED_FULL_CATALOG_ROWS,
        "full_subjects": EXPECTED_FULL_SUBJECTS,
        "train_subjects": EXPECTED_TRAIN_SUBJECTS,
        "validation_subjects": EXPECTED_VALIDATION_SUBJECTS,
        "validation_rows": EXPECTED_VALIDATION_ROWS,
        "unique_validation_subjects": EXPECTED_VALIDATION_SUBJECTS,
        "unique_validation_keys": EXPECTED_VALIDATION_ROWS,
        "tumor_containing": EXPECTED_TUMOR_CONTAINING,
        "tumor_free": EXPECTED_TUMOR_FREE,
        "train_validation_subject_overlap": 0,
    }

    for key, expected_value in expected.items():
        observed_value = checks[key]

        if observed_value != expected_value:
            raise RuntimeError(
                f"{key}: expected {expected_value}, "
                f"observed {observed_value}"
            )

    counts = manifest.groupby("volume")["slice_index"].count()

    if counts.min() != 155 or counts.max() != 155:
        raise RuntimeError(
            "Validation subjects do not all contain exactly 155 slices."
        )

    expected_slice_indices = set(range(155))

    for volume, group in manifest.groupby("volume"):
        observed = set(group["slice_index"].astype(int))

        if observed != expected_slice_indices:
            raise RuntimeError(
                f"Validation volume {int(volume)} does not contain "
                "exactly slice indices 0 through 154."
            )

    tumor_rule_matches = (
        manifest["has_tumor"].astype(bool)
        == (manifest["whole_tumor_pixels"] > 0)
    )

    if not tumor_rule_matches.all():
        raise RuntimeError(
            "has_tumor and whole_tumor_pixels are inconsistent."
        )

    manifest.to_csv(
        OUTPUT_MANIFEST_PATH,
        index=False,
    )

    audit = {
        "artifact": OUTPUT_MANIFEST_PATH.name,
        "purpose": (
            "Frozen BraTS downstream validation cohort used for "
            "model selection in downstream segmentation experiments."
        ),
        "separation_level": "subject-level",
        "source_artifacts": {
            str(CATALOG_PATH.relative_to(REPO_ROOT)): {
                "sha256": sha256(CATALOG_PATH),
            },
            str(SUBJECT_SPLIT_PATH.relative_to(REPO_ROOT)): {
                "sha256": sha256(SUBJECT_SPLIT_PATH),
            },
        },
        "counts": checks,
        "validation_volume_ids": sorted(validation_volumes),
        "manifest_sha256": sha256(OUTPUT_MANIFEST_PATH),
    }

    with OUTPUT_AUDIT_PATH.open("w") as f:
        json.dump(audit, f, indent=2)
        f.write("\n")

    print("Created:")
    print(OUTPUT_MANIFEST_PATH)
    print(OUTPUT_AUDIT_PATH)

    print()

    for key, value in checks.items():
        print(f"{key}: {value}")

    print()
    print(
        "Manifest SHA256:",
        sha256(OUTPUT_MANIFEST_PATH),
    )
    print(
        "Audit SHA256:",
        sha256(OUTPUT_AUDIT_PATH),
    )


if __name__ == "__main__":
    main()
