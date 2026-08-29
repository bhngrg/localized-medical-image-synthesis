#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_ROOT = REPO_ROOT / "downstream_evaluation" / "manifests"

CATALOG_PATH = MANIFEST_ROOT / "brats_real_catalog.csv"
SUBJECT_SPLIT_PATH = MANIFEST_ROOT / "brats_downstream_subject_split.csv"
DONOR_POOL_PATH = MANIFEST_ROOT / "brats_downstream_training_donor_pool.csv"

LIBRARY_MANIFEST_PATH = (
    MANIFEST_ROOT
    / "br_lora_library_design_10000"
    / "br_lora_library_design_10000.csv"
)

OUTPUT_MANIFEST_PATH = (
    MANIFEST_ROOT
    / "downstream_real_training_manifest.csv"
)

OUTPUT_AUDIT_PATH = (
    MANIFEST_ROOT
    / "downstream_real_training_manifest_audit.json"
)


EXPECTED_FULL_CATALOG_ROWS = 57195
EXPECTED_FULL_SUBJECTS = 369
EXPECTED_TRAIN_SUBJECTS = 332
EXPECTED_VALIDATION_SUBJECTS = 37
EXPECTED_PRE_EXCLUSION_ROWS = 51460
EXPECTED_DONOR_SLICES = 10000
EXPECTED_COMMON_REAL_ROWS = 41460
EXPECTED_TUMOR_CONTAINING = 11975
EXPECTED_TUMOR_FREE = 29485


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    catalog = pd.read_csv(CATALOG_PATH)
    subject_split = pd.read_csv(SUBJECT_SPLIT_PATH)
    donor_pool = pd.read_csv(DONOR_POOL_PATH)
    library = pd.read_csv(LIBRARY_MANIFEST_PATH)

    train_volumes = set(
        subject_split.loc[
            subject_split["downstream_split"] == "train",
            "volume",
        ].astype(int)
    )

    validation_volumes = sorted(
        subject_split.loc[
            subject_split["downstream_split"] == "validation",
            "volume",
        ].astype(int)
    )

    donor_pool_volumes = set(
        donor_pool["volume"].astype(int)
    )

    if train_volumes != donor_pool_volumes:
        raise RuntimeError(
            "Downstream-training volume IDs do not match "
            "the frozen donor-pool volume IDs."
        )

    donor_keys = set(
        zip(
            library["donor_volume"].astype(int),
            library["donor_slice_index"].astype(int),
        )
    )

    train_catalog = catalog.loc[
        catalog["volume"].astype(int).isin(train_volumes)
    ].copy()

    train_catalog["_key"] = list(
        zip(
            train_catalog["volume"].astype(int),
            train_catalog["slice_index"].astype(int),
        )
    )

    manifest = train_catalog.loc[
        ~train_catalog["_key"].isin(donor_keys)
    ].copy()

    manifest = manifest.drop(columns="_key")

    manifest["downstream_split"] = "train"
    manifest["common_real_training"] = True
    manifest["is_synthetic_library_donor"] = False

    manifest = (
        manifest
        .sort_values(["volume", "slice_index"])
        .reset_index(drop=True)
    )

    manifest_keys = set(
        zip(
            manifest["volume"].astype(int),
            manifest["slice_index"].astype(int),
        )
    )

    checks = {
        "full_catalog_rows": len(catalog),
        "full_subjects": catalog["volume"].nunique(),
        "train_subjects": len(train_volumes),
        "validation_subjects": len(validation_volumes),
        "pre_exclusion_rows": len(train_catalog),
        "frozen_donor_slices": len(donor_keys),
        "common_real_rows": len(manifest),
        "common_real_subjects": manifest["volume"].nunique(),
        "tumor_containing": int(
            manifest["has_tumor"].astype(bool).sum()
        ),
        "tumor_free": int(
            (~manifest["has_tumor"].astype(bool)).sum()
        ),
        "residual_donor_overlap": len(
            manifest_keys & donor_keys
        ),
    }

    expected = {
        "full_catalog_rows": EXPECTED_FULL_CATALOG_ROWS,
        "full_subjects": EXPECTED_FULL_SUBJECTS,
        "train_subjects": EXPECTED_TRAIN_SUBJECTS,
        "validation_subjects": EXPECTED_VALIDATION_SUBJECTS,
        "pre_exclusion_rows": EXPECTED_PRE_EXCLUSION_ROWS,
        "frozen_donor_slices": EXPECTED_DONOR_SLICES,
        "common_real_rows": EXPECTED_COMMON_REAL_ROWS,
        "common_real_subjects": EXPECTED_TRAIN_SUBJECTS,
        "tumor_containing": EXPECTED_TUMOR_CONTAINING,
        "tumor_free": EXPECTED_TUMOR_FREE,
        "residual_donor_overlap": 0,
    }

    for key, expected_value in expected.items():
        observed_value = checks[key]

        if observed_value != expected_value:
            raise RuntimeError(
                f"{key}: expected {expected_value}, "
                f"observed {observed_value}"
            )

    if len(
        manifest.drop_duplicates(
            ["volume", "slice_index"]
        )
    ) != len(manifest):
        raise RuntimeError(
            "Duplicate (volume, slice_index) keys found."
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
            "Common real BraTS downstream-training pool for "
            "Experiments 1 and 2 after excluding every exact "
            "donor slice used in the frozen 10,000-case "
            "BR-LoRA synthetic library."
        ),
        "separation_level": "slice-level",
        "source_artifacts": {
            str(CATALOG_PATH.relative_to(REPO_ROOT)): {
                "sha256": sha256(CATALOG_PATH),
            },
            str(SUBJECT_SPLIT_PATH.relative_to(REPO_ROOT)): {
                "sha256": sha256(SUBJECT_SPLIT_PATH),
            },
            str(DONOR_POOL_PATH.relative_to(REPO_ROOT)): {
                "sha256": sha256(DONOR_POOL_PATH),
            },
            str(LIBRARY_MANIFEST_PATH.relative_to(REPO_ROOT)): {
                "sha256": sha256(LIBRARY_MANIFEST_PATH),
            },
        },
        "counts": checks,
        "downstream_validation_volume_ids": validation_volumes,
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
