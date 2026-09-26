#!/usr/bin/env python3
"""
Consolidate locked feathered and non-feathered UCSF-PDGM PEFT results.

This is a reporting step only. It reads the two existing locked analysis
tables and places their results side by side. It does not recompute Dice
values, bootstrap estimates, confidence intervals, or paired comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PEFT_RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "downstream_segmentation"
    / "external_validation"
    / "ucsf_pdgm"
    / "peft_comparison"
)

DEFAULT_FEATHERED_TABLE = (
    PEFT_RESULT_ROOT
    / "feathered"
    / "seed_42"
    / "ucsf_pdgm_volumetric_dice_table.csv"
)

DEFAULT_NON_FEATHERED_TABLE = (
    PEFT_RESULT_ROOT
    / "non_feathered"
    / "seed_42"
    / "ucsf_pdgm_volumetric_dice_table.csv"
)

DEFAULT_OUTPUT_DIR = (
    PEFT_RESULT_ROOT
    / "combined"
    / "seed_42"
)

OUTPUT_NAMES = (
    "ucsf_pdgm_feathering_comparison_table.csv",
    "ucsf_pdgm_feathering_comparison_table.md",
    "ucsf_pdgm_feathering_comparison.json",
)

REQUIRED_COLUMNS = (
    "training_regime",
    "n_subjects",
    "mean_volumetric_dice",
    "bootstrap_se_volumetric_dice",
    "ci_95_low",
    "ci_95_high",
    "paired_delta_vs_real_only",
    "paired_delta_bootstrap_se_vs_real_only",
    "paired_delta_ci_95_low",
    "paired_delta_ci_95_high",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate locked feathered and non-feathered UCSF-PDGM "
            "PEFT volumetric-Dice result tables."
        )
    )
    parser.add_argument(
        "--feathered-table",
        type=Path,
        default=DEFAULT_FEATHERED_TABLE,
    )
    parser.add_argument(
        "--non-feathered-table",
        type=Path,
        default=DEFAULT_NON_FEATHERED_TABLE,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_table(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label} table: {path}")

    table = pd.read_csv(path)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in table.columns
    ]
    if missing:
        raise ValueError(
            f"{label} table is missing required columns: {missing}"
        )

    if table["training_regime"].isna().any():
        raise ValueError(f"{label} table contains missing training regimes.")

    if table["training_regime"].duplicated().any():
        raise ValueError(
            f"{label} table contains duplicate training regimes."
        )

    return table.loc[:, REQUIRED_COLUMNS].copy()


def refuse_overwrite(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing combined artifact(s):\n"
            f"{formatted}"
        )


def fmt_mean_se(mean: float, se: float) -> str:
    return f"{mean:.3f} ± {se:.3f}"


def fmt_ci(low: float, high: float) -> str:
    return f"[{low:.3f}, {high:.3f}]"


def fmt_delta(value: float, low: float, high: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:+.3f} [{low:.3f}, {high:.3f}]"


def main() -> None:
    args = parse_args()

    feathered_path = args.feathered_table.expanduser().resolve()
    non_feathered_path = args.non_feathered_table.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    feathered = load_table(feathered_path, "feathered")
    non_feathered = load_table(non_feathered_path, "non-feathered")

    feathered_regimes = feathered["training_regime"].tolist()
    non_feathered_regimes = non_feathered["training_regime"].tolist()

    if feathered_regimes != non_feathered_regimes:
        raise RuntimeError(
            "Feathered and non-feathered training regimes or row order "
            "do not match."
        )

    if not (
        feathered["n_subjects"].to_numpy()
        == non_feathered["n_subjects"].to_numpy()
    ).all():
        raise RuntimeError(
            "Feathered and non-feathered subject counts do not match."
        )

    combined = pd.DataFrame(
        {
            "training_regime": feathered["training_regime"],
            "n_subjects": feathered["n_subjects"],
            "feathered_mean_volumetric_dice":
                feathered["mean_volumetric_dice"],
            "feathered_bootstrap_se_volumetric_dice":
                feathered["bootstrap_se_volumetric_dice"],
            "feathered_ci_95_low": feathered["ci_95_low"],
            "feathered_ci_95_high": feathered["ci_95_high"],
            "feathered_paired_delta_vs_real_only":
                feathered["paired_delta_vs_real_only"],
            "feathered_paired_delta_ci_95_low":
                feathered["paired_delta_ci_95_low"],
            "feathered_paired_delta_ci_95_high":
                feathered["paired_delta_ci_95_high"],
            "non_feathered_mean_volumetric_dice":
                non_feathered["mean_volumetric_dice"],
            "non_feathered_bootstrap_se_volumetric_dice":
                non_feathered["bootstrap_se_volumetric_dice"],
            "non_feathered_ci_95_low": non_feathered["ci_95_low"],
            "non_feathered_ci_95_high": non_feathered["ci_95_high"],
            "non_feathered_paired_delta_vs_real_only":
                non_feathered["paired_delta_vs_real_only"],
            "non_feathered_paired_delta_ci_95_low":
                non_feathered["paired_delta_ci_95_low"],
            "non_feathered_paired_delta_ci_95_high":
                non_feathered["paired_delta_ci_95_high"],
        }
    )

    display_rows = []
    for index in range(len(feathered)):
        f = feathered.iloc[index]
        nf = non_feathered.iloc[index]

        display_rows.append(
            {
                "Training regime": f["training_regime"],
                "Feathered Dice (mean ± bootstrap SE)": fmt_mean_se(
                    f["mean_volumetric_dice"],
                    f["bootstrap_se_volumetric_dice"],
                ),
                "Feathered 95% CI": fmt_ci(
                    f["ci_95_low"],
                    f["ci_95_high"],
                ),
                "Feathered Δ vs. real-only": fmt_delta(
                    f["paired_delta_vs_real_only"],
                    f["paired_delta_ci_95_low"],
                    f["paired_delta_ci_95_high"],
                ),
                "Non-feathered Dice (mean ± bootstrap SE)": fmt_mean_se(
                    nf["mean_volumetric_dice"],
                    nf["bootstrap_se_volumetric_dice"],
                ),
                "Non-feathered 95% CI": fmt_ci(
                    nf["ci_95_low"],
                    nf["ci_95_high"],
                ),
                "Non-feathered Δ vs. real-only": fmt_delta(
                    nf["paired_delta_vs_real_only"],
                    nf["paired_delta_ci_95_low"],
                    nf["paired_delta_ci_95_high"],
                ),
            }
        )

    display_table = pd.DataFrame(display_rows)

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / OUTPUT_NAMES[0]
    md_path = output_dir / OUTPUT_NAMES[1]
    json_path = output_dir / OUTPUT_NAMES[2]

    refuse_overwrite([csv_path, md_path, json_path])

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": (
            "Side-by-side consolidation of locked feathered and "
            "non-feathered UCSF-PDGM PEFT results"
        ),
        "note": (
            "Reporting-only consolidation. No statistical estimates were "
            "recomputed and no feathered-versus-non-feathered hypothesis "
            "test or paired comparison was performed."
        ),
        "inputs": {
            "feathered": {
                "path": str(feathered_path),
                "sha256": sha256_file(feathered_path),
            },
            "non_feathered": {
                "path": str(non_feathered_path),
                "sha256": sha256_file(non_feathered_path),
            },
        },
        "validation": {
            "training_regimes_match": True,
            "row_order_matches": True,
            "subject_counts_match": True,
            "training_regimes": feathered_regimes,
            "n_subjects": [
                int(value)
                for value in feathered["n_subjects"].tolist()
            ],
        },
        "artifacts": {
            "table_csv": str(csv_path),
            "table_markdown": str(md_path),
            "provenance_json": str(json_path),
        },
    }

    combined.to_csv(csv_path, index=False)

    md_path.write_text(
        display_table.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )

    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(display_table.to_string(index=False))
    print()
    print("Reporting-only consolidation:")
    print("  Statistical recomputation       : no")
    print("  Feathered vs non-feathered test : no")
    print("  Training regimes match          : yes")
    print("  Subject counts match            : yes")
    print()
    print("Saved:")
    print(f"  {csv_path}")
    print(f"  {md_path}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()
