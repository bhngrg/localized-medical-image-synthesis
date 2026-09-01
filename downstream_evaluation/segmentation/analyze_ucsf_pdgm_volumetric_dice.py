#!/usr/bin/env python3
"""
Analyze UCSF-PDGM external-validation subject-level volumetric Dice.

Locked analysis protocol:
- analysis unit: subject
- primary metric: volumetric_dice
- expected cohort size: 202 matched UCSF-PDGM subjects
- point estimate: arithmetic mean across subjects
- variability: sample standard deviation (ddof=1)
- bootstrap: nonparametric subject-level bootstrap with replacement
- bootstrap replicates: 10,000
- bootstrap seed: 2026
- interval: 95% percentile interval (2.5th, 97.5th percentiles)
- paired comparisons: resample subject IDs jointly across regimes
- same bootstrap index matrix for every regime and comparison
- slices are never treated as independent bootstrap units
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_EVALUATION_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "downstream_segmentation"
    / "evaluations"
    / "ucsf_pdgm_hardened_seed42"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "downstream_segmentation"
    / "external_validation"
    / "ucsf_pdgm"
    / "br_lora"
    / "seed_42"
)

HISTORICAL_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "historical"
    / "downstream_segmentation"
    / "external_validation"
    / "ucsf_pdgm"
    / "br_lora"
    / "seed_42"
)

ANALYSIS_ARTIFACT_NAMES = (
    "ucsf_pdgm_volumetric_dice_table.csv",
    "ucsf_pdgm_volumetric_dice_table.md",
    "ucsf_pdgm_volumetric_dice_analysis.json",
)

EXPECTED_SUBJECTS = 202
METRIC = "volumetric_dice"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2026
CI_LEVEL = 0.95
CI_QUANTILES = (0.025, 0.975)
SD_DDOF = 1

REGIMES = {
    "real_only": {
        "label": "Real only",
        "relative_path": Path("real_only") / "subject_metrics.csv",
    },
    "posterior_mean": {
        "label": "Real + BR-LoRA posterior mean",
        "relative_path": Path("real_plus_br_lora_posterior_mean") / "subject_metrics.csv",
    },
    "posterior_sampling": {
        "label": "Real + BR-LoRA posterior sampling",
        "relative_path": Path("real_plus_br_lora_posterior_sampling") / "subject_metrics.csv",
    },
}

PAIRED_COMPARISONS = [
    ("posterior_mean", "real_only"),
    ("posterior_sampling", "real_only"),
    ("posterior_sampling", "posterior_mean"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the locked UCSF-PDGM volumetric-Dice bootstrap analysis "
            "from matched subject-level metrics."
        )
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=DEFAULT_EVALUATION_ROOT,
        help=(
            "Evaluation directory containing the three regime subdirectories "
            "and subject_metrics.csv files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for curated analysis outputs. If omitted, the "
            "canonical UCSF-PDGM BR-LoRA result directory under results/ "
            "is used."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refuse_overwrite(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing analysis artifact(s):\n"
            f"{formatted}"
        )


def archive_existing_analysis_artifacts(
    output_dir: Path,
) -> Path | None:
    """
    Archive only artifacts owned by this analysis script.

    Other curated files that share the canonical result directory, including
    subject-level metrics, summaries, and run metadata, are left untouched.
    """
    existing = [
        output_dir / name
        for name in ANALYSIS_ARTIFACT_NAMES
        if (output_dir / name).exists()
    ]

    if not existing:
        return None

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")

    archive_dir = (
        HISTORICAL_OUTPUT_ROOT
        / timestamp
    )

    if archive_dir.exists():
        raise RuntimeError(
            "Historical analysis archive already exists:\n"
            f"{archive_dir}"
        )

    archive_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(
        "Archiving existing canonical analysis artifacts:"
    )
    print(
        "Historical destination:",
        archive_dir,
    )

    for source in existing:
        destination = archive_dir / source.name

        print(
            "  ",
            source,
            "->",
            destination,
        )

        shutil.move(
            str(source),
            str(destination),
        )

    return archive_dir


def load_subject_metric(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing subject metrics file: {path}")

    df = pd.read_csv(path)
    required = {"subject_id", METRIC}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {sorted(missing)}"
        )

    df = df[["subject_id", METRIC]].copy()

    if df["subject_id"].isna().any():
        raise ValueError(f"Missing subject_id values in {path}")
    if df["subject_id"].duplicated().any():
        duplicates = (
            df.loc[df["subject_id"].duplicated(), "subject_id"]
            .astype(str)
            .tolist()
        )
        raise ValueError(f"Duplicate subject IDs in {path}: {duplicates[:10]}")
    if df[METRIC].isna().any():
        raise ValueError(f"Missing {METRIC} values in {path}")

    values = df[METRIC].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite {METRIC} values in {path}")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError(f"{METRIC} values outside [0, 1] detected in {path}")

    return df.set_index("subject_id").sort_index()


def main() -> None:
    args = parse_args()
    evaluation_root = (
        args.evaluation_root
        .expanduser()
        .resolve()
    )

    using_default_output_dir = (
        args.output_dir is None
    )

    output_dir = (
        DEFAULT_OUTPUT_DIR
        if using_default_output_dir
        else args.output_dir.expanduser().resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = output_dir / "ucsf_pdgm_volumetric_dice_table.csv"
    md_path = output_dir / "ucsf_pdgm_volumetric_dice_table.md"
    json_path = output_dir / "ucsf_pdgm_volumetric_dice_analysis.json"

    dataframes: dict[str, pd.DataFrame] = {}
    input_files: dict[str, dict[str, str]] = {}

    for regime, spec in REGIMES.items():
        path = evaluation_root / spec["relative_path"]
        df = load_subject_metric(path)
        dataframes[regime] = df
        input_files[regime] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }

    reference_ids = list(dataframes["real_only"].index)
    reference_set = set(reference_ids)
    if len(reference_ids) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            f"Expected {EXPECTED_SUBJECTS} real-only subjects, "
            f"found {len(reference_ids)}."
        )

    for regime, df in dataframes.items():
        ids = set(df.index)
        if ids != reference_set:
            missing = sorted(reference_set - ids)
            extra = sorted(ids - reference_set)
            raise RuntimeError(
                f"Subject IDs do not match for {regime}. "
                f"Missing={missing[:10]}, extra={extra[:10]}"
            )

    subjects = sorted(reference_set)
    n_subjects = len(subjects)
    values = {
        regime: df.loc[subjects, METRIC].to_numpy(dtype=float)
        for regime, df in dataframes.items()
    }

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_indices = rng.integers(
        0,
        n_subjects,
        size=(BOOTSTRAP_REPLICATES, n_subjects),
    )

    regime_results: dict[str, dict] = {}
    for regime, spec in REGIMES.items():
        x = values[regime]
        bootstrap_means = x[bootstrap_indices].mean(axis=1)
        ci_low, ci_high = np.quantile(bootstrap_means, CI_QUANTILES)
        regime_results[regime] = {
            "label": spec["label"],
            "n_subjects": n_subjects,
            "mean": float(np.mean(x)),
            "sd": float(np.std(x, ddof=SD_DDOF)),
            "ci_95_low": float(ci_low),
            "ci_95_high": float(ci_high),
        }

    comparison_results: dict[str, dict] = {}
    for numerator, denominator in PAIRED_COMPARISONS:
        paired_difference = values[numerator] - values[denominator]
        bootstrap_differences = paired_difference[bootstrap_indices].mean(axis=1)
        ci_low, ci_high = np.quantile(bootstrap_differences, CI_QUANTILES)
        key = f"{numerator}_minus_{denominator}"
        comparison_results[key] = {
            "numerator": numerator,
            "denominator": denominator,
            "n_subjects": n_subjects,
            "mean_difference": float(np.mean(paired_difference)),
            "ci_95_low": float(ci_low),
            "ci_95_high": float(ci_high),
        }

    table_rows = []
    for regime in ("real_only", "posterior_mean", "posterior_sampling"):
        stats = regime_results[regime]
        if regime == "real_only":
            delta = delta_low = delta_high = None
        else:
            comparison = comparison_results[f"{regime}_minus_real_only"]
            delta = comparison["mean_difference"]
            delta_low = comparison["ci_95_low"]
            delta_high = comparison["ci_95_high"]

        table_rows.append(
            {
                "training_regime": stats["label"],
                "n_subjects": n_subjects,
                "mean_volumetric_dice": stats["mean"],
                "sd_volumetric_dice": stats["sd"],
                "ci_95_low": stats["ci_95_low"],
                "ci_95_high": stats["ci_95_high"],
                "paired_delta_vs_real_only": delta,
                "paired_delta_ci_95_low": delta_low,
                "paired_delta_ci_95_high": delta_high,
            }
        )

    table = pd.DataFrame(table_rows)

    def fmt_mean_sd(row: pd.Series) -> str:
        return f"{row['mean_volumetric_dice']:.3f} ± {row['sd_volumetric_dice']:.3f}"

    def fmt_ci(row: pd.Series) -> str:
        return f"[{row['ci_95_low']:.3f}, {row['ci_95_high']:.3f}]"

    def fmt_delta(row: pd.Series) -> str:
        value = row["paired_delta_vs_real_only"]
        if pd.isna(value):
            return "—"
        return (
            f"{value:+.3f} "
            f"[{row['paired_delta_ci_95_low']:.3f}, "
            f"{row['paired_delta_ci_95_high']:.3f}]"
        )

    display_table = pd.DataFrame(
        {
            "Training regime": table["training_regime"],
            "External volumetric Dice (mean ± SD)": table.apply(fmt_mean_sd, axis=1),
            "95% CI": table.apply(fmt_ci, axis=1),
            "Paired Δ vs. real-only": table.apply(fmt_delta, axis=1),
        }
    )

    markdown_text = (
        display_table.to_markdown(index=False)
        + "\n"
    )

    settings = {
        "analysis_name": "UCSF-PDGM external volumetric Dice bootstrap",
        "analysis_unit": "subject",
        "primary_metric": METRIC,
        "expected_subject_count": EXPECTED_SUBJECTS,
        "observed_subject_count": n_subjects,
        "point_estimate": "arithmetic mean across subjects",
        "variability": {
            "statistic": "sample standard deviation across subjects",
            "ddof": SD_DDOF,
        },
        "bootstrap": {
            "type": "nonparametric subject-level bootstrap with replacement",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": CI_LEVEL,
            "interval_method": "percentile",
            "quantiles": list(CI_QUANTILES),
            "paired_comparisons": True,
            "pairing_unit": "subject_id",
            "same_resample_indices_for_all_regimes_and_comparisons": True,
            "slice_level_resampling": False,
        },
        "table_primary_comparisons": [
            "posterior_mean_minus_real_only",
            "posterior_sampling_minus_real_only",
        ],
        "additional_paired_comparison": "posterior_sampling_minus_posterior_mean",
        "regimes": {
            key: {
                "label": value["label"],
                "relative_subject_metrics_path": str(value["relative_path"]),
            }
            for key, value in REGIMES.items()
        },
    }

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_root": str(evaluation_root),
        "settings": settings,
        "input_files": input_files,
        "results": {
            "regimes": regime_results,
            "paired_comparisons": comparison_results,
        },
        "artifacts": {
            "table_csv": str(csv_path),
            "table_markdown": str(md_path),
            "analysis_json": str(json_path),
        },
    }

    if using_default_output_dir:
        archive_existing_analysis_artifacts(
            output_dir
        )
    else:
        refuse_overwrite(
            [
                csv_path,
                md_path,
                json_path,
            ]
        )

    table.to_csv(
        csv_path,
        index=False,
    )

    md_path.write_text(
        markdown_text,
        encoding="utf-8",
    )

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(display_table.to_string(index=False))
    print()
    print("Locked bootstrap settings:")
    print("  Analysis unit          : subject")
    print(f"  Metric                 : {METRIC}")
    print(f"  Matched subjects       : {n_subjects}")
    print(f"  Bootstrap replicates   : {BOOTSTRAP_REPLICATES:,}")
    print(f"  Bootstrap seed         : {BOOTSTRAP_SEED}")
    print("  Interval               : 95% percentile")
    print(f"  Sample SD ddof         : {SD_DDOF}")
    print("  Paired resampling      : subject_id")
    print("  Shared bootstrap draws : yes")
    print()
    print("Saved:")
    print(f"  {csv_path}")
    print(f"  {md_path}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()
