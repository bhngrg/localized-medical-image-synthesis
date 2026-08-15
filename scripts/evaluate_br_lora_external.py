#!/usr/bin/env python3

"""
Evaluate Bayesian Regional LoRA (BR-LoRA) on a fixed external BraTS cohort.

The evaluator consumes

- a registered BraTS 2020 validation-dataset specification,
- a fixed external evaluation manifest,
- a fitted BR-LoRA checkpoint, and
- the validated baseline / BR-LoRA configurations.

External base images are loaded from the official BraTS 2020 validation
release. Donor pathological appearance and transferred lesion masks are loaded
from labeled BraTS training H5 slices specified explicitly by the evaluation
manifest.

The evaluator does not discover or select cases internally. This guarantees
that different trained models can be evaluated on exactly the same external
cases.

Posterior inference uses a fixed prepared diffusion realization for each case
so that variation across posterior realizations is attributable to sampled
BR-LoRA adapter parameters rather than resampled diffusion noise.

This script produces inference artifacts only. Reliability metrics and
downstream benchmarking are intentionally outside its scope.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

# Keep NumPy before PyTorch for the current macOS development environment.
import numpy as np
import torch
import yaml

from src.data import (
    load_validation_dataset_specification,
)
from src.diffusion import DiffusionSchedule
from src.inference import (
    compute_posterior_products,
    load_external_evaluation_manifest,
    load_fitted_br_lora,
    posterior_sample_inference,
    prepare_br_lora_batch,
    prepare_external_pair,
    reconstruct_composite_mean,
)


EVALUATION_NAME = "br_lora_external_evaluation"

DEFAULT_POSTERIOR_SAMPLES = 100


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a fitted BR-LoRA model on a fixed external BraTS "
            "validation cohort."
        )
    )

    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=Path(
            "configs/baseline_patch_x0.yaml"
        ),
        help=(
            "Baseline configuration used to reconstruct the validated "
            "diffusion and image-preprocessing contract."
        ),
    )

    parser.add_argument(
        "--br-lora-config",
        type=Path,
        default=Path(
            "configs/br_lora.yaml"
        ),
        help=(
            "Bayesian Regional LoRA configuration."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Path to a fitted BR-LoRA checkpoint."
        ),
    )

    parser.add_argument(
        "--validation-dataset",
        type=Path,
        required=True,
        help=(
            "Registered BraTS 2020 validation_dataset.yaml."
        ),
    )

    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        required=True,
        help=(
            "Fixed external evaluation case manifest."
        ),
    )

    parser.add_argument(
        "--posterior-samples",
        type=int,
        default=DEFAULT_POSTERIOR_SAMPLES,
        help=(
            "Number of BR-LoRA posterior realizations generated per case. "
            f"Default: {DEFAULT_POSTERIOR_SAMPLES}."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional evaluation seed override. When omitted, the baseline "
            "configuration seed is used."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/br_lora_external_evaluation"
        ),
        help=(
            "Root directory for external BR-LoRA evaluation artifacts."
        ),
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "mps",
            "cuda",
        ],
        default="auto",
        help=(
            "Execution device."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted evaluation by validating and skipping "
            "case directories that already contain complete matching "
            "artifacts. Incomplete or inconsistent existing cases cause "
            "a hard failure."
        ),
    )

    return parser.parse_args()


def load_config(
    path: Path,
    *,
    name: str,
) -> dict:
    """Load one YAML configuration mapping."""

    resolved = (
        path
        .expanduser()
        .resolve()
    )

    if not resolved.is_file():
        raise FileNotFoundError(
            f"{name} not found:\n{resolved}"
        )

    with resolved.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(
            file
        )

    if not isinstance(
        config,
        dict,
    ):
        raise ValueError(
            f"{name} must contain a YAML mapping."
        )

    return config


def resolve_device(
    requested: str,
) -> torch.device:
    """Resolve an explicit or automatic Torch device."""

    if requested == "cpu":
        return torch.device(
            "cpu"
        )

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable."
            )

        return torch.device(
            "cuda"
        )

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested but is unavailable."
            )

        return torch.device(
            "mps"
        )

    if torch.cuda.is_available():
        return torch.device(
            "cuda"
        )

    if torch.backends.mps.is_available():
        return torch.device(
            "mps"
        )

    return torch.device(
        "cpu"
    )


def set_seed(
    seed: int,
) -> None:
    """Seed Python, NumPy, and Torch."""

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )


def derive_case_seed(
    *,
    evaluation_seed: int,
    case_id: str,
) -> int:
    """
    Derive one stable per-case seed from the evaluation seed and case id.

    SHA-256 is used instead of Python's built-in ``hash`` so the mapping is
    stable across Python processes and interpreter hash-randomization states.
    """

    payload = (
        f"{evaluation_seed}:{case_id}"
        .encode(
            "utf-8"
        )
    )

    digest = hashlib.sha256(
        payload
    ).digest()

    return int.from_bytes(
        digest[
            :8
        ],
        byteorder="big",
        signed=False,
    ) % (
        2**31
    )


def resolve_git_commit() -> str | None:
    """Return the current Git commit when available."""

    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        return None

    commit = result.stdout.strip()

    if not commit:
        return None

    return commit


def cpu_tensor(
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Detach one tensor and move it to CPU for portable serialization."""

    return (
        tensor
        .detach()
        .cpu()
    )


def write_json(
    path: Path,
    payload: dict,
) -> None:
    """Write one JSON metadata artifact."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            sort_keys=True,
        )

        file.write(
            "\n"
        )


def validate_completed_case_directory(
    *,
    case_dir: Path,
    case_id: str,
    checkpoint_path: Path,
    evaluation_manifest_path: Path,
    evaluation_seed: int,
    posterior_samples: int,
) -> None:
    """
    Validate one existing case directory before it is accepted during resume.

    Resume never trusts directory existence alone. Every expected case artifact
    must exist and metadata must match the current evaluation contract.
    """

    required_artifacts = (
        "posterior_samples.pt",
        "posterior_mean.pt",
        "posterior_variance.pt",
        "posterior_std.pt",
        "composite_mean.pt",
        "metadata.json",
    )

    missing = [
        name
        for name in required_artifacts
        if not (
            case_dir
            / name
        ).is_file()
    ]

    if missing:
        raise RuntimeError(
            "Existing case directory is incomplete and cannot be resumed "
            "safely. Remove or repair this case directory before retrying.\n\n"
            f"Case: {case_id}\n"
            f"Directory: {case_dir}\n"
            "Missing artifact(s):\n"
            + "\n".join(
                f"  {name}"
                for name in missing
            )
        )

    metadata_path = (
        case_dir
        / "metadata.json"
    )

    try:
        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(
                file
            )

    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise RuntimeError(
            "Existing case metadata could not be validated for resume.\n\n"
            f"Case: {case_id}\n"
            f"Metadata: {metadata_path}"
        ) from exc

    expected = {
        "evaluation_name":
            EVALUATION_NAME,

        "case_id":
            case_id,

        "checkpoint":
            str(
                checkpoint_path
            ),

        "evaluation_manifest":
            str(
                evaluation_manifest_path
            ),

        "evaluation_seed":
            evaluation_seed,

        "posterior_samples":
            posterior_samples,

        "resample_diffusion_noise":
            False,
    }

    mismatches = []

    for key, expected_value in expected.items():
        observed_value = metadata.get(
            key
        )

        if observed_value != expected_value:
            mismatches.append(
                f"  {key}: expected {expected_value!r}, "
                f"observed {observed_value!r}"
            )

    if mismatches:
        raise RuntimeError(
            "Existing case directory does not match the current evaluation "
            "contract and cannot be skipped during resume.\n\n"
            f"Case: {case_id}\n"
            f"Directory: {case_dir}\n"
            "Metadata mismatch(es):\n"
            + "\n".join(
                mismatches
            )
        )


def main() -> None:
    """Construct the fixed external BR-LoRA evaluation state."""

    args = parse_args()

    baseline_config = load_config(
        args.baseline_config,
        name="Baseline configuration",
    )

    br_lora_config = load_config(
        args.br_lora_config,
        name="BR-LoRA configuration",
    )

    data_cfg = baseline_config[
        "data"
    ]

    diffusion_cfg = baseline_config[
        "diffusion"
    ]

    baseline_inference_cfg = baseline_config.get(
        "inference",
        {},
    )

    br_inference_cfg = br_lora_config.get(
        "inference",
        {},
    )

    if not isinstance(
        data_cfg,
        dict,
    ):
        raise ValueError(
            "Baseline data configuration must be a mapping."
        )

    if not isinstance(
        diffusion_cfg,
        dict,
    ):
        raise ValueError(
            "Baseline diffusion configuration must be a mapping."
        )

    if not isinstance(
        baseline_inference_cfg,
        dict,
    ):
        raise ValueError(
            "Baseline inference configuration must be a mapping."
        )

    if not isinstance(
        br_inference_cfg,
        dict,
    ):
        raise ValueError(
            "BR-LoRA inference configuration must be a mapping."
        )

    seed = (
        int(
            baseline_config.get(
                "seed",
                42,
            )
        )
        if args.seed is None
        else int(
            args.seed
        )
    )

    posterior_samples = int(
        args.posterior_samples
    )

    if posterior_samples <= 1:
        raise ValueError(
            "posterior_samples must be greater than one."
        )

    resample_diffusion_noise = bool(
        br_inference_cfg.get(
            "resample_diffusion_noise",
            False,
        )
    )

    if resample_diffusion_noise:
        raise ValueError(
            "External BR-LoRA posterior evaluation requires "
            "inference.resample_diffusion_noise=false so variation across "
            "posterior realizations is isolated from diffusion-noise "
            "variation."
        )

    image_channel = int(
        data_cfg.get(
            "image_channel",
            0,
        )
    )

    if image_channel < 0:
        raise ValueError(
            "data.image_channel must be non-negative."
        )

    timestep_fraction = float(
        baseline_inference_cfg.get(
            "timestep_fraction",
            0.75,
        )
    )

    if not (
        0.0
        <= timestep_fraction
        <= 1.0
    ):
        raise ValueError(
            "inference.timestep_fraction must lie in [0, 1]."
        )

    device = resolve_device(
        args.device
    )

    set_seed(
        seed
    )

    checkpoint_path = (
        args.checkpoint
        .expanduser()
        .resolve()
    )

    validation_dataset_path = (
        args.validation_dataset
        .expanduser()
        .resolve()
    )

    evaluation_manifest_path = (
        args.evaluation_manifest
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"BR-LoRA checkpoint not found:\n{checkpoint_path}"
        )

    if not validation_dataset_path.is_file():
        raise FileNotFoundError(
            "Validation dataset specification not found:\n"
            f"{validation_dataset_path}"
        )

    if not evaluation_manifest_path.is_file():
        raise FileNotFoundError(
            "External evaluation manifest not found:\n"
            f"{evaluation_manifest_path}"
        )

    validation_dataset = (
        load_validation_dataset_specification(
            validation_dataset_path
        )
    )

    cases = load_external_evaluation_manifest(
        evaluation_manifest_path
    )

    loaded = load_fitted_br_lora(
        checkpoint_path,
        device=device,
    )

    model = loaded.model

    schedule = DiffusionSchedule(
        timesteps=int(
            diffusion_cfg.get(
                "timesteps",
                200,
            )
        ),
        beta_start=float(
            diffusion_cfg.get(
                "beta_start",
                1.0e-4,
            )
        ),
        beta_end=float(
            diffusion_cfg.get(
                "beta_end",
                0.02,
            )
        ),
        device=device,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    git_commit = resolve_git_commit()

    print()
    print(
        "=" * 78
    )
    print(
        "BR-LoRA EXTERNAL EVALUATION CONSTRUCTION"
    )
    print(
        "=" * 78
    )

    print()
    print(
        "Evaluation"
    )
    print(
        "-" * 78
    )

    print(
        "Evaluation name          :",
        EVALUATION_NAME,
    )

    print(
        "Checkpoint               :",
        checkpoint_path,
    )

    print(
        "Device                   :",
        device,
    )

    print(
        "Seed                     :",
        seed,
    )

    print(
        "Posterior realizations   :",
        posterior_samples,
    )

    print(
        "Fixed diffusion noise    :",
        True,
    )

    print(
        "Timestep fraction        :",
        timestep_fraction,
    )

    print(
        "Resume mode              :",
        args.resume,
    )

    print()
    print(
        "External dataset"
    )
    print(
        "-" * 78
    )

    print(
        "Specification            :",
        validation_dataset_path,
    )

    print(
        "Dataset root             :",
        validation_dataset.raw_data_root,
    )

    print(
        "Registered subjects      :",
        validation_dataset.subject_count,
    )

    print()
    print(
        "Evaluation manifest"
    )
    print(
        "-" * 78
    )

    print(
        "Manifest                 :",
        evaluation_manifest_path,
    )

    print(
        "Cases                    :",
        len(
            cases
        ),
    )

    print()
    print(
        "BR-LoRA"
    )
    print(
        "-" * 78
    )

    print(
        "Variational modules      :",
        len(
            loaded.variational_module_names
        ),
    )

    print(
        "Variational parameters   :",
        f"{loaded.variational_parameter_count:,}",
    )

    print(
        "Donor image channel      :",
        image_channel,
    )

    print()
    print(
        "Provenance"
    )
    print(
        "-" * 78
    )

    print(
        "Git commit               :",
        (
            git_commit
            if git_commit is not None
            else "unavailable"
        ),
    )

    print(
        "Future output directory  :",
        output_dir,
    )

    print()
    print(
        "=" * 78
    )

    print(
        "CONSTRUCTION VERDICT: PASS"
    )

    print(
        "The fitted BR-LoRA checkpoint, registered external dataset, fixed "
        "evaluation manifest, and diffusion schedule were constructed "
        "successfully."
    )

    print(
        "=" * 78
    )

    print()
    print(
        "=" * 78
    )

    print(
        "RUNNING EXTERNAL POSTERIOR INFERENCE"
    )

    print(
        "=" * 78
    )

    evaluation_started_utc = datetime.now(
        timezone.utc
    ).isoformat()

    completed_cases = 0

    for case_index, case in enumerate(
        cases,
        start=1,
    ):
        print()
        print(
            f"[{case_index}/{len(cases)}] {case.case_id}"
        )

        case_dir = (
            output_dir
            / case.case_id
        )

        if case_dir.exists():
            if not args.resume:
                raise RuntimeError(
                    "External evaluation case output already exists. "
                    "Refusing to overwrite it.\n\n"
                    f"Case: {case.case_id}\n"
                    f"Directory: {case_dir}\n\n"
                    "Use --resume only when continuing the same evaluation."
                )

            validate_completed_case_directory(
                case_dir=case_dir,
                case_id=case.case_id,
                checkpoint_path=checkpoint_path,
                evaluation_manifest_path=evaluation_manifest_path,
                evaluation_seed=seed,
                posterior_samples=posterior_samples,
            )

            completed_cases += 1

            print(
                "  Existing completed case : VALIDATED"
            )

            print(
                "  Case output             :",
                case_dir,
            )

            print(
                "  Verdict                 : SKIPPED (resume)"
            )

            continue

        case_seed = derive_case_seed(
            evaluation_seed=seed,
            case_id=case.case_id,
        )

        set_seed(
            case_seed
        )

        external_pair = prepare_external_pair(
            case=case,
            validation_dataset=validation_dataset,
            donor_image_channel=image_channel,
        )

        prepared = prepare_br_lora_batch(
            external_pair.batch,
            schedule=schedule,
            device=device,
            timestep_fraction=timestep_fraction,
        )

        if prepared.target.shape != (
            1,
            1,
            240,
            240,
        ):
            raise RuntimeError(
                "Prepared external BR-LoRA target has an unexpected shape.\n"
                f"Observed: {tuple(prepared.target.shape)}"
            )

        if not torch.equal(
            prepared.target.detach().cpu(),
            external_pair.base_image.unsqueeze(
                0
            ).detach().cpu(),
        ):
            raise RuntimeError(
                "Prepared target does not exactly match the external base "
                f"image for case {case.case_id}."
            )

        result = posterior_sample_inference(
            model=model,
            prepared=prepared,
            posterior_samples=posterior_samples,
        )

        if result.posterior_samples != posterior_samples:
            raise RuntimeError(
                "Returned posterior sample count does not match the "
                f"requested value for case {case.case_id}."
            )

        if result.prediction_samples.shape != (
            posterior_samples,
            1,
            1,
            240,
            240,
        ):
            raise RuntimeError(
                "Posterior prediction stack has an unexpected shape for "
                f"case {case.case_id}.\n"
                f"Observed: {tuple(result.prediction_samples.shape)}"
            )

        if not torch.isfinite(
            result.prediction_samples
        ).all():
            raise RuntimeError(
                "Posterior prediction samples contain non-finite values for "
                f"case {case.case_id}."
            )

        prediction_samples_cpu = cpu_tensor(
            result.prediction_samples
        )

        products = compute_posterior_products(
            prediction_samples_cpu
        )

        base_batch_cpu = cpu_tensor(
            external_pair.base_image
        ).unsqueeze(
            0
        ).to(
            dtype=prediction_samples_cpu.dtype,
        )

        mask_batch_cpu = cpu_tensor(
            external_pair.transferred_mask
        ).unsqueeze(
            0
        ).to(
            dtype=prediction_samples_cpu.dtype,
        )

        composite_mean = reconstruct_composite_mean(
            prediction_mean=products.prediction_mean,
            base_image=base_batch_cpu,
            transferred_mask=mask_batch_cpu,
        )

        case_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        posterior_stack_path = (
            case_dir
            / "posterior_samples.pt"
        )

        posterior_mean_path = (
            case_dir
            / "posterior_mean.pt"
        )

        posterior_variance_path = (
            case_dir
            / "posterior_variance.pt"
        )

        posterior_std_path = (
            case_dir
            / "posterior_std.pt"
        )

        composite_mean_path = (
            case_dir
            / "composite_mean.pt"
        )

        metadata_path = (
            case_dir
            / "metadata.json"
        )

        posterior_stack_payload = {
            "evaluation_name": EVALUATION_NAME,
            "case_id": case.case_id,
            "checkpoint": str(
                checkpoint_path
            ),
            "evaluation_seed": seed,
            "case_seed": case_seed,
            "posterior_samples": posterior_samples,
            "resample_diffusion_noise": False,

            "prediction_samples": (
                products.prediction_samples
            ),

            "base_image": cpu_tensor(
                external_pair.base_image
            ),

            "transferred_mask": cpu_tensor(
                external_pair.transferred_mask
            ),

            "known": cpu_tensor(
                external_pair.known
            ),

            "donor_image": cpu_tensor(
                external_pair.donor_image
            ),

            "donor_patch": cpu_tensor(
                external_pair.donor_patch
            ),

            "donor_condition": cpu_tensor(
                external_pair.donor_condition
            ),

            "timestep": cpu_tensor(
                prepared.timestep
            ),

            "diffusion_noise": cpu_tensor(
                prepared.diffusion_noise
            ),

            "x_t": cpu_tensor(
                prepared.x_t
            ),
        }

        torch.save(
            posterior_stack_payload,
            posterior_stack_path,
        )

        torch.save(
            cpu_tensor(
                products.prediction_mean
            ),
            posterior_mean_path,
        )

        torch.save(
            cpu_tensor(
                products.prediction_variance
            ),
            posterior_variance_path,
        )

        torch.save(
            cpu_tensor(
                products.prediction_std
            ),
            posterior_std_path,
        )

        torch.save(
            cpu_tensor(
                composite_mean
            ),
            composite_mean_path,
        )

        metadata = {
            "evaluation_name": EVALUATION_NAME,

            "case_id": case.case_id,

            "external_subject_numeric_id": (
                case.external_subject_numeric_id
            ),

            "external_subject_name": (
                external_pair.external_subject_name
            ),

            "external_slice_index": (
                case.external_slice_index
            ),

            "external_modality": (
                case.external_modality
            ),

            "external_source_path": (
                external_pair.external_source_path
            ),

            "donor_h5_path": str(
                case.donor_h5_path
            ),

            "checkpoint": str(
                checkpoint_path
            ),

            "baseline_config": str(
                args.baseline_config
                .expanduser()
                .resolve()
            ),

            "br_lora_config": str(
                args.br_lora_config
                .expanduser()
                .resolve()
            ),

            "validation_dataset": str(
                validation_dataset_path
            ),

            "evaluation_manifest": str(
                evaluation_manifest_path
            ),

            "evaluation_seed": seed,

            "case_seed": case_seed,

            "posterior_samples": (
                posterior_samples
            ),

            "resample_diffusion_noise": (
                False
            ),

            "donor_image_channel": (
                image_channel
            ),

            "timestep_fraction": (
                timestep_fraction
            ),

            "timestep": int(
                prepared.timestep[
                    0
                ].item()
            ),

            "variational_module_count": len(
                loaded.variational_module_names
            ),

            "variational_parameter_count": (
                loaded.variational_parameter_count
            ),

            "git_commit": git_commit,

            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),

            "artifacts": {
                "posterior_samples": (
                    "posterior_samples.pt"
                ),
                "posterior_mean": (
                    "posterior_mean.pt"
                ),
                "posterior_variance": (
                    "posterior_variance.pt"
                ),
                "posterior_std": (
                    "posterior_std.pt"
                ),
                "composite_mean": (
                    "composite_mean.pt"
                ),
            },
        }

        write_json(
            metadata_path,
            metadata,
        )

        completed_cases += 1

        print(
            "  Case seed              :",
            case_seed,
        )

        print(
            "  External subject       :",
            external_pair.external_subject_name,
        )

        print(
            "  External slice         :",
            case.external_slice_index,
        )

        print(
            "  Donor mask pixels      :",
            int(
                external_pair.transferred_mask.sum().item()
            ),
        )

        print(
            "  Posterior stack        :",
            tuple(
                products.prediction_samples.shape
            ),
        )

        print(
            "  Case output            :",
            case_dir,
        )

        print(
            "  Verdict                : PASS"
        )

    if completed_cases != len(
        cases
    ):
        raise RuntimeError(
            "External evaluation did not complete every manifest case."
        )

    evaluation_completed_utc = datetime.now(
        timezone.utc
    ).isoformat()

    summary_path = (
        output_dir
        / "evaluation_summary.json"
    )

    summary = {
        "evaluation_name": EVALUATION_NAME,
        "checkpoint": str(
            checkpoint_path
        ),
        "validation_dataset": str(
            validation_dataset_path
        ),
        "evaluation_manifest": str(
            evaluation_manifest_path
        ),
        "seed": seed,
        "posterior_samples_per_case": (
            posterior_samples
        ),
        "fixed_diffusion_noise": True,
        "resume_mode": bool(
            args.resume
        ),
        "case_count": len(
            cases
        ),
        "completed_case_count": (
            completed_cases
        ),
        "git_commit": git_commit,
        "evaluation_started_utc": (
            evaluation_started_utc
        ),
        "evaluation_completed_utc": (
            evaluation_completed_utc
        ),
    }

    write_json(
        summary_path,
        summary,
    )

    print()
    print(
        "=" * 78
    )

    print(
        "BR-LoRA EXTERNAL EVALUATION: PASS"
    )

    print(
        "Completed cases          :",
        completed_cases,
    )

    print(
        "Posterior realizations   :",
        posterior_samples,
        "per case",
    )

    print(
        "Evaluation summary       :",
        summary_path,
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    try:
        main()

    except (
        FileNotFoundError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(
            "\nBR-LoRA EXTERNAL EVALUATION FAILED",
            file=sys.stderr,
        )

        print(
            exc,
            file=sys.stderr,
        )

        sys.exit(
            1
        )
