#!/usr/bin/env python3

"""
Evaluate one fitted deterministic PEFT comparator on a fixed external cohort.

Supported methods:
- Regional LoRA
- DoRA
- LoKr
- BitFit

The evaluator reuses the accepted BR-LoRA external-case construction and,
case by case, reuses the exact stored BR-LoRA diffusion-noise realization.
This holds fixed:

- external base image,
- donor H5 slice,
- transferred tumor mask,
- donor conditioning,
- diffusion timestep, and
- diffusion-noise tensor.

The only intended model-level difference is the fitted PEFT adaptation.

One deterministic prediction is written per case. Downstream feathering is not
performed here; the downstream segmentation loader applies the same inner-only
distance feathering used for the BR-LoRA posterior-mean comparison.
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

# Preserve the repository's NumPy-before-PyTorch import convention.
import numpy as np
import torch
import yaml

from src.data import (
    load_validation_dataset_specification,
)
from src.diffusion import DiffusionSchedule
from src.inference import (
    adaptation_inference,
    load_external_evaluation_manifest,
    load_fitted_adaptation,
    prepare_adaptation_batch,
    prepare_external_pair,
)


EVALUATION_NAME = "deterministic_adaptation_external_evaluation"

REQUIRED_REFERENCE_KEYS = {
    "base_image",
    "transferred_mask",
    "known",
    "donor_image",
    "donor_patch",
    "donor_condition",
    "timestep",
    "diffusion_noise",
    "x_t",
}

REQUIRED_CASE_ARTIFACTS = (
    "prediction.pt",
    "synthesis_payload.pt",
    "metadata.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a fitted deterministic PEFT model on a fixed "
            "external BraTS case manifest."
        )
    )

    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=Path(
            "configs/baseline_patch_x0_full_train.yaml"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--validation-dataset",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--reference-br-lora-batch-root",
        type=Path,
        required=True,
        help=(
            "Accepted BR-LoRA batch directory containing the matching "
            "case directories and posterior_samples.pt artifacts."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
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
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    return parser.parse_args()


def load_config(
    path: Path,
    *,
    name: str,
) -> dict:
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
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def derive_case_seed(
    *,
    evaluation_seed: int,
    case_id: str,
) -> int:
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

    return (
        commit
        if commit
        else None
    )


def cpu_tensor(
    value: torch.Tensor,
) -> torch.Tensor:
    return (
        value
        .detach()
        .cpu()
    )


def write_json(
    path: Path,
    payload: dict,
) -> None:
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


def load_reference_payload(
    *,
    reference_batch_root: Path,
    case_id: str,
) -> tuple[Path, dict]:
    path = (
        reference_batch_root
        / case_id
        / "posterior_samples.pt"
    )

    if not path.is_file():
        raise FileNotFoundError(
            "Accepted BR-LoRA reference payload not found:\n"
            f"{path}"
        )

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            "BR-LoRA reference payload must contain a dictionary:\n"
            f"{path}"
        )

    missing = sorted(
        REQUIRED_REFERENCE_KEYS
        - set(
            payload
        )
    )

    if missing:
        raise KeyError(
            "BR-LoRA reference payload is missing required keys:\n"
            + "\n".join(
                missing
            )
            + f"\nPayload: {path}"
        )

    return (
        path,
        payload,
    )


def require_exact_tensor(
    *,
    observed: torch.Tensor,
    expected: torch.Tensor,
    name: str,
    case_id: str,
) -> None:
    observed_cpu = cpu_tensor(
        observed
    )

    expected_cpu = cpu_tensor(
        expected
    )

    if not torch.equal(
        observed_cpu,
        expected_cpu,
    ):
        raise RuntimeError(
            f"{case_id}: {name} does not exactly match the "
            "accepted BR-LoRA reference artifact."
        )


def validate_reference_pair(
    *,
    case_id: str,
    external_pair,
    reference: dict,
) -> None:
    comparisons = (
        (
            "base_image",
            external_pair.base_image,
        ),
        (
            "transferred_mask",
            external_pair.transferred_mask,
        ),
        (
            "known",
            external_pair.known,
        ),
        (
            "donor_image",
            external_pair.donor_image,
        ),
        (
            "donor_patch",
            external_pair.donor_patch,
        ),
        (
            "donor_condition",
            external_pair.donor_condition,
        ),
    )

    for name, observed in comparisons:
        expected = reference[
            name
        ]

        if not isinstance(
            expected,
            torch.Tensor,
        ):
            raise TypeError(
                f"{case_id}: reference {name!r} is not a tensor."
            )

        require_exact_tensor(
            observed=observed,
            expected=expected,
            name=name,
            case_id=case_id,
        )


def validate_prepared_state(
    *,
    case_id: str,
    prepared,
    reference: dict,
) -> float:
    reference_timestep = reference[
        "timestep"
    ]

    reference_noise = reference[
        "diffusion_noise"
    ]

    reference_x_t = reference[
        "x_t"
    ]

    for name, value in (
        (
            "timestep",
            reference_timestep,
        ),
        (
            "diffusion_noise",
            reference_noise,
        ),
        (
            "x_t",
            reference_x_t,
        ),
    ):
        if not isinstance(
            value,
            torch.Tensor,
        ):
            raise TypeError(
                f"{case_id}: reference {name!r} is not a tensor."
            )

    require_exact_tensor(
        observed=prepared.timestep,
        expected=reference_timestep,
        name="timestep",
        case_id=case_id,
    )

    require_exact_tensor(
        observed=prepared.diffusion_noise,
        expected=reference_noise,
        name="diffusion_noise",
        case_id=case_id,
    )

    observed_x_t = cpu_tensor(
        prepared.x_t
    )

    expected_x_t = cpu_tensor(
        reference_x_t
    )

    if (
        observed_x_t.shape
        != expected_x_t.shape
    ):
        raise RuntimeError(
            f"{case_id}: reconstructed x_t shape does not match "
            "the BR-LoRA reference."
        )

    max_abs_diff = float(
        (
            observed_x_t
            - expected_x_t
        )
        .abs()
        .max()
        .item()
    )

    if not torch.allclose(
        observed_x_t,
        expected_x_t,
        rtol=1.0e-6,
        atol=1.0e-6,
    ):
        raise RuntimeError(
            f"{case_id}: reconstructed x_t does not numerically match "
            "the accepted BR-LoRA reference.\n"
            f"Maximum absolute difference: {max_abs_diff:.9g}"
        )

    return max_abs_diff


def validate_completed_case_directory(
    *,
    case_dir: Path,
    case_id: str,
    method: str,
    checkpoint_path: Path,
    evaluation_manifest_path: Path,
    reference_batch_root: Path,
    evaluation_seed: int,
) -> float:
    missing = [
        name
        for name in REQUIRED_CASE_ARTIFACTS
        if not (
            case_dir
            / name
        ).is_file()
    ]

    if missing:
        raise RuntimeError(
            "Existing deterministic case directory is incomplete and "
            "cannot be resumed safely.\n"
            f"Case: {case_id}\n"
            f"Directory: {case_dir}\n"
            "Missing:\n"
            + "\n".join(
                f"  {name}"
                for name in missing
            )
        )

    metadata_path = (
        case_dir
        / "metadata.json"
    )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(
            file
        )

    expected = {
        "evaluation_name":
            EVALUATION_NAME,
        "case_id":
            case_id,
        "adaptation_method":
            method,
        "checkpoint":
            str(
                checkpoint_path
            ),
        "evaluation_manifest":
            str(
                evaluation_manifest_path
            ),
        "reference_br_lora_batch_root":
            str(
                reference_batch_root
            ),
        "evaluation_seed":
            evaluation_seed,
    }

    mismatches = []

    for key, expected_value in expected.items():
        observed = metadata.get(
            key
        )

        if observed != expected_value:
            mismatches.append(
                f"{key}: expected {expected_value!r}, "
                f"observed {observed!r}"
            )

    if mismatches:
        raise RuntimeError(
            "Existing deterministic case does not match the current "
            "evaluation contract and cannot be resumed.\n"
            f"Case: {case_id}\n"
            + "\n".join(
                mismatches
            )
        )

    stored_x_t_difference = metadata.get(
        "reference_x_t_max_abs_difference"
    )

    if not isinstance(
        stored_x_t_difference,
        (
            int,
            float,
        ),
    ):
        raise RuntimeError(
            "Existing deterministic case metadata is missing a valid "
            "reference_x_t_max_abs_difference and cannot be resumed.\n"
            f"Case: {case_id}\n"
            f"Metadata: {metadata_path}"
        )

    return float(
        stored_x_t_difference
    )


def main() -> None:
    args = parse_args()

    baseline_config = load_config(
        args.baseline_config,
        name="Baseline configuration",
    )

    data_cfg = baseline_config[
        "data"
    ]

    diffusion_cfg = baseline_config[
        "diffusion"
    ]

    inference_cfg = baseline_config.get(
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
        inference_cfg,
        dict,
    ):
        raise ValueError(
            "Baseline inference configuration must be a mapping."
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
        inference_cfg.get(
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

    reference_batch_root = (
        args.reference_br_lora_batch_root
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    for path, name in (
        (
            checkpoint_path,
            "Deterministic PEFT checkpoint",
        ),
        (
            validation_dataset_path,
            "Validation dataset specification",
        ),
        (
            evaluation_manifest_path,
            "External evaluation manifest",
        ),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"{name} not found:\n{path}"
            )

    if not reference_batch_root.is_dir():
        raise FileNotFoundError(
            "Accepted BR-LoRA reference batch root not found:\n"
            f"{reference_batch_root}"
        )

    validation_dataset = (
        load_validation_dataset_specification(
            validation_dataset_path
        )
    )

    cases = load_external_evaluation_manifest(
        evaluation_manifest_path
    )

    loaded = load_fitted_adaptation(
        checkpoint_path,
        device=device,
    )

    model = loaded.model
    method = loaded.adaptation_method

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
    print("=" * 78)
    print("DETERMINISTIC PEFT EXTERNAL EVALUATION")
    print("=" * 78)
    print("Method                   :", method)
    print("Checkpoint               :", checkpoint_path)
    print("Device                   :", device)
    print("Seed                     :", seed)
    print("Cases                    :", len(cases))
    print("Timestep fraction        :", timestep_fraction)
    print("Reference BR-LoRA batch  :", reference_batch_root)
    print("Output                   :", output_dir)
    print("Resume                   :", args.resume)
    print(
        "Git commit               :",
        git_commit if git_commit is not None else "unavailable",
    )
    print("=" * 78)

    evaluation_started_utc = datetime.now(
        timezone.utc
    ).isoformat()

    completed_cases = 0
    maximum_x_t_abs_difference = 0.0

    for case_index, case in enumerate(
        cases,
        start=1,
    ):
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
                    "Deterministic case output already exists. "
                    "Refusing to overwrite it.\n"
                    f"Case: {case.case_id}\n"
                    f"Directory: {case_dir}"
                )

            resumed_x_t_difference = validate_completed_case_directory(
                case_dir=case_dir,
                case_id=case.case_id,
                method=method,
                checkpoint_path=checkpoint_path,
                evaluation_manifest_path=evaluation_manifest_path,
                reference_batch_root=reference_batch_root,
                evaluation_seed=seed,
            )

            maximum_x_t_abs_difference = max(
                maximum_x_t_abs_difference,
                resumed_x_t_difference,
            )

            completed_cases += 1
            print("  Verdict                 : SKIPPED (resume)")
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

        (
            reference_payload_path,
            reference,
        ) = load_reference_payload(
            reference_batch_root=reference_batch_root,
            case_id=case.case_id,
        )

        validate_reference_pair(
            case_id=case.case_id,
            external_pair=external_pair,
            reference=reference,
        )

        reference_noise = reference[
            "diffusion_noise"
        ]

        prepared = prepare_adaptation_batch(
            external_pair.batch,
            schedule=schedule,
            device=device,
            timestep_fraction=timestep_fraction,
            diffusion_noise=reference_noise,
        )

        max_abs_diff = validate_prepared_state(
            case_id=case.case_id,
            prepared=prepared,
            reference=reference,
        )

        maximum_x_t_abs_difference = max(
            maximum_x_t_abs_difference,
            max_abs_diff,
        )

        with torch.inference_mode():
            result = adaptation_inference(
                model=model,
                prepared=prepared,
            )

        prediction = cpu_tensor(
            result.prediction
        )

        if prediction.shape != (
            1,
            1,
            240,
            240,
        ):
            raise RuntimeError(
                f"{case.case_id}: deterministic prediction has "
                f"unexpected shape {tuple(prediction.shape)}."
            )

        if not torch.isfinite(
            prediction
        ).all():
            raise RuntimeError(
                f"{case.case_id}: deterministic prediction contains "
                "non-finite values."
            )

        prediction_case = prediction.squeeze(
            0
        )

        case_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        prediction_path = (
            case_dir
            / "prediction.pt"
        )

        payload_path = (
            case_dir
            / "synthesis_payload.pt"
        )

        metadata_path = (
            case_dir
            / "metadata.json"
        )

        torch.save(
            prediction_case,
            prediction_path,
        )

        payload = {
            "evaluation_name":
                EVALUATION_NAME,
            "case_id":
                case.case_id,
            "adaptation_method":
                method,
            "checkpoint":
                str(
                    checkpoint_path
                ),
            "evaluation_seed":
                seed,
            "case_seed":
                case_seed,
            "reference_br_lora_payload":
                str(
                    reference_payload_path
                ),
            "prediction":
                prediction_case,
            "base_image":
                cpu_tensor(
                    external_pair.base_image
                ),
            "transferred_mask":
                cpu_tensor(
                    external_pair.transferred_mask
                ),
            "known":
                cpu_tensor(
                    external_pair.known
                ),
            "donor_image":
                cpu_tensor(
                    external_pair.donor_image
                ),
            "donor_patch":
                cpu_tensor(
                    external_pair.donor_patch
                ),
            "donor_condition":
                cpu_tensor(
                    external_pair.donor_condition
                ),
            "timestep":
                cpu_tensor(
                    prepared.timestep
                ),
            "diffusion_noise":
                cpu_tensor(
                    prepared.diffusion_noise
                ),
            "x_t":
                cpu_tensor(
                    prepared.x_t
                ),
            "reference_x_t_max_abs_difference":
                max_abs_diff,
            "trainable_parameter_count":
                loaded.trainable_parameter_count,
            "trainable_tensor_count":
                loaded.trainable_tensor_count,
        }

        torch.save(
            payload,
            payload_path,
        )

        metadata = {
            "evaluation_name":
                EVALUATION_NAME,
            "case_id":
                case.case_id,
            "adaptation_method":
                method,
            "checkpoint":
                str(
                    checkpoint_path
                ),
            "evaluation_manifest":
                str(
                    evaluation_manifest_path
                ),
            "validation_dataset":
                str(
                    validation_dataset_path
                ),
            "reference_br_lora_batch_root":
                str(
                    reference_batch_root
                ),
            "reference_br_lora_payload":
                str(
                    reference_payload_path
                ),
            "evaluation_seed":
                seed,
            "case_seed":
                case_seed,
            "external_subject_numeric_id":
                case.external_subject_numeric_id,
            "external_subject_name":
                external_pair.external_subject_name,
            "external_slice_index":
                case.external_slice_index,
            "external_modality":
                case.external_modality,
            "donor_h5_path":
                str(
                    case.donor_h5_path
                ),
            "timestep":
                int(
                    prepared.timestep[
                        0
                    ].item()
                ),
            "fixed_reference_diffusion_noise":
                True,
            "reference_x_t_max_abs_difference":
                max_abs_diff,
            "prediction_artifact":
                "prediction.pt",
            "synthesis_payload_artifact":
                "synthesis_payload.pt",
            "git_commit":
                git_commit,
        }

        write_json(
            metadata_path,
            metadata,
        )

        completed_cases += 1

        print(
            "  x_t max abs difference :",
            f"{max_abs_diff:.9g}",
        )
        print(
            "  Verdict                 : PASS"
        )

    summary = {
        "evaluation_name":
            EVALUATION_NAME,
        "adaptation_method":
            method,
        "checkpoint":
            str(
                checkpoint_path
            ),
        "evaluation_manifest":
            str(
                evaluation_manifest_path
            ),
        "validation_dataset":
            str(
                validation_dataset_path
            ),
        "reference_br_lora_batch_root":
            str(
                reference_batch_root
            ),
        "evaluation_seed":
            seed,
        "cases":
            len(
                cases
            ),
        "completed_cases":
            completed_cases,
        "fixed_reference_diffusion_noise":
            True,
        "timestep_fraction":
            timestep_fraction,
        "maximum_reference_x_t_abs_difference":
            maximum_x_t_abs_difference,
        "started_utc":
            evaluation_started_utc,
        "completed_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "git_commit":
            git_commit,
    }

    write_json(
        output_dir
        / "evaluation_summary.json",
        summary,
    )

    print()
    print("=" * 78)
    print("DETERMINISTIC PEFT EXTERNAL EVALUATION COMPLETE")
    print("=" * 78)
    print("Method                   :", method)
    print("Completed cases          :", completed_cases)
    print(
        "Maximum x_t abs diff     :",
        f"{maximum_x_t_abs_difference:.9g}",
    )
    print("Output                   :", output_dir)


if __name__ == "__main__":
    main()
