"""
External BR-LoRA pair preparation.

This module converts one fixed external evaluation case into the same tensor
contract consumed by ``prepare_br_lora_batch``.

The external base image is loaded from the registered BraTS 2020 validation
release. The donor image, donor lesion mask, and donor conditioning vector are
loaded from one labeled training H5 slice.

No case discovery, posterior sampling, model inference, or output writing is
performed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data import (
    RegisteredValidationDataset,
    load_h5_full,
    load_validation_slice,
)

from .external_manifest import (
    ExternalEvaluationCase,
)


@dataclass(
    frozen=True,
    slots=True,
)
class PreparedExternalPair:
    """One external BR-LoRA evaluation pair and its retained tensors."""

    case: ExternalEvaluationCase

    batch: dict[
        str,
        torch.Tensor,
    ]

    base_image: torch.Tensor
    transferred_mask: torch.Tensor
    known: torch.Tensor
    donor_image: torch.Tensor
    donor_patch: torch.Tensor
    donor_condition: torch.Tensor

    external_subject_name: str
    external_source_path: str


def prepare_external_pair(
    *,
    case: ExternalEvaluationCase,
    validation_dataset: RegisteredValidationDataset,
    donor_image_channel: int,
) -> PreparedExternalPair:
    """
    Prepare one external validation base / training donor pair for BR-LoRA.

    Parameters
    ----------
    case
        Validated external evaluation-manifest record.
    validation_dataset
        Registered BraTS 2020 validation-dataset specification.
    donor_image_channel
        Training H5 MRI channel used for the donor image. This should match the
        image channel used to train the BR-LoRA model.

    Returns
    -------
    PreparedExternalPair
        Dataset-style batch plus retained case tensors and provenance.
    """

    if not isinstance(
        case,
        ExternalEvaluationCase,
    ):
        raise TypeError(
            "`case` must be an ExternalEvaluationCase."
        )

    if not isinstance(
        validation_dataset,
        RegisteredValidationDataset,
    ):
        raise TypeError(
            "`validation_dataset` must be a RegisteredValidationDataset."
        )

    if (
        isinstance(
            donor_image_channel,
            bool,
        )
        or not isinstance(
            donor_image_channel,
            int,
        )
        or donor_image_channel < 0
    ):
        raise ValueError(
            "donor_image_channel must be a non-negative integer."
        )

    external = load_validation_slice(
        validation_dataset,
        subject_numeric_id=(
            case.external_subject_numeric_id
        ),
        slice_index=(
            case.external_slice_index
        ),
        modality=(
            case.external_modality
        ),
    )

    (
        donor_image,
        donor_mask,
        donor_condition,
    ) = load_h5_full(
        case.donor_h5_path,
        image_channel=donor_image_channel,
    )

    base_image = external.image

    if base_image.ndim != 3:
        raise RuntimeError(
            "External base image must have shape [1, H, W]."
        )

    if donor_image.shape != base_image.shape:
        raise RuntimeError(
            "External base and donor image shapes do not match.\n\n"
            f"External base: {tuple(base_image.shape)}\n"
            f"Donor image:   {tuple(donor_image.shape)}"
        )

    if donor_mask.shape != base_image.shape:
        raise RuntimeError(
            "External base and transferred-mask shapes do not match.\n\n"
            f"External base:    {tuple(base_image.shape)}\n"
            f"Transferred mask: {tuple(donor_mask.shape)}"
        )

    if donor_condition.ndim != 1:
        raise RuntimeError(
            "Donor conditioning vector must be one-dimensional."
        )

    if not torch.isfinite(
        base_image
    ).all():
        raise RuntimeError(
            "External base image contains non-finite values."
        )

    if not torch.isfinite(
        donor_image
    ).all():
        raise RuntimeError(
            "Donor image contains non-finite values."
        )

    if not torch.isfinite(
        donor_mask
    ).all():
        raise RuntimeError(
            "Transferred donor mask contains non-finite values."
        )

    if not torch.isfinite(
        donor_condition
    ).all():
        raise RuntimeError(
            "Donor conditioning vector contains non-finite values."
        )

    if not (
        donor_mask > 0
    ).any():
        raise RuntimeError(
            "The donor H5 slice contains an empty whole-tumor mask."
        )

    donor_patch = (
        donor_image
        * donor_mask
    )

    known = (
        base_image
        * (
            1.0
            - donor_mask
        )
    )

    batch = {
        "x0": base_image.unsqueeze(
            0
        ),
        "known": known.unsqueeze(
            0
        ),
        "mask": donor_mask.unsqueeze(
            0
        ),
        "donor_patch": donor_patch.unsqueeze(
            0
        ),
        "cond": donor_condition.unsqueeze(
            0
        ),
    }

    expected_image_shape = (
        1,
        1,
        240,
        240,
    )

    for name in (
        "x0",
        "known",
        "mask",
        "donor_patch",
    ):
        if tuple(
            batch[
                name
            ].shape
        ) != expected_image_shape:
            raise RuntimeError(
                f"External BR-LoRA batch tensor {name!r} has an "
                "unexpected shape.\n\n"
                f"Observed: {tuple(batch[name].shape)}\n"
                f"Expected: {expected_image_shape}"
            )

    if tuple(
        batch[
            "cond"
        ].shape
    ) != (
        1,
        4,
    ):
        raise RuntimeError(
            "External BR-LoRA conditioning batch must have shape [1, 4].\n\n"
            f"Observed: {tuple(batch['cond'].shape)}"
        )

    return PreparedExternalPair(
        case=case,
        batch=batch,
        base_image=base_image,
        transferred_mask=donor_mask,
        known=known,
        donor_image=donor_image,
        donor_patch=donor_patch,
        donor_condition=donor_condition,
        external_subject_name=external.subject_name,
        external_source_path=str(
            external.source_path
        ),
    )


__all__ = [
    "PreparedExternalPair",
    "prepare_external_pair",
]
