"""
Reusable training utilities for the patch-conditioned x0 diffusion baseline.

Internal-validation behavior reproduces the reference notebook. A separate
full-training path is provided for fixed-epoch refits using 100% of eligible
training slices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.diffusion import DiffusionSchedule
from .losses import masked_x0_loss


def prepare_model_input(
    batch: dict,
    schedule: DiffusionSchedule,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    x0 = batch["x0"].to(device)
    known = batch["known"].to(device)
    mask = batch["mask"].to(device)
    donor_patch = batch["donor_patch"].to(device)
    cond = batch["cond"].to(device)

    batch_size = x0.shape[0]

    t = torch.randint(
        low=0,
        high=schedule.timesteps,
        size=(batch_size,),
        device=device,
        dtype=torch.long,
    )

    noise = torch.randn_like(x0)

    x_t_full = schedule.q_sample(
        x0=x0,
        t=t,
        noise=noise,
    )

    x_t = (
        x0 * (1.0 - mask)
        + x_t_full * mask
    )

    model_input = torch.cat(
        [
            x_t,
            known,
            mask,
            donor_patch,
        ],
        dim=1,
    )

    return model_input, t, cond, x0


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    schedule: DiffusionSchedule,
    device: torch.device,
    outside_loss_weight: float = 0.05,
    description: str | None = None,
) -> float:
    model.train()
    total_loss = 0.0

    iterator = tqdm(
        loader,
        desc=description,
    )

    for batch in iterator:
        model_input, t, cond, x0 = prepare_model_input(
            batch=batch,
            schedule=schedule,
            device=device,
        )

        pred_x0 = model(
            model_input,
            t,
            cond,
        )

        mask = batch["mask"].to(device)

        loss = masked_x0_loss(
            pred_x0=pred_x0,
            x0=x0,
            mask=mask,
            outside_weight=outside_loss_weight,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    device: torch.device,
    outside_loss_weight: float = 0.05,
) -> float:
    model.eval()
    total_loss = 0.0

    for batch in loader:
        model_input, t, cond, x0 = prepare_model_input(
            batch=batch,
            schedule=schedule,
            device=device,
        )

        pred_x0 = model(
            model_input,
            t,
            cond,
        )

        mask = batch["mask"].to(device)

        loss = masked_x0_loss(
            pred_x0=pred_x0,
            x0=x0,
            mask=mask,
            outside_weight=outside_loss_weight,
        )

        total_loss += loss.item()

    return total_loss / len(loader)


def build_checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int | None,
    best_val_loss: float | None,
    timesteps: int,
    base_channels: int,
    image_channel: int,
    min_tumor_pixels: int,
    batch_size: int,
    learning_rate: float,
    cond_dim: int,
    training_mode: str = "patch_conditioned_x0_diffusion",
    final_train_loss: float | None = None,
    final_val_loss: float | None = None,
    epochs: int | None = None,
    split_mode: str | None = None,
) -> dict:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "timesteps": timesteps,
        "base_channels": base_channels,
        "image_channel": image_channel,
        "min_tumor_pixels": min_tumor_pixels,
        "batch_size": batch_size,
        "lr": learning_rate,
        "cond_dim": cond_dim,
        "training_mode": training_mode,
    }

    if epoch is not None:
        payload["epoch"] = epoch

    if epochs is not None:
        payload["epochs"] = epochs

    if best_val_loss is not None:
        payload["best_val_loss"] = best_val_loss

    if final_train_loss is not None:
        payload["final_train_loss"] = final_train_loss

    if final_val_loss is not None:
        payload["final_val_loss"] = final_val_loss

    if split_mode is not None:
        payload["split_mode"] = split_mode

    return payload


def save_checkpoint(
    path: str | Path,
    payload: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    torch.save(
        payload,
        path,
    )


def fit(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    schedule: DiffusionSchedule,
    device: torch.device,
    epochs: int,
    outside_loss_weight: float,
    best_checkpoint_path: str | Path,
    checkpoint_metadata: dict,
    epoch_callback: Callable[[int, float, float], None] | None = None,
) -> tuple[list[float], list[float], float]:
    """
    Notebook-equivalent internal-validation training path.
    """
    if epochs <= 0:
        raise ValueError(
            "epochs must be positive."
        )

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            schedule=schedule,
            device=device,
            outside_loss_weight=outside_loss_weight,
            description=f"Epoch {epoch}/{epochs}",
        )

        val_loss = validate_one_epoch(
            model=model,
            loader=val_loader,
            schedule=schedule,
            device=device,
            outside_loss_weight=outside_loss_weight,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}"
        )

        if epoch_callback is not None:
            epoch_callback(
                epoch,
                train_loss,
                val_loss,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            payload = build_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                **checkpoint_metadata,
            )

            save_checkpoint(
                best_checkpoint_path,
                payload,
            )

            print(
                "Saved best patch-conditioned x0 model."
            )

    return train_losses, val_losses, best_val_loss


def fit_full_train(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    schedule: DiffusionSchedule,
    device: torch.device,
    epochs: int,
    outside_loss_weight: float,
    epoch_callback: Callable[[int, float], None] | None = None,
) -> list[float]:
    """
    Train for a fixed number of epochs using 100% of eligible training slices.

    No validation loss or best-checkpoint rule is computed.
    """
    if epochs <= 0:
        raise ValueError(
            "epochs must be positive."
        )

    train_losses = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            schedule=schedule,
            device=device,
            outside_loss_weight=outside_loss_weight,
            description=f"Epoch {epoch}/{epochs}",
        )

        train_losses.append(train_loss)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}"
        )

        if epoch_callback is not None:
            epoch_callback(
                epoch,
                train_loss,
            )

    return train_losses
