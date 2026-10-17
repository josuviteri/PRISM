"""Shared checkpoint and sampling helpers for the command-line scripts."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loaders import construir_test_loader  # noqa: E402
from src.training.cifar_loop import (  # noqa: E402
    GaussianDiffusion,
    _save_image_grid,
    construir_modelo,
)


def cargar_modelo(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Load the EMA model and its embedded training configuration."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "config" not in checkpoint or "ema" not in checkpoint:
        raise ValueError("checkpoint must contain 'config' and 'ema' entries")
    cfg = checkpoint["config"]
    model = construir_modelo(cfg)
    model.load_state_dict(checkpoint["ema"])
    model.to(device).eval()
    return model, cfg


def construir_difusion(cfg: dict[str, Any], device: torch.device) -> GaussianDiffusion:
    """Build the sampler using the schedule stored in the checkpoint config."""
    model_cfg = cfg["modelo"]
    return GaussianDiffusion(
        model_cfg["diffusion_steps"],
        model_cfg["beta_start"],
        model_cfg["beta_end"],
        device,
    )


def construir_test_loader_con_batch(cfg: dict[str, Any], batch_size: int) -> DataLoader:
    """Build the configured test loader with a command-line batch size."""
    loader_cfg = copy.deepcopy(cfg)
    loader_cfg["datos"]["batch_size"] = batch_size
    return construir_test_loader(loader_cfg)


def obtener_normalizacion(loader) -> tuple[Tensor, Tensor]:
    """Read the exact train-only normalization attached to the test dataset."""
    transform = getattr(loader.dataset, "transform", None)
    transforms = getattr(transform, "transforms", ())
    for operation in transforms:
        if hasattr(operation, "mean") and hasattr(operation, "std"):
            return operation.mean.flatten(), operation.std.flatten()
    raise ValueError("could not find normalization statistics on the test dataset")


def a_rgb(images: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """Convert normalized BCHW tensors to clipped float RGB tensors in [0, 1]."""
    return (images.detach().cpu() * std[:, None, None] + mean[:, None, None]).clamp(0.0, 1.0)


def guardar_pngs(images: Tensor, output_dir: Path, start: int) -> None:
    """Save a BCHW float tensor as numbered PNG files."""
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    for offset, image in enumerate(images):
        array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
        Image.fromarray(array, mode="RGB").save(output_dir / f"sample_{start + offset:05d}.png")
