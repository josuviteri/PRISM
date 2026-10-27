"""Checkpoint-backed DDPM and DDIM sampling for the PRISM web application."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn

from src.models.unet import model as build_unet
from src.utils.diffusion_utils import GaussianDiffusion


@dataclass(frozen=True)
class CheckpointInfo:
    """Metadata needed to present an available diffusion checkpoint."""

    identifier: str
    path: Path
    dataset: str
    timesteps: int
    num_classes: int
    p_uncond: float | None
    guidance_scale: float | None


def discover_checkpoints(directory: Path) -> dict[str, CheckpointInfo]:
    """Return valid PRISM checkpoints found directly inside ``directory``."""
    checkpoints: dict[str, CheckpointInfo] = {}
    for path in sorted(directory.glob("*.pt")):
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
            config = state["config"]
            if "ema" not in state:
                raise ValueError("missing EMA weights")
            dataset = config["datos"]["nombre"]
            timesteps = int(config["modelo"]["diffusion_steps"])
            num_classes = int(config["modelo"].get("num_classes", 1))
            if dataset not in {"mnist", "cifar10"}:
                raise ValueError(f"unsupported dataset: {dataset}")
            if num_classes < 1:
                raise ValueError("num_classes must be positive")
        except (KeyError, TypeError, ValueError, RuntimeError):
            continue
        training = config.get("entrenamiento", {})
        checkpoints[path.name] = CheckpointInfo(
            path.name, path, dataset, timesteps, num_classes,
            float(training["p_uncond"]) if "p_uncond" in training else None,
            float(training["guidance_scale"]) if "guidance_scale" in training else None,
        )
    return checkpoints


def _build_model(config: dict[str, Any]) -> nn.Module:
    """Build the unconditional U-Net described by a saved checkpoint config."""
    model_config = config["modelo"]
    channels = 1 if config["datos"]["nombre"] == "mnist" else 3
    return build_unet(
        in_ch=channels,
        out_ch=channels,
        ch=model_config["base_channels"],
        ch_mult=tuple(model_config["channel_multipliers"]),
        num_res_blocks=model_config["residual_blocks"],
        attn_resolutions=tuple(model_config["attention_resolutions"]),
        dropout=model_config["dropout"],
        num_classes=model_config.get("num_classes", 1),
    )


def image_grid(images: Tensor) -> Image.Image:
    """Render normalized BCHW diffusion outputs as a compact RGB grid."""
    images = ((images.detach().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0)
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    count, _, height, width = images.shape
    columns = math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    grid = Image.new("RGB", (columns * width, rows * height))
    for index, image in enumerate(images):
        array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
        grid.paste(Image.fromarray(array, mode="RGB"), ((index % columns) * width, (index // columns) * height))
    return grid


class DiffusionExplorer:
    """Load PRISM artifacts and generate reproducible DDPM or DDIM sample grids."""

    def __init__(self, checkpoints: dict[str, CheckpointInfo], device: torch.device):
        self.checkpoints = checkpoints
        self.device = device
        self._loaded: dict[str, tuple[nn.Module, dict[str, Any]]] = {}

    def generate(
        self,
        identifier: str,
        method: str,
        sampling_steps: int | None,
        num_samples: int,
        seed: int,
        label: int | None = None,
        guidance_scale: float | None = None,
    ) -> tuple[Image.Image, Image.Image, list[tuple[int, int, Image.Image]], CheckpointInfo, int]:
        """Generate image grids and an inspectable reverse-diffusion trajectory."""
        if identifier not in self.checkpoints:
            raise ValueError("unknown checkpoint")
        if method not in {"ddpm", "ddim"}:
            raise ValueError("method must be DDPM or DDIM")
        if not 1 <= num_samples <= 64:
            raise ValueError("number of samples must be between 1 and 64")
        info = self.checkpoints[identifier]
        if info.num_classes > 1:
            if label is None or not 0 <= label < info.num_classes:
                raise ValueError(f"label must be between 0 and {info.num_classes - 1}")
            if guidance_scale is None:
                guidance_scale = info.guidance_scale if info.guidance_scale is not None else 1.0
            if not math.isfinite(guidance_scale) or not 0.0 <= guidance_scale <= 20.0:
                raise ValueError("guidance scale must be between 0 and 20")
        elif label is not None or guidance_scale not in (None, 1.0):
            raise ValueError("class labels and guidance require a conditional checkpoint")
        if method == "ddpm":
            steps = info.timesteps
        else:
            steps = 50 if sampling_steps is None else sampling_steps
            if not 1 <= steps <= info.timesteps:
                raise ValueError(f"DDIM steps must be between 1 and {info.timesteps}")
        model, config = self._model(info)
        channels, size = (1, 28) if info.dataset == "mnist" else (3, 32)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        model_config = config["modelo"]
        sampler = GaussianDiffusion(
            model_config["diffusion_steps"],
            model_config["beta_start"],
            model_config["beta_end"],
            self.device,
        )
        labels = None if label is None else torch.full(
            (num_samples,), label, device=self.device, dtype=torch.long
        )
        noise, samples, trajectory = sampler.sample_with_trajectory(
            model, (num_samples, channels, size, size), method, steps, generator,
            labels=labels, guidance_scale=guidance_scale if guidance_scale is not None else 1.0,
        )
        frames = [(iteration, timestep, image_grid(images)) for iteration, timestep, images in trajectory]
        return image_grid(noise), image_grid(samples), frames, info, steps

    def _model(self, info: CheckpointInfo) -> tuple[nn.Module, dict[str, Any]]:
        """Load and cache the EMA U-Net held in a selected checkpoint."""
        cached = self._loaded.get(info.identifier)
        if cached is not None:
            return cached
        state = torch.load(info.path, map_location="cpu", weights_only=True)
        config = state["config"]
        model = _build_model(config)
        model.load_state_dict(state["ema"])
        model.to(self.device).eval()
        self._loaded[info.identifier] = (model, config)
        return model, config
