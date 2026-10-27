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


@dataclass(frozen=True)
class CheckpointInfo:
    """Metadata needed to present an available diffusion checkpoint."""

    identifier: str
    path: Path
    dataset: str
    timesteps: int


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
            if dataset not in {"mnist", "cifar10"}:
                raise ValueError(f"unsupported dataset: {dataset}")
        except (KeyError, TypeError, ValueError, RuntimeError):
            continue
        checkpoints[path.name] = CheckpointInfo(path.name, path, dataset, timesteps)
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


class _Sampler:
    """Reverse diffusion schedule used by the inference-only application."""

    def __init__(self, config: dict[str, Any], device: torch.device):
        model_config = config["modelo"]
        self.timesteps = int(model_config["diffusion_steps"])
        betas = torch.linspace(
            model_config["beta_start"], model_config["beta_end"], self.timesteps, dtype=torch.float64
        )
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bars[:-1]])
        self.betas = betas.to(device=device, dtype=torch.float32)
        self.alpha_bars = alpha_bars.to(device=device, dtype=torch.float32)
        self.sqrt_recip_alpha_bars = alpha_bars.rsqrt().to(device=device, dtype=torch.float32)
        self.sqrt_recipm1_alpha_bars = (1.0 / alpha_bars - 1.0).sqrt().to(device=device, dtype=torch.float32)
        self.posterior_mean_coef1 = (betas * alpha_bars_prev.sqrt() / (1.0 - alpha_bars)).to(
            device=device, dtype=torch.float32
        )
        self.posterior_mean_coef2 = ((1.0 - alpha_bars_prev) * alphas.sqrt() / (1.0 - alpha_bars)).to(
            device=device, dtype=torch.float32
        )

    @torch.no_grad()
    def sample(
        self,
        denoiser: nn.Module,
        shape: tuple[int, ...],
        method: str,
        sampling_steps: int,
        generator: torch.Generator,
        max_frames: int = 16,
    ) -> tuple[Tensor, Tensor, list[tuple[int, int, Tensor]]]:
        """Generate an image batch together with evenly-spaced reverse-process snapshots."""
        device = self.betas.device
        sample = torch.randn(shape, device=device, generator=generator)
        initial_noise = sample.detach().clone()
        if method == "ddpm":
            schedule = torch.arange(self.timesteps - 1, -1, -1, device=device)
        else:
            schedule = torch.linspace(0, self.timesteps - 1, sampling_steps, device=device).round().long().flip(0)
        frame_positions = {
            int(position)
            for position in torch.linspace(0, len(schedule), min(max_frames, len(schedule) + 1)).round().tolist()
        }
        trajectory = [(0, int(schedule[0]), initial_noise.detach().clone())]

        denoiser.eval()
        for position, step in enumerate(schedule):
            timestep = torch.full((shape[0],), step, device=device, dtype=torch.long)
            predicted_noise = denoiser(sample, timestep).float()
            x0 = (
                self.sqrt_recip_alpha_bars[step] * sample
                - self.sqrt_recipm1_alpha_bars[step] * predicted_noise
            ).clamp(-1.0, 1.0)
            if method == "ddpm":
                mean = self.posterior_mean_coef1[step] * x0 + self.posterior_mean_coef2[step] * sample
                sample = mean if step == 0 else mean + self.betas[step].sqrt() * torch.randn(
                    shape, device=device, generator=generator
                )
            else:
                previous_step = schedule[position + 1] if position + 1 < len(schedule) else -1
                alpha_bar_previous = self.alpha_bars[previous_step] if previous_step > 0 else torch.ones((), device=device)
                sample = alpha_bar_previous.sqrt() * x0 + (1.0 - alpha_bar_previous).sqrt() * predicted_noise
            if position + 1 in frame_positions:
                state_timestep = int(schedule[position + 1]) if position + 1 < len(schedule) else 0
                trajectory.append((position + 1, state_timestep, sample.detach().clone()))
        return initial_noise, sample.clamp(-1.0, 1.0), trajectory


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
    ) -> tuple[Image.Image, Image.Image, list[tuple[int, int, Image.Image]], CheckpointInfo, int]:
        """Generate image grids and an inspectable reverse-diffusion trajectory."""
        if identifier not in self.checkpoints:
            raise ValueError("unknown checkpoint")
        if method not in {"ddpm", "ddim"}:
            raise ValueError("method must be DDPM or DDIM")
        if not 1 <= num_samples <= 64:
            raise ValueError("number of samples must be between 1 and 64")
        info = self.checkpoints[identifier]
        if method == "ddpm":
            steps = info.timesteps
        else:
            steps = 50 if sampling_steps is None else sampling_steps
            if not 1 <= steps <= info.timesteps:
                raise ValueError(f"DDIM steps must be between 1 and {info.timesteps}")
        model, config = self._model(info)
        channels, size = (1, 28) if info.dataset == "mnist" else (3, 32)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        sampler = _Sampler(config, self.device)
        noise, samples, trajectory = sampler.sample(model, (num_samples, channels, size, size), method, steps, generator)
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
