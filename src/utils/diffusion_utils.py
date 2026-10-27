"""Shared DDPM and DDIM diffusion calculations."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _extract(values: Tensor, timesteps: Tensor, shape: tuple[int, ...]) -> Tensor:
    return values.gather(0, timesteps).reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


class GaussianDiffusion:
    """Forward and reverse-process calculations for epsilon prediction."""

    def __init__(self, timesteps: int, beta_start: float, beta_end: float, device: torch.device):
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bars[:-1]])

        def to_device(values: Tensor) -> Tensor:
            return values.to(device=device, dtype=torch.float32)

        self.timesteps = timesteps
        self.device = device
        self.betas = to_device(betas)
        self.alphas = to_device(alphas)
        self.alpha_bars = to_device(alpha_bars)
        self.alpha_bars_prev = to_device(alpha_bars_prev)
        self.sqrt_alpha_bars = to_device(alpha_bars.sqrt())
        self.sqrt_one_minus_alpha_bars = to_device((1.0 - alpha_bars).sqrt())
        self.sqrt_recip_alpha_bars = to_device(alpha_bars.rsqrt())
        self.sqrt_recipm1_alpha_bars = to_device((1.0 / alpha_bars - 1.0).sqrt())
        self.posterior_mean_coef1 = to_device(betas * alpha_bars_prev.sqrt() / (1.0 - alpha_bars))
        self.posterior_mean_coef2 = to_device(
            (1.0 - alpha_bars_prev) * alphas.sqrt() / (1.0 - alpha_bars)
        )

    def q_sample(self, x_start: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        shape = tuple(x_start.shape)
        return (
            _extract(self.sqrt_alpha_bars, timesteps, shape) * x_start
            + _extract(self.sqrt_one_minus_alpha_bars, timesteps, shape) * noise
        )

    def loss(
        self,
        denoiser: nn.Module,
        x_start: Tensor,
        generator: torch.Generator | None = None,
        labels: Tensor | None = None,
        p_uncond: float = 0.0,
    ) -> Tensor:
        """Epsilon-prediction MSE with optional classifier-free label dropout."""
        if not 0.0 <= p_uncond <= 1.0:
            raise ValueError("p_uncond must be between 0 and 1")
        device = x_start.device
        timesteps = torch.randint(0, self.timesteps, (x_start.shape[0],), device=device, generator=generator)
        noise = torch.randn(x_start.shape, device=device, dtype=x_start.dtype, generator=generator)
        noisy = self.q_sample(x_start, timesteps, noise)
        if labels is not None:
            labels = labels.to(device=device, dtype=torch.long)
            if labels.ndim != 1 or labels.shape[0] != x_start.shape[0]:
                raise ValueError("labels must have shape (B,)")
            num_classes = getattr(denoiser, "num_classes", 1)
            if num_classes <= 1:
                raise ValueError("labels require a class-conditional denoiser")
            if torch.any((labels < 0) | (labels >= num_classes)):
                raise ValueError(f"labels must be in [0, {num_classes - 1}]")
            if p_uncond:
                drop = torch.rand(labels.shape, device=device, generator=generator) < p_uncond
                labels = labels.masked_fill(drop, num_classes)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = denoiser(noisy, timesteps, labels) if labels is not None else denoiser(noisy, timesteps)
        return torch.nn.functional.mse_loss(prediction.float(), noise)

    @staticmethod
    def _predict_noise(
        denoiser: nn.Module,
        sample: Tensor,
        timesteps: Tensor,
        labels: Tensor | None,
        guidance_scale: float,
    ) -> Tensor:
        if labels is None:
            if guidance_scale != 1.0:
                raise ValueError("guidance_scale requires class labels")
            return denoiser(sample, timesteps)
        num_classes = getattr(denoiser, "num_classes", 1)
        if num_classes <= 1:
            raise ValueError("labels require a class-conditional denoiser")
        labels = labels.to(device=sample.device, dtype=torch.long)
        if labels.ndim != 1 or labels.shape[0] != sample.shape[0]:
            raise ValueError("labels must have shape (B,)")
        if torch.any((labels < 0) | (labels >= num_classes)):
            raise ValueError(f"labels must be in [0, {num_classes - 1}]")
        conditional = denoiser(sample, timesteps, labels)
        if guidance_scale == 1.0:
            return conditional
        unconditional = denoiser(sample, timesteps, torch.full_like(labels, num_classes))
        return unconditional + guidance_scale * (conditional - unconditional)

    def _sample_schedule(self, method: str, sampling_steps: int) -> Tensor:
        if method == "ddpm":
            return torch.arange(self.timesteps - 1, -1, -1, device=self.device)
        if method != "ddim":
            raise ValueError("method must be DDPM or DDIM")
        if not 1 <= sampling_steps <= self.timesteps:
            raise ValueError("sampling_steps must be between 1 and the training timestep count")
        return torch.linspace(0, self.timesteps - 1, sampling_steps, device=self.device).round().long().flip(0)

    @torch.no_grad()
    def sample_with_trajectory(
        self,
        denoiser: nn.Module,
        shape: tuple[int, ...],
        method: str,
        sampling_steps: int,
        generator: torch.Generator | None = None,
        max_frames: int = 16,
        amp: bool = True,
        labels: Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> tuple[Tensor, Tensor, list[tuple[int, int, Tensor]]]:
        """Sample and return the initial noise, final sample, and reverse trajectory."""
        schedule = self._sample_schedule(method, sampling_steps)
        sample = torch.randn(shape, device=self.device, generator=generator)
        initial_noise = sample.detach().clone()
        frame_positions = set()
        if max_frames > 0:
            frame_positions = {
                int(position)
                for position in torch.linspace(0, len(schedule), min(max_frames, len(schedule) + 1)).round().tolist()
            }
        trajectory = [(0, int(schedule[0]), initial_noise.detach().clone())]
        was_training = denoiser.training
        denoiser.eval()
        use_amp = amp and self.device.type == "cuda"
        for position, step in enumerate(schedule):
            timesteps = torch.full((shape[0],), step, device=self.device, dtype=torch.long)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                predicted_noise = self._predict_noise(denoiser, sample, timesteps, labels, guidance_scale)
            predicted_noise = predicted_noise.float()
            x0 = (
                self.sqrt_recip_alpha_bars[step] * sample
                - self.sqrt_recipm1_alpha_bars[step] * predicted_noise
            ).clamp(-1.0, 1.0)
            if method == "ddpm":
                mean = self.posterior_mean_coef1[step] * x0 + self.posterior_mean_coef2[step] * sample
                sample = mean if step == 0 else mean + self.betas[step].sqrt() * torch.randn(
                    shape, device=self.device, generator=generator
                )
            else:
                previous_step = schedule[position + 1] if position + 1 < len(schedule) else -1
                alpha_bar_previous = self.alpha_bars[previous_step] if previous_step > 0 else torch.ones((), device=self.device)
                sample = alpha_bar_previous.sqrt() * x0 + (1.0 - alpha_bar_previous).sqrt() * predicted_noise
            if position + 1 in frame_positions:
                state_timestep = int(schedule[position + 1]) if position + 1 < len(schedule) else 0
                trajectory.append((position + 1, state_timestep, sample.detach().clone()))
        if was_training:
            denoiser.train()
        return initial_noise, sample.clamp(-1.0, 1.0), trajectory

    @torch.no_grad()
    def sample(
        self,
        denoiser: nn.Module,
        shape: tuple[int, ...],
        generator: torch.Generator | None = None,
        amp: bool = True,
        labels: Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> Tensor:
        """Ancestral DDPM sampling with x0 clipping at every step."""
        return self.sample_with_trajectory(
            denoiser, shape, "ddpm", self.timesteps, generator, 0, amp, labels, guidance_scale
        )[1]

    @torch.no_grad()
    def sample_ddim(
        self,
        denoiser: nn.Module,
        shape: tuple[int, ...],
        sampling_steps: int = 50,
        generator: torch.Generator | None = None,
        amp: bool = True,
        labels: Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> Tensor:
        """Deterministic DDIM sampling on a subsampled training schedule."""
        return self.sample_with_trajectory(
            denoiser, shape, "ddim", sampling_steps, generator, 0, amp, labels, guidance_scale
        )[1]