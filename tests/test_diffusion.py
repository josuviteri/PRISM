"""Focused tests for diffusion schedules, sampling, and EMA updates."""

import pytest
import torch
from torch import nn

from src.training.diffusion_loop import ExponentialMovingAverage, GaussianDiffusion


class ZeroDenoiser(nn.Module):
    """Small denoiser with the interface expected by the diffusion process."""

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        del timesteps
        return torch.zeros_like(x)


def test_forward_diffusion_matches_schedule_equation():
    diffusion = GaussianDiffusion(4, 0.01, 0.04, torch.device("cpu"))
    clean = torch.ones(2, 1, 3, 3)
    noise = torch.full_like(clean, 2.0)
    steps = torch.tensor([0, 3])
    result = diffusion.q_sample(clean, steps, noise)
    expected = (diffusion.sqrt_alpha_bars[steps, None, None, None] * clean
                + diffusion.sqrt_one_minus_alpha_bars[steps, None, None, None] * noise)
    assert torch.allclose(result, expected)


def test_sampling_is_reproducible_clamped_and_restores_training_mode():
    diffusion = GaussianDiffusion(3, 0.01, 0.03, torch.device("cpu"))
    model = ZeroDenoiser().train()
    first = diffusion.sample(model, (2, 1, 4, 4), torch.Generator().manual_seed(8), amp=False)
    second = diffusion.sample(model, (2, 1, 4, 4), torch.Generator().manual_seed(8), amp=False)
    assert torch.equal(first, second)
    assert first.shape == (2, 1, 4, 4)
    assert first.min() >= -1 and first.max() <= 1
    assert model.training


def test_ddim_sampling_validates_step_count():
    diffusion = GaussianDiffusion(3, 0.01, 0.03, torch.device("cpu"))
    with pytest.raises(ValueError, match="sampling_steps"):
        diffusion.sample_ddim(ZeroDenoiser(), (1, 1, 4, 4), sampling_steps=4)


def test_ema_tracks_parameters_and_copies_buffers():
    model = nn.BatchNorm1d(2)
    ema = ExponentialMovingAverage(model, decay=0.9, warmup=False)
    with torch.no_grad():
        model.weight.fill_(3.0)
        model.running_mean.fill_(2.0)
    ema.update()
    assert torch.allclose(ema.module.weight, torch.full((2,), 1.2))
    assert torch.equal(ema.module.running_mean, model.running_mean)
    ema.copy_to(model)
    assert torch.equal(model.weight, ema.module.weight)


def test_invalid_scheduler_warmup_is_rejected():
    from src.training.diffusion_loop import _warmup_scheduler

    optimizer = torch.optim.SGD([nn.Parameter(torch.ones(()))], lr=0.1)
    with pytest.raises(ValueError, match="non-negative"):
        _warmup_scheduler(optimizer, -1)
