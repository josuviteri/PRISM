"""Regression tests for checkpoint-only diffusion application services."""

from __future__ import annotations

from pathlib import Path

import torch

from src.app.diffusion_service import DiffusionExplorer, _build_model, discover_checkpoints


def _checkpoint_config() -> dict:
    """Return a small MNIST checkpoint configuration for CPU sampling."""
    return {
        "datos": {"nombre": "mnist"},
        "modelo": {
            "base_channels": 8,
            "channel_multipliers": [1],
            "residual_blocks": 1,
            "attention_resolutions": [],
            "dropout": 0.0,
            "diffusion_steps": 1,
            "beta_start": 0.0001,
            "beta_end": 0.02,
        },
    }


def test_explorer_discovers_and_samples_an_ema_checkpoint(tmp_path: Path):
    """The app can load its own checkpoint contract without training imports."""
    config = _checkpoint_config()
    model = _build_model(config)
    checkpoint = tmp_path / "mnist.pt"
    torch.save({"ema": model.state_dict(), "config": config}, checkpoint)

    available = discover_checkpoints(tmp_path)
    assert available["mnist.pt"].dataset == "mnist"
    explorer = DiffusionExplorer(available, torch.device("cpu"))
    noise, samples, trajectory, info, steps = explorer.generate("mnist.pt", "ddim", 1, 1, seed=11)

    assert noise.size == samples.size == (28, 28)
    assert len(trajectory) == 2
    assert trajectory[0][0] == 0 and trajectory[-1][0] == 1
    assert info.identifier == "mnist.pt"
    assert steps == 1
