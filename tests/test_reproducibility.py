"""Reproducibility guarantees for RNGs, model initialization, and data splits."""

import random

import numpy as np
import torch

from src.data.loaders import construir_datasets
from src.training.diffusion_loop import construir_modelo
from src.utils.seed_demo import fijar_seed


def _model_config() -> dict:
    """Return a tiny model config so initialization checks stay fast on CPU."""
    return {
        "datos": {"nombre": "mnist"},
        "modelo": {
            "base_channels": 8,
            "channel_multipliers": [1],
            "residual_blocks": 1,
            "attention_resolutions": [],
            "dropout": 0.0,
        },
    }


def test_same_seed_reproduces_python_numpy_and_torch_rngs():
    """Resetting the project seed reproduces draws from each supported RNG."""
    fijar_seed(42)
    first = (random.random(), np.random.random(), torch.rand(4))
    fijar_seed(42)
    second = (random.random(), np.random.random(), torch.rand(4))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_same_seed_reproduces_project_model_initialization():
    """Model weights built after resetting the configured seed are identical."""
    cfg = _model_config()
    fijar_seed(42)
    first = construir_modelo(cfg)
    fijar_seed(42)
    second = construir_modelo(cfg)
    assert [name for name, _ in first.named_parameters()] == [
        name for name, _ in second.named_parameters()
    ]
    for first_param, second_param in zip(first.parameters(), second.parameters()):
        assert torch.equal(first_param, second_param)


def test_different_seeds_change_rng_draws_and_model_weights():
    """A seed change affects random draws and initialization, avoiding false determinism."""
    fijar_seed(42)
    first_draw = torch.rand(4)
    first_model = construir_modelo(_model_config())
    fijar_seed(1234)
    second_draw = torch.rand(4)
    second_model = construir_modelo(_model_config())
    assert not torch.equal(first_draw, second_draw)
    assert any(not torch.equal(a, b) for a, b in zip(first_model.parameters(), second_model.parameters()))


def test_split_indices_repeat_for_the_same_seed(mnist_root):
    """The local train/validation split is stable across dataset construction."""
    cfg = {"datos": {"nombre": "mnist", "ruta": str(mnist_root), "val_frac": 0.2,
                     "seed": 17, "augment": False}}
    train_a, val_a, _ = construir_datasets(cfg)
    train_b, val_b, _ = construir_datasets(cfg)
    assert train_a.indices == train_b.indices
    assert val_a.indices == val_b.indices
