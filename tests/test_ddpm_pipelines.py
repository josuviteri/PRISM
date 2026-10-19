from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from src.data.loaders import construir_datasets
from src.training import diffusion_loop


def _config(tmp_path: Path, dataset: str, data_root: Path) -> dict:
    """Build a tiny one epoch config for a fixture-backed dataset."""
    channels = 1 if dataset == "mnist" else 3
    return {
        "experimento": {"nombre": f"{dataset}_test", "seed": 7},
        "datos": {"nombre": dataset, "ruta": str(data_root), "val_frac": 0.2,
                  "seed": 19, "subset": None, "augment": False},
        "modelo": {"base_channels": 8, "channel_multipliers": [1, 2],
                   "residual_blocks": 1, "attention_resolutions": [], "dropout": 0.0,
                   "diffusion_steps": 2, "beta_start": 0.0001, "beta_end": 0.02,
                   "ema_decay": 0.9, "ema_warmup": True},
        "entrenamiento": {"epochs": 1, "max_steps": 1, "compile": False,
                          "dispositivo": "cpu", "directorio_checkpoints": str(tmp_path),
                          "learning_rate": 0.0002, "optimizador": "adam", "grad_clip": 1.0,
                          "warmup_steps": 0, "horizontal_flip": False},
        "wandb": {"modo": "disabled"},
        "_expected_channels": channels,
    }


@pytest.mark.parametrize("dataset", ["mnist", "cifar10"])
def test_loader_shapes_splits_and_finite_normalized_values(dataset, request):
    """Both local dataset readers produce finite, normalized image tensors."""
    root = request.getfixturevalue("mnist_root" if dataset == "mnist" else "cifar_root")
    train, val, test = construir_datasets({"datos": {
        "nombre": dataset, "ruta": str(root), "val_frac": 0.2, "seed": 19,
        "augment": False,
    }})
    image, label = train[0]
    channels, size = (1, 28) if dataset == "mnist" else (3, 32)
    assert len(train) == 16
    assert len(val) == 4
    assert len(test) == 5
    assert image.shape == (channels, size, size)
    assert image.dtype == torch.float32 and torch.isfinite(image).all()
    if dataset == "mnist":
        train_images = torch.stack([train[index][0] for index in range(len(train))])
        assert torch.isclose(train_images.mean(), torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(train_images.std(unbiased=False), torch.tensor(1.0), atol=1e-5)
    else:
        assert -1.0 <= image.min() <= image.max() <= 1.0
    assert 0 <= label <= 9


def test_split_is_deterministic_disjoint_and_subset_only_trims_train(mnist_root):
    cfg = {"datos": {"nombre": "mnist", "ruta": str(mnist_root), "val_frac": 0.2,
                     "seed": 13, "subset": 3, "augment": False}}
    train_a, val_a, _ = construir_datasets(cfg)
    train_b, val_b, _ = construir_datasets(cfg)
    assert len(train_a) == len(train_b) == 3
    assert len(val_a) == len(val_b) == 4
    assert train_a.indices == train_b.indices
    assert val_a.indices == val_b.indices
    selected_original_indices = {
        train_a.dataset.indices[index] for index in train_a.indices
    }
    assert selected_original_indices.isdisjoint(val_a.indices)


@pytest.mark.parametrize("dataset", ["mnist", "cifar10"])
def test_pipeline_trains_and_saves_loadable_checkpoint(tmp_path, dataset, request):
    """A single optimization step exercises the real train/evaluate/checkpoint path."""
    root = request.getfixturevalue("mnist_root" if dataset == "mnist" else "cifar_root")
    cfg = _config(tmp_path, dataset, root)
    train, val, _ = construir_datasets(cfg)
    train_loader = DataLoader(Subset(train, range(2)), batch_size=2)
    val_loader = DataLoader(Subset(val, range(2)), batch_size=2)
    model = diffusion_loop.construir_modelo(cfg)
    metrics = diffusion_loop.entrenar(model, train_loader, val_loader, cfg)
    checkpoint = tmp_path / f"{dataset}_test.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)

    assert checkpoint.is_file()
    assert set(("model", "ema", "config", "epoch", "step", "val/loss")).issubset(state)
    assert state["epoch"] == 1 and state["step"] == 1
    assert metrics["val/loss"] >= 0.0
    assert metrics["train/loss"] >= 0.0
    image_channels = cfg["_expected_channels"]
    output = model(torch.randn(2, image_channels, 28 if dataset == "mnist" else 32,
                               28 if dataset == "mnist" else 32), torch.tensor([0, 1]))
    assert output.shape == (2, image_channels, 28 if dataset == "mnist" else 32,
                            28 if dataset == "mnist" else 32)


def test_mnist_rejects_malformed_idx_header(tmp_path):
    """Corrupt IDX files fail with a useful validation error instead of parsing garbage."""
    from src.data.loaders import LocalMNIST

    image_file = tmp_path / "train-images.idx3-ubyte"
    image_file.write_bytes(b"bad header")
    (tmp_path / "train-labels.idx1-ubyte").write_bytes(b"")
    with pytest.raises(ValueError, match="too short"):
        LocalMNIST(tmp_path, train=True)
