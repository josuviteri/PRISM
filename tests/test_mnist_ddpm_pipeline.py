from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from run import _pipeline
from src.data.loaders import construir_datasets


ROOT = Path(__file__).resolve().parents[1]
MNIST_ROOT = ROOT / "data" / "data" / "mnist"


def _config(tmp_path: Path) -> dict:
    with (ROOT / "configs" / "mnist.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    cfg["datos"].update({"ruta": str(MNIST_ROOT), "num_workers": 0, "pin_memory": False})
    cfg["modelo"].update({
        "base_channels": 8,
        "channel_multipliers": [1, 2],
        "residual_blocks": 1,
        "attention_resolutions": [14],
        "diffusion_steps": 2,
    })
    cfg["entrenamiento"].update({
        "epochs": 1,
        "max_steps": 1,
        "compile": False,
        "dispositivo": "cpu",
        "directorio_checkpoints": str(tmp_path),
    })
    cfg["wandb"]["modo"] = "disabled"
    return cfg


def test_mnist_loader_returns_grayscale_and_split():
    cfg = {
        "datos": {
            "nombre": "mnist",
            "ruta": str(MNIST_ROOT),
            "val_frac": 0.1,
            "seed": 42,
            "subset": 8,
            "augment": False,
        }
    }
    train, val, test = construir_datasets(cfg)

    image, label = train[0]
    assert len(train) == 8
    assert len(val) == 6000
    assert len(test) == 10000
    assert image.shape == (1, 28, 28)
    assert image.dtype == torch.float32
    assert torch.isfinite(image).all()
    assert image.min() < 0.0 < image.max()
    assert 0 <= label <= 9


def test_mnist_pipeline_trains_and_writes_checkpoint(tmp_path):
    cfg = _config(tmp_path)
    pipeline = _pipeline(cfg)
    train, val, _ = construir_datasets(cfg)
    train_loader = DataLoader(Subset(train, range(2)), batch_size=2, shuffle=False)
    val_loader = DataLoader(Subset(val, range(2)), batch_size=2, shuffle=False)
    model = pipeline.construir_modelo(cfg)

    metrics = pipeline.entrenar(model, train_loader, val_loader, cfg)
    checkpoint = tmp_path / "mnist_ddpm_v0.1.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)

    assert metrics["val/loss"] >= 0.0
    assert checkpoint.is_file()
    assert set(("model", "ema", "config", "epoch", "val/loss")).issubset(state)
    output = model(torch.randn(2, 1, 28, 28), torch.tensor([0, 1]))
    assert output.shape == (2, 1, 28, 28)
