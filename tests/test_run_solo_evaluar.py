"""run.py --solo-evaluar must load the MNIST evaluator checkpoint (no EMA entry)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

import run


def _evaluator_config(tmp_path: Path, data_root: Path) -> Path:
    cfg = {
        "experimento": {"nombre": "mnist_evaluator_test", "seed": 7},
        "datos": {"nombre": "mnist", "ruta": str(data_root), "val_frac": 0.2, "seed": 19,
                  "subset": None, "batch_size": 4, "num_workers": 0, "augment": False},
        "modelo": {"nombre": "lenet_mnist_evaluator", "channels": [4, 8, 8],
                   "embedding_dim": 8, "num_classes": 10},
        "entrenamiento": {"optimizador": "adamw", "learning_rate": 0.001, "weight_decay": 0.0,
                          "epochs": 1, "early_stopping_patience": 2, "dispositivo": "cpu", "max_batches": 1,
                          "directorio_checkpoints": str(tmp_path / "ckpt")},
        "wandb": {"modo": "disabled"},
    }
    path = tmp_path / "evaluator.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_solo_evaluar_loads_mnist_evaluator_checkpoint(tmp_path, mnist_root, monkeypatch, capsys):
    config = _evaluator_config(tmp_path, mnist_root)
    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(config)])
    assert run.main() == 0
    assert (tmp_path / "ckpt" / "mnist_evaluator_test.pt").is_file()

    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(config), "--solo-evaluar"])
    assert run.main() == 0
    assert "val/accuracy" in capsys.readouterr().out
