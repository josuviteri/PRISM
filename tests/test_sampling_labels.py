"""Label and guidance selection used by the metric and inference scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ddpm_utils  # noqa: E402


def _cfg(num_classes: int, p_uncond: float, guidance_scale: float) -> dict:
    return {
        "modelo": {"num_classes": num_classes},
        "entrenamiento": {"p_uncond": p_uncond, "guidance_scale": guidance_scale},
    }


def test_unconditional_model_gets_no_labels():
    cfg = _cfg(1, 0.0, 0.0)
    assert ddpm_utils.etiquetas_de_muestreo(cfg, 0, 8, torch.device("cpu")) is None
    assert ddpm_utils.escala_de_guiado(cfg) == 1.0


def test_conditional_model_gets_balanced_labels_across_batches():
    cfg = _cfg(10, 0.0, 0.0)
    first = ddpm_utils.etiquetas_de_muestreo(cfg, 0, 7, torch.device("cpu"))
    second = ddpm_utils.etiquetas_de_muestreo(cfg, 7, 13, torch.device("cpu"))
    labels = torch.cat([first, second])
    assert labels.dtype == torch.long
    assert torch.bincount(labels, minlength=10).tolist() == [2] * 10


def test_null_label_mode_and_fixed_class():
    cfg = _cfg(10, 0.1, 3.0)
    cpu = torch.device("cpu")
    assert ddpm_utils.etiquetas_de_muestreo(cfg, 0, 4, cpu, usar_etiquetas=False) is None
    assert ddpm_utils.etiquetas_de_muestreo(cfg, 0, 4, cpu, clase=3).tolist() == [3] * 4
    with pytest.raises(ValueError):
        ddpm_utils.etiquetas_de_muestreo(cfg, 0, 4, cpu, clase=10)


def test_guided_model_uses_checkpoint_scale_unless_overridden():
    cfg = _cfg(10, 0.1, 3.0)
    assert ddpm_utils.escala_de_guiado(cfg) == 3.0
    assert ddpm_utils.escala_de_guiado(cfg, guidance_scale=5.0) == 5.0
    assert ddpm_utils.escala_de_guiado(cfg, usar_etiquetas=False) == 1.0


def test_unguided_conditional_model_rejects_guidance():
    cfg = _cfg(10, 0.0, 0.0)  # the null class was never trained
    assert ddpm_utils.escala_de_guiado(cfg) == 1.0
    with pytest.raises(ValueError):
        ddpm_utils.escala_de_guiado(cfg, guidance_scale=3.0)
