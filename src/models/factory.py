"""Shared model factory for training and checkpoint-only inference.

Keeping the architecture dispatch here means ``src/app`` can build models
without importing ``src/training`` and that U-Net checkpoints keep working
unchanged. A config selects DiT explicitly with ``modelo.arquitectura: dit`` or
implicitly through a ``dit`` token in ``modelo.nombre`` (for example
``dit_cifar10`` or ``guided_dit_mnist``).
"""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from src.models.dit import model as build_dit
from src.models.unet import model as build_unet


def arquitectura_de_config(model_cfg: Mapping[str, Any]) -> str:
  """Returns ``"unet"`` or ``"dit"`` for a model configuration section."""
  explicit = model_cfg.get("arquitectura")
  if explicit is not None:
    architecture = str(explicit).strip().lower()
    if architecture not in {"unet", "dit"}:
      raise ValueError(f"unsupported model architecture: {explicit}")
    return architecture
  name = str(model_cfg.get("nombre", "")).strip().lower()
  return "dit" if "dit" in name.split("_") else "unet"


def construir_modelo_desde_config(cfg: Mapping[str, Any]) -> nn.Module:
  """Builds the model described by a full project config.

  Image channels are inferred from ``datos.nombre`` so CIFAR-10 and MNIST share
  the same entry point. Unknown keys are ignored by each builder, which allows
  switching architectures without duplicating the rest of the config.
  """
  model_cfg = cfg["modelo"]
  dataset = (cfg.get("datos") or {}).get("nombre", "cifar10")
  channels = 1 if dataset == "mnist" else 3
  architecture = arquitectura_de_config(model_cfg)

  if architecture == "dit":
    return build_dit(
      in_ch=channels,
      out_ch=channels,
      hidden_size=int(model_cfg.get("hidden_size", 256)),
      depth=int(model_cfg.get("depth", 6)),
      num_heads=int(model_cfg.get("num_heads", 4)),
      patch_size=int(model_cfg.get("patch_size", 2)),
      mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
      dropout=float(model_cfg.get("dropout", 0.0)),
      num_classes=int(model_cfg.get("num_classes", 1)),
      frequency_embedding_size=int(model_cfg.get("frequency_embedding_size", 256)),
    )

  return build_unet(
    in_ch=channels,
    out_ch=channels,
    ch=model_cfg["base_channels"],
    ch_mult=tuple(model_cfg["channel_multipliers"]),
    num_res_blocks=model_cfg["residual_blocks"],
    attn_resolutions=tuple(model_cfg["attention_resolutions"]),
    dropout=model_cfg["dropout"],
    num_classes=model_cfg.get("num_classes", 1),
  )
