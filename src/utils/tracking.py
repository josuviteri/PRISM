"""Seguimiento de experimentos con Weights & Biases.

Aislado en un modulo para que el resto del codigo no dependa de wandb y para
que los tests puedan correr con modo='disabled'.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any



def abrir_run(cfg: dict, evaluar: bool = False) -> Any | None:
    """Abre un run de W&B, o devuelve None si esta desactivado."""
    wcfg = cfg.get("wandb", {})
    if wcfg.get("modo") == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("aviso: wandb no esta instalado; se sigue sin registrar")
        return None

    return wandb.init(
        project=wcfg.get("proyecto", "tiiac-2026"),
        entity=wcfg.get("entidad"),
        name=(
            f"{cfg['experimento']['nombre']}_eval"
            if evaluar
            else cfg["experimento"]["nombre"]
        ),
        config=cfg,
        mode=wcfg.get("modo", "online"),
    )


def registrar(run: Any | None, metricas: dict, paso: int | None = None) -> None:
    """Registra metricas si hay run abierto. Silencioso si no."""
    if run is not None:
        run.log(metricas, step=paso)


def registrar_checkpoint(run: Any | None, checkpoint: Path, nombre: str) -> None:
    """Sube un checkpoint terminado a W&B como Artifact."""
    if run is None:
        return
    import wandb

    artifact = wandb.Artifact(nombre, type="model")
    artifact.add_file(str(checkpoint))
    run.log_artifact(artifact)
