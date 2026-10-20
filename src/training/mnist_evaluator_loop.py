"""Training loop for the MNIST classifier used by IS-MNIST and FID-MNIST."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from src.models.LeNet import LeNet, model as build_lenet
from src.utils.tracking import registrar, registrar_checkpoint


def construir_modelo(cfg: dict[str, Any]) -> LeNet:
    """Construct the MNIST feature extractor described by ``cfg``."""
    return build_lenet(cfg)


def _device(cfg: dict[str, Any]) -> torch.device:
    """Resolve the configured device, failing clearly when CUDA was required."""
    requested = cfg["entrenamiento"].get("dispositivo", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _checkpoint_path(cfg: dict[str, Any]) -> Path:
    """Return the path of the best-validation evaluator checkpoint."""
    train_cfg = cfg["entrenamiento"]
    return Path(train_cfg["directorio_checkpoints"]) / f'{cfg["experimento"]["nombre"]}.pt'


def _cpu_state_dict(module: nn.Module) -> dict[str, Tensor]:
    """Copy a state dictionary to CPU without moving the training module."""
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def _normalizacion(loader: Any) -> dict[str, list[float]]:
    """Extract the train-fitted normalisation used by a project data loader."""
    dataset = loader.dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    transform = getattr(dataset, "transform", None)
    for operation in getattr(transform, "transforms", []):
        if operation.__class__.__name__ == "_Normalize":
            return {
                "mean": operation.mean.flatten().tolist(),
                "std": operation.std.flatten().tolist(),
            }
    raise ValueError("the evaluator loader must include the MNIST train normalisation")


def _save_checkpoint(
    model: LeNet,
    cfg: dict[str, Any],
    epoch: int,
    global_step: int,
    metrics: dict[str, float],
    normalizacion: dict[str, list[float]],
) -> Path:
    """Atomically persist the best-validation feature extractor and its provenance."""
    path = _checkpoint_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": "mnist_evaluator_v1",
            "model": _cpu_state_dict(model),
            "architecture": model.architecture_config(),
            "seed": cfg["experimento"]["seed"],
            "normalizacion": normalizacion,
            "config": cfg,
            "epoch": epoch,
            "step": global_step,
            **metrics,
        },
        temporary,
    )
    temporary.replace(path)
    return path


def _make_scheduler(optimizer: torch.optim.Optimizer, cfg: dict[str, Any]):
    """Create the configured epoch-level learning-rate scheduler, if any."""
    scheduler_cfg = cfg["entrenamiento"].get("scheduler", {"nombre": "none"})
    name = scheduler_cfg.get("nombre", "none").lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=scheduler_cfg["t_max"], eta_min=scheduler_cfg["eta_min"]
        )
    if name == "reduce_lr_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=scheduler_cfg["factor"],
            patience=scheduler_cfg["patience"],
            min_lr=scheduler_cfg["min_lr"],
        )
    raise ValueError(f"unknown scheduler: {name}")


def _train_epoch(
    model: LeNet,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None,
) -> tuple[float, float, int]:
    """Optimise one epoch and return loss, accuracy, and processed batches."""
    model.train()
    loss_sum = 0.0
    correct = 0
    count = 0
    batches = 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite MNIST evaluator training loss")
        loss.backward()
        optimizer.step()
        loss_sum += loss.detach().item() * labels.numel()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += labels.numel()
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    return loss_sum / max(count, 1), correct / max(count, 1), batches


@torch.no_grad()
def evaluar(
    model: LeNet,
    loader: Any,
    cfg: dict[str, Any],
    device: torch.device | None = None,
) -> dict[str, float]:
    """Evaluate classification loss and accuracy without using test data for selection."""
    active_device = device or _device(cfg)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images, labels = images.to(active_device, non_blocking=True), labels.to(active_device, non_blocking=True)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * labels.numel()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += labels.numel()
    return {"val/loss": loss_sum / max(count, 1), "val/accuracy": correct / max(count, 1)}


def entrenar(model: LeNet, train_loader: Any, val_loader: Any, cfg: dict[str, Any], run=None) -> dict[str, float]:
    """Train LeNet, early-stop on validation accuracy, and retain the best checkpoint.

    The caller is responsible for calling ``fijar_seed`` before creating loaders
    and the model; seeds intentionally remain centralised in ``seed_demo.py``.
    """
    device = _device(cfg)
    train_cfg = cfg["entrenamiento"]
    model.to(device)
    optimizer_name = train_cfg["optimizador"].lower()
    optimizers = {"adamw": torch.optim.AdamW}
    if optimizer_name not in optimizers:
        raise ValueError(f"unsupported MNIST evaluator optimizer: {optimizer_name}")
    optimizer = optimizers[optimizer_name](
        model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler = _make_scheduler(optimizer, cfg)
    criterion = nn.CrossEntropyLoss()
    patience = train_cfg["early_stopping_patience"]
    normalizacion = _normalizacion(train_loader)
    best_accuracy = -math.inf
    best_metrics: dict[str, float] = {}
    bad_epochs = 0
    global_step = 0
    checkpoint: Path | None = None

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss, train_accuracy, batches = _train_epoch(
            model, train_loader, optimizer, criterion, device, train_cfg.get("max_batches")
        )
        global_step += batches
        val_metrics = evaluar(model, val_loader, cfg, device)
        metrics = {
            "epoch": float(epoch),
            "train/loss": train_loss,
            "train/accuracy": train_accuracy,
            "train/learning_rate": optimizer.param_groups[0]["lr"],
            **val_metrics,
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f"non-finite evaluator metric at epoch {epoch}")
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["val/accuracy"])
            else:
                scheduler.step()
        registrar(run, metrics, paso=global_step)
        print(
            f"epoch {epoch}/{train_cfg['epochs']}: train/accuracy={train_accuracy:.4%} "
            f"val/accuracy={val_metrics['val/accuracy']:.4%} val/loss={val_metrics['val/loss']:.4f}"
        )

        if val_metrics["val/accuracy"] > best_accuracy:
            best_accuracy = val_metrics["val/accuracy"]
            best_metrics = metrics
            bad_epochs = 0
            checkpoint = _save_checkpoint(model, cfg, epoch, global_step, metrics, normalizacion)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping after {epoch} epochs (patience={patience})")
                break

    if checkpoint is None:
        raise RuntimeError("MNIST evaluator did not complete an epoch")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    registrar_checkpoint(run, checkpoint, cfg["experimento"]["nombre"])
    return best_metrics
