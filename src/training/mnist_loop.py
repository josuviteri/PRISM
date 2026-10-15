"""Bucle de entrenamiento y evaluacion para MNIST."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from src.models.unet import model as build_unet
from src.utils.tracking import registrar, registrar_checkpoint


def _extract(values: Tensor, timesteps: Tensor, shape: tuple[int, ...]) -> Tensor:
    return values.gather(0, timesteps).reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


class GaussianDiffusion:
    """Forward and reverse-process calculations for epsilon prediction."""

    def __init__(self, timesteps: int, beta_start: float, beta_end: float, device: torch.device):
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.timesteps = timesteps
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = alpha_bars.sqrt()
        self.sqrt_one_minus_alpha_bars = (1.0 - alpha_bars).sqrt()

    def q_sample(self, x_start: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        shape = tuple(x_start.shape)
        return (
            _extract(self.sqrt_alpha_bars, timesteps, shape) * x_start
            + _extract(self.sqrt_one_minus_alpha_bars, timesteps, shape) * noise
        )

    def loss(self, denoiser: nn.Module, x_start: Tensor) -> Tensor:
        timesteps = torch.randint(0, self.timesteps, (x_start.shape[0],), device=x_start.device)
        noise = torch.randn_like(x_start)
        noisy = self.q_sample(x_start, timesteps, noise)
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=x_start.device.type == "cuda",
        ):
            prediction = denoiser(noisy, timesteps)
        return torch.nn.functional.mse_loss(prediction, noise)

    @torch.no_grad()
    def sample(self, denoiser: nn.Module, shape: tuple[int, ...]) -> Tensor:
        sample = torch.randn(shape, device=self.betas.device)
        was_training = denoiser.training
        denoiser.eval()
        for step in reversed(range(self.timesteps)):
            timesteps = torch.full((shape[0],), step, device=sample.device, dtype=torch.long)
            predicted_noise = denoiser(sample, timesteps)
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            beta = self.betas[step]
            mean = (sample - beta * predicted_noise / (1.0 - alpha_bar).sqrt()) / alpha.sqrt()
            sample = mean + beta.sqrt() * torch.randn_like(sample) if step > 0 else mean
        if was_training:
            denoiser.train()
        return sample.clamp(-1.0, 1.0)


class ExponentialMovingAverage:
    """Maintains the EMA weights used by DDPM for sampling."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {name: value.detach().clone() for name, value in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, value in model.named_parameters():
            value.copy_(self.shadow[name])


def construir_modelo(cfg: dict[str, Any]) -> nn.Module:
    """Builds the one-channel MNIST U-Net from the model configuration."""
    model_cfg = cfg["modelo"]
    return build_unet(
        in_ch=1,
        out_ch=1,
        ch=model_cfg["base_channels"],
        ch_mult=tuple(model_cfg["channel_multipliers"]),
        num_res_blocks=model_cfg["residual_blocks"],
        attn_resolutions=tuple(model_cfg["attention_resolutions"]),
        dropout=model_cfg["dropout"],
    )


def _device(cfg: dict[str, Any]) -> torch.device:
    requested = cfg.get("entrenamiento", {}).get("dispositivo", "auto")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _checkpoint_path(cfg: dict[str, Any]) -> Path:
    directory = cfg["entrenamiento"].get("directorio_checkpoints", "checkpoints")
    return Path(directory) / f'{cfg["experimento"]["nombre"]}.pt'


def _gradient_norm(model: nn.Module) -> Tensor:
    gradients = [parameter.grad.detach() for parameter in model.parameters()
                 if parameter.grad is not None]
    if not gradients:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.sqrt(torch.stack([gradient.float().pow(2).sum() for gradient in gradients]).sum())


def _apply_horizontal_flip(images: Tensor) -> Tensor:
    """Randomly flips half of a batch of image tensors left-to-right."""
    if images.ndim != 4:
        return images
    mask = torch.rand(images.shape[0], device=images.device) < 0.5
    return torch.where(mask[:, None, None, None], images.flip(dims=(-1,)), images)


def _save_checkpoint(
    model: nn.Module,
    ema: ExponentialMovingAverage,
    cfg: dict[str, Any],
    epoch: int,
    val_loss: float,
) -> Path:
    checkpoint = _checkpoint_path(cfg)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    ema_model = deepcopy(model).cpu()
    ema.copy_to(ema_model)
    torch.save({
        "model": model.cpu().state_dict(),
        "ema": ema_model.state_dict(),
        "config": cfg,
        "epoch": epoch,
        "val/loss": val_loss,
    }, checkpoint)
    model.to(next(iter(ema.shadow.values())).device)
    return checkpoint


def _epoch(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    loader,
    optimizer,
    device: torch.device,
    grad_clip: float | None,
    ema: ExponentialMovingAverage | None,
    max_batches: int | None = None,
    ema_model: nn.Module | None = None,
) -> tuple[float, int, float]:
    model.train()
    total_loss = 0.0
    total_items = 0
    batches = 0
    maximum_gradient_norm = 0.0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = diffusion.loss(model, images)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at batch {batches + 1}: {loss.item()}")
        loss.backward()
        gradient_norm = _gradient_norm(model)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(
                f"non-finite gradient norm at batch {batches + 1}: {gradient_norm.item()}")
        maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm.item())
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise FloatingPointError(f"non-finite model parameters at batch {batches + 1}")
        if ema is not None:
            ema.update(model if ema_model is None else ema_model)
        total_loss += loss.item() * images.shape[0]
        total_items += images.shape[0]
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    return total_loss / max(total_items, 1), batches, maximum_gradient_norm


@torch.no_grad()
def evaluar(model: nn.Module, loader, cfg: dict[str, Any], device: torch.device | None = None) -> dict[str, float]:
    """Computes the average DDPM noise-prediction loss on a loader."""
    device = device or _device(cfg)
    model.eval()
    model_cfg = cfg["modelo"]
    diffusion = GaussianDiffusion(
        model_cfg["diffusion_steps"], model_cfg["beta_start"], model_cfg["beta_end"], device)
    total_loss = 0.0
    total_items = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        loss = diffusion.loss(model, images)
        total_loss += loss.item() * images.shape[0]
        total_items += images.shape[0]
    return {"val/loss": total_loss / max(total_items, 1)}


def entrenar(model: nn.Module, train_loader, val_loader, cfg: dict[str, Any], run=None) -> dict[str, float]:
    """Trains a DDPM and writes its final model and EMA weights to disk."""
    device = _device(cfg)
    model.to(device)
    model_cfg = cfg["modelo"]
    train_cfg = cfg["entrenamiento"]
    train_model = model
    if train_cfg.get("compile", False) and device.type == "cuda":
        train_model = torch.compile(model)
    diffusion = GaussianDiffusion(
        model_cfg["diffusion_steps"], model_cfg["beta_start"], model_cfg["beta_end"], device)
    optimizer_name = train_cfg["optimizador"].lower()
    optimizer_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[optimizer_name]
    optimizer = optimizer_cls(train_model.parameters(), lr=train_cfg["learning_rate"])
    ema = ExponentialMovingAverage(model, model_cfg["ema_decay"])
    max_steps = train_cfg.get("max_steps")
    global_step = 0
    best_val_loss = float("inf")
    metrics: dict[str, float] = {}

    for epoch in range(train_cfg["epochs"]):
        remaining_steps = None if max_steps is None else max_steps - global_step
        if remaining_steps is not None and remaining_steps <= 0:
            break
        train_loss, batches, gradient_norm = _epoch(
            train_model,
            diffusion,
            train_loader,
            optimizer,
            device,
            train_cfg.get("grad_clip"),
            ema,
            remaining_steps,
            model,
        )
        global_step += batches
        metrics = evaluar(model, val_loader, cfg, device)
        metrics["train/loss"] = train_loss
        metrics["train/grad_norm"] = gradient_norm
        metrics["train/learning_rate"] = optimizer.param_groups[0]["lr"]
        metrics["epoch"] = float(epoch + 1)
        registrar(run, metrics, paso=global_step)
        print(
            f"epoch {epoch + 1}: train/loss={train_loss:.4f} "
            f"val/loss={metrics['val/loss']:.4f} "
            f"grad_norm={gradient_norm:.4f} lr={optimizer.param_groups[0]['lr']:.6g}"
        )
        if not torch.isfinite(torch.tensor(metrics["val/loss"])):
            raise FloatingPointError(f"non-finite validation loss at epoch {epoch + 1}")
        if metrics["val/loss"] < best_val_loss:
            best_val_loss = metrics["val/loss"]
        checkpoint = _save_checkpoint(model, ema, cfg, epoch + 1, best_val_loss)
        registrar_checkpoint(run, checkpoint, cfg["experimento"]["nombre"])
        if max_steps is not None and global_step >= max_steps:
            break

    model.to(device)
    return metrics
