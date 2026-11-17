"""PyTorch training and (soft metric) evaluation for diffusion models."""

from __future__ import annotations

import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from src.models.unet import model as build_unet
from src.utils.diffusion_utils import GaussianDiffusion
from src.utils.tracking import registrar, registrar_checkpoint


def _extract(values: Tensor, timesteps: Tensor, shape: tuple[int, ...]) -> Tensor:
    return values.gather(0, timesteps).reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


class ExponentialMovingAverage:
    """EMA copy of a model, used for validation and sampling.

    `module` is a frozen copy of the model whose weights follow the live model. The decay
    warms up as min(decay, (1 + n) / (10 + n)), so the average is not dominated by the random
    initialisation in short runs (with a plain 0.9999 and ~50k steps, ~1% of the EMA weights
    would still be the untrained init plus the very poor early weights, which shows up as
    noisy samples). Buffers, if a model has any (e.g. fixed positional embeddings), are copied.
    """

    def __init__(self, model: nn.Module, decay: float, warmup: bool = True):
        self.decay = decay
        self.warmup = warmup
        self.num_updates = 0
        self.module = deepcopy(model).eval().requires_grad_(False)
        self._source = list(model.parameters())
        self._target = list(self.module.parameters())
        self._source_buffers = list(model.buffers())
        self._target_buffers = list(self.module.buffers())

    def current_decay(self) -> float:
        if not self.warmup:
            return self.decay
        return min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))

    @torch.no_grad()
    def update(self) -> None:
        self.num_updates += 1
        decay = self.current_decay()
        torch._foreach_lerp_(self._target, self._source, 1.0 - decay)
        for target, source in zip(self._target_buffers, self._source_buffers):
            target.copy_(source)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.module.state_dict())


def construir_modelo(cfg: dict[str, Any]) -> nn.Module:
    """Builds the CIFAR U-Net from the model section of the configuration."""
    model_cfg = cfg["modelo"]
    is_mnist = cfg["datos"]["nombre"] == "mnist"
    return build_unet(
        in_ch=1 if is_mnist else 3,
        out_ch=1 if is_mnist else 3,
        ch=model_cfg["base_channels"],
        ch_mult=tuple(model_cfg["channel_multipliers"]),
        num_res_blocks=model_cfg["residual_blocks"],
        attn_resolutions=tuple(model_cfg["attention_resolutions"]),
        dropout=model_cfg["dropout"],
        num_classes=model_cfg.get("num_classes", 1),
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


def _resume_path(cfg: dict[str, Any]) -> Path:
    directory = cfg["entrenamiento"].get("directorio_checkpoints", "checkpoints")
    return Path(directory) / f'{cfg["experimento"]["nombre"]}.resume.pt'


def _apply_horizontal_flip(images: Tensor) -> Tensor:
    """Randomly flips half of a batch of image tensors left-to-right."""
    if images.ndim != 4:
        return images
    mask = torch.rand(images.shape[0], device=images.device) < 0.5
    return torch.where(mask[:, None, None, None], images.flip(dims=(-1,)), images)


def _warmup_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int):
    """Creates a per-update linear warmup scheduler."""
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")

    def warmup_factor(step: int) -> float:
        if warmup_steps == 0:
            return 1.0
        return min(1.0, (step + 1) / warmup_steps)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_factor)


def _cpu_state_dict(module: nn.Module) -> dict[str, Tensor]:
    # Copies tensors to the CPU without moving the module itself, so a compiled model and the
    # optimizer keep pointing at the same GPU parameters.
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / (path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _save_checkpoint(
    model: nn.Module,
    ema: ExponentialMovingAverage,
    cfg: dict[str, Any],
    epoch: int,
    step: int,
    val_loss: float,
) -> Path:
    """Weights-only checkpoint (model + EMA + config): this is what inference loads."""
    checkpoint = _checkpoint_path(cfg)
    _atomic_save({
        "model": _cpu_state_dict(model),
        "ema": _cpu_state_dict(ema.module),
        "config": cfg,
        "epoch": epoch,
        "step": step,
        "val/loss": val_loss,
    }, checkpoint)
    return checkpoint


def _save_resume_state(
    optimizer: torch.optim.Optimizer,
    scheduler,
    ema: ExponentialMovingAverage,
    cfg: dict[str, Any],
    epoch: int,
    step: int,
) -> Path:
    """Optimizer/scheduler/counters needed to resume; kept apart so the main file stays small."""
    path = _resume_path(cfg)
    _atomic_save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ema_updates": ema.num_updates,
        "epoch": epoch,
        "step": step,
    }, path)
    return path


def _save_image_grid(images: Tensor, path: Path, scale: int = 2) -> None:
    """Writes images in [-1, 1] as a near-square PNG grid (no torchvision needed)."""
    from PIL import Image

    pixels = ((images.detach().float().cpu() + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    count, channels, height, width = pixels.shape
    columns = int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(count / columns))
    canvas = torch.zeros(channels, rows * height, columns * width, dtype=torch.uint8)
    for index in range(count):
        row, column = divmod(index, columns)
        canvas[:, row * height:(row + 1) * height, column * width:(column + 1) * width] = pixels[index]
    array = canvas.permute(1, 2, 0).numpy()
    if channels == 1:
        array = array[..., 0]
    image = Image.fromarray(array)
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _log_image(run, path: Path, step: int) -> None:
    if run is None:
        return
    try:
        import wandb

        run.log({"samples/ema": wandb.Image(str(path))}, step=step)
    except Exception as error:  # image logging must never kill a training run
        print(f"could not log samples to wandb: {error}")


def _epoch(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    loader,
    optimizer,
    scheduler,
    device: torch.device,
    grad_clip: float | None,
    ema: ExponentialMovingAverage | None,
    max_batches: int | None = None,
    horizontal_flip: bool = False,
    p_uncond: float = 0.0,
    num_classes: int = 1,
    check_every: int = 50,
) -> tuple[float, int, float]:
    """One training epoch with no per-step host syncs (finiteness is checked periodically)."""
    model.train()
    total_loss = torch.zeros((), device=device)
    maximum_gradient_norm = torch.zeros((), device=device)
    total_items = 0
    batches = 0
    max_norm = grad_clip if grad_clip is not None else float("inf")

    def check_finite() -> None:
        # torch.maximum propagates NaN, so a single bad step anywhere in the window is caught.
        if not bool(torch.isfinite(torch.stack([total_loss, maximum_gradient_norm])).all()):
            raise FloatingPointError(
                f"non-finite training loss or gradient norm within the first {batches} batches "
                "of this epoch")

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        if horizontal_flip:
            images = _apply_horizontal_flip(images)
        optimizer.zero_grad(set_to_none=True)
        conditional_labels = labels if num_classes > 1 else None
        loss = diffusion.loss(
            model, images, labels=conditional_labels, p_uncond=p_uncond)
        loss.backward()
        # Returns the norm measured before clipping, so it is also the monitored value.
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update()
        total_loss += loss.detach() * images.shape[0]
        maximum_gradient_norm = torch.maximum(maximum_gradient_norm, gradient_norm.detach())
        total_items += images.shape[0]
        batches += 1
        if batches % check_every == 0:
            check_finite()
        if max_batches is not None and batches >= max_batches:
            break
    check_finite()
    return total_loss.item() / max(total_items, 1), batches, maximum_gradient_norm.item()


@torch.no_grad()
def evaluar(
    model: nn.Module,
    loader,
    cfg: dict[str, Any],
    device: torch.device | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Average DDPM noise-prediction loss on a loader.

    The timesteps and noise come from a fixed generator, so the number is comparable between
    epochs (and between the raw and EMA weights) instead of jittering with the random draws.
    """
    device = device or _device(cfg)
    model.eval()
    model_cfg = cfg["modelo"]
    diffusion = GaussianDiffusion(
        model_cfg["diffusion_steps"], model_cfg["beta_start"], model_cfg["beta_end"], device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    total_loss = torch.zeros((), device=device)
    total_items = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        conditional_labels = labels if getattr(model, "num_classes", 1) > 1 else None
        total_loss += diffusion.loss(
            model, images, generator=generator, labels=conditional_labels, p_uncond=0.0
        ) * images.shape[0]
        total_items += images.shape[0]
    return {"val/loss": total_loss.item() / max(total_items, 1)}


def entrenar(model: nn.Module, train_loader, val_loader, cfg: dict[str, Any], run=None) -> dict[str, float]:
    """Trains a DDPM and writes model + EMA weights (and resume state) to disk every epoch."""
    device = _device(cfg)
    model_cfg = cfg["modelo"]
    train_cfg = cfg["entrenamiento"]
    experiment_name = cfg["experimento"]["nombre"]
    seed = cfg["experimento"].get("seed")
    if seed is not None:
        torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model.to(device)
    train_model = model
    if train_cfg.get("compile", False) and device.type == "cuda":
        train_model = torch.compile(model)
    diffusion = GaussianDiffusion(
        model_cfg["diffusion_steps"], model_cfg["beta_start"], model_cfg["beta_end"], device)
    optimizer_name = train_cfg["optimizador"].lower()
    optimizer_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[optimizer_name]
    optimizer = optimizer_cls(
        train_model.parameters(), lr=train_cfg["learning_rate"], fused=device.type == "cuda")
    scheduler = _warmup_scheduler(optimizer, train_cfg.get("warmup_steps", 0))
    ema = ExponentialMovingAverage(model, model_cfg["ema_decay"], model_cfg.get("ema_warmup", True))
    max_steps = train_cfg.get("max_steps")
    total_epochs = train_cfg["epochs"]
    sample_every = train_cfg.get("muestras_cada_epocas", 0)
    sample_count = train_cfg.get("muestras_n", 64)
    sample_dir = Path(train_cfg.get("directorio_muestras", "samples")) / experiment_name
    global_step = 0
    start_epoch = 0
    metrics: dict[str, float] = {}
    checkpoint: Path | None = None

    if train_cfg.get("reanudar", False):
        checkpoint, resume_file = _checkpoint_path(cfg), _resume_path(cfg)
        if checkpoint.exists() and resume_file.exists():
            weights = torch.load(checkpoint, map_location=device, weights_only=False)
            state = torch.load(resume_file, map_location=device, weights_only=False)
            if weights["epoch"] != state["epoch"]:
                print(f"warning: weights are from epoch {weights['epoch']} but optimizer state is "
                      f"from epoch {state['epoch']} (interrupted save?)")
            model.load_state_dict(weights["model"])
            ema.module.load_state_dict(weights["ema"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            ema.num_updates = state["ema_updates"]
            global_step = state["step"]
            start_epoch = state["epoch"]
            print(f"resumed from {checkpoint} at epoch {start_epoch}, step {global_step}")
        else:
            checkpoint = None
            print("reanudar is true but no checkpoint was found; starting from scratch")

    for epoch in range(start_epoch, total_epochs):
        remaining_steps = None if max_steps is None else max_steps - global_step
        if remaining_steps is not None and remaining_steps <= 0:
            break
        started = time.perf_counter()
        train_loss, batches, gradient_norm = _epoch(
            train_model,
            diffusion,
            train_loader,
            optimizer,
            scheduler,
            device,
            train_cfg.get("grad_clip"),
            ema,
            remaining_steps,
            train_cfg.get("horizontal_flip", False),
            train_cfg.get("p_uncond", 0.0),
            model_cfg.get("num_classes", 1),
        )
        elapsed = time.perf_counter() - started
        global_step += batches
        # Validation loss is reported for the EMA weights (the ones used for sampling) and for
        # the raw weights. Both use the same fixed noise, so they are directly comparable.
        metrics = {"val/loss": evaluar(ema.module, val_loader, cfg, device)["val/loss"]}
        metrics["val/loss_raw"] = evaluar(model, val_loader, cfg, device)["val/loss"]
        metrics["train/loss"] = train_loss
        metrics["train/grad_norm"] = gradient_norm
        metrics["train/learning_rate"] = optimizer.param_groups[0]["lr"]
        metrics["train/steps_per_second"] = batches / max(elapsed, 1e-9)
        metrics["epoch"] = float(epoch + 1)
        registrar(run, metrics, paso=global_step)
        print(
            f"epoch {epoch + 1}/{total_epochs}: train/loss={train_loss:.4f} "
            f"val/loss={metrics['val/loss']:.4f} val/loss_raw={metrics['val/loss_raw']:.4f} "
            f"grad_norm_max={gradient_norm:.4f} lr={optimizer.param_groups[0]['lr']:.6g} "
            f"{metrics['train/steps_per_second']:.2f} it/s"
        )
        if not math.isfinite(metrics["val/loss"]) or not math.isfinite(metrics["val/loss_raw"]):
            raise FloatingPointError(f"non-finite validation loss at epoch {epoch + 1}")

        _save_resume_state(optimizer, scheduler, ema, cfg, epoch + 1, global_step)
        checkpoint = _save_checkpoint(
            model, ema, cfg, epoch + 1, global_step, metrics["val/loss"])

        finished = epoch + 1 == total_epochs or (max_steps is not None and global_step >= max_steps)
        if sample_every and ((epoch + 1) % sample_every == 0 or finished):
            # Same seed every time: the same starting noise lets you watch the samples improve.
            generator = torch.Generator(device=device)
            generator.manual_seed(0 if seed is None else seed)
            image_shape = tuple(val_loader.dataset[0][0].shape)
            num_classes = model_cfg.get("num_classes", 1)
            sample_labels = (
                torch.arange(sample_count, device=device, dtype=torch.long) % num_classes
                if num_classes > 1 else None
            )
            samples = diffusion.sample(
                ema.module, (sample_count, *image_shape), generator=generator,
                labels=sample_labels,
                guidance_scale=float(train_cfg.get("guidance_scale", 1.0)))
            grid_path = sample_dir / f"epoch_{epoch + 1:04d}.png"
            _save_image_grid(samples, grid_path)
            _log_image(run, grid_path, global_step)
            print(f"saved {sample_count} EMA samples to {grid_path}")
        if finished:
            break

    if checkpoint is not None:
        registrar_checkpoint(run, checkpoint, experiment_name)
    model.to(device)
    return metrics