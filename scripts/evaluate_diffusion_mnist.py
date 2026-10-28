"""Compute Inception Score and FID for a MNIST DDPM checkpoint."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import Tensor
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from .ddpm_utils import cargar_modelo, cargar_evaluador, construir_difusion, construir_test_loader_con_batch, obtener_normalizacion
except ImportError:
    from ddpm_utils import cargar_modelo, cargar_evaluador, construir_difusion, construir_test_loader_con_batch, obtener_normalizacion


def _normalizar_para_evaluador(
    images: Tensor,
    source_mean: Tensor,
    source_std: Tensor,
    evaluator_mean: Tensor,
    evaluator_std: Tensor,
) -> Tensor:
    """Convert DDPM-normalized images to the evaluator's train-only normalization."""
    source_mean = source_mean.to(images.device)[None, :, None, None]
    source_std = source_std.to(images.device)[None, :, None, None]
    evaluator_mean = evaluator_mean.to(images.device)[None, :, None, None]
    evaluator_std = evaluator_std.to(images.device)[None, :, None, None]
    pixels = (images * source_std + source_mean).clamp(0.0, 1.0)
    return (pixels - evaluator_mean) / evaluator_std


def _statistics(features: Tensor) -> tuple[Tensor, Tensor]:
    mean = features.mean(dim=0)
    centered = features - mean
    covariance = centered.T @ centered / max(features.shape[0] - 1, 1)
    return mean, covariance


def _fid(real: Tensor, generated: Tensor) -> float:
    real_mean, real_covariance = _statistics(real)
    generated_mean, generated_covariance = _statistics(generated)
    mean_distance = (real_mean - generated_mean).square().sum()
    covariance_root, eigenvectors = torch.linalg.eigh(real_covariance.clamp_min(0))
    covariance_root = (eigenvectors * covariance_root.clamp_min(0).sqrt()) @ eigenvectors.T
    product = covariance_root @ generated_covariance @ covariance_root
    product_eigenvalues = torch.linalg.eigvalsh((product + product.T) / 2).clamp_min(0)
    covariance_distance = torch.trace(real_covariance + generated_covariance) - 2 * product_eigenvalues.sqrt().sum()
    return float((mean_distance + covariance_distance).clamp_min(0).item())


def _inception_score(logits: Tensor, splits: int = 10) -> tuple[float, float]:
    probabilities = logits.softmax(dim=1)
    split_count = min(splits, probabilities.shape[0])
    scores = []
    for split in torch.chunk(probabilities, split_count):
        marginal = split.mean(dim=0, keepdim=True)
        kl = split * (split.clamp_min(1e-12).log() - marginal.clamp_min(1e-12).log())
        scores.append(kl.sum(dim=1).mean().exp())
    values = torch.stack(scores)
    return float(values.mean().item()), float(values.std(unbiased=False).item())


def _stratified_indices(dataset: Dataset, num_samples: int, seed: int) -> list[int]:
    """Select a deterministic, class-balanced subset of a labelled dataset."""
    if not 0 < num_samples < len(dataset):
        raise ValueError("num_samples must be between 1 and the dataset size")

    labels = [int(dataset[index][1]) for index in range(len(dataset))]
    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_class.setdefault(label, []).append(index)

    generator = torch.Generator().manual_seed(seed)
    classes = sorted(by_class)
    per_class, remainder = divmod(num_samples, len(classes))
    extra_classes = set(torch.randperm(len(classes), generator=generator)[:remainder].tolist())
    selected: list[int] = []
    for class_index, label in enumerate(classes):
        count = per_class + (class_index in extra_classes)
        class_indices = torch.tensor(by_class[label], dtype=torch.long)
        permutation = torch.randperm(len(class_indices), generator=generator)
        selected.extend(class_indices[permutation[:count]].tolist())
    selected.sort()
    return selected


def _stratified_loader(loader: DataLoader, num_samples: int, seed: int) -> DataLoader:
    """Return a loader over a deterministic class-balanced test subset."""
    indices = _stratified_indices(loader.dataset, num_samples, seed)
    return DataLoader(
        Subset(loader.dataset, indices),
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
    )


def evaluar(
    checkpoint: Path,
    evaluator_checkpoint: Path,
    num_samples: int = 10000,
    batch_size: int = 64,
    device_name: str = "auto",
    seed: int = 42,
    method: str = "DDPM",
) -> dict[str, float]:
    """Evaluate a MNIST checkpoint and return IS/FID metrics."""
    if num_samples < 2 or batch_size < 1:
        raise ValueError("num_samples must be at least 2 and batch_size must be positive")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if not evaluator_checkpoint.is_file():
        raise FileNotFoundError(f"evaluator checkpoint does not exist: {evaluator_checkpoint}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if method not in {"DDPM", "DDIM"}:
        raise ValueError("method must be DDPM or DDIM")
    device_name = "cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name
    device = torch.device(device_name)
    torch.manual_seed(seed)

    model, cfg = cargar_modelo(checkpoint, device)
    evaluator, evaluator_cfg, evaluator_normalization = cargar_evaluador(
        evaluator_checkpoint, device
    )
    if evaluator_cfg["datos"].get("nombre") != "mnist":
        raise ValueError("evaluator-checkpoint must be an MNIST evaluator checkpoint")
    if not hasattr(evaluator, "forward_features"):
        raise ValueError("evaluator model must expose forward_features() for FID-MNIST")
    loader = construir_test_loader_con_batch(cfg, batch_size)
    if num_samples > len(loader.dataset):
        raise ValueError(f"num_samples cannot exceed the test set size ({len(loader.dataset)})")
    source_mean, source_std = obtener_normalizacion(loader)
    if evaluator_normalization is None:
        evaluator_loader = construir_test_loader_con_batch(evaluator_cfg, batch_size)
        evaluator_mean, evaluator_std = obtener_normalizacion(evaluator_loader)
    else:
        evaluator_mean = torch.tensor(evaluator_normalization["mean"], dtype=torch.float32)
        evaluator_std = torch.tensor(evaluator_normalization["std"], dtype=torch.float32)
    if num_samples < len(loader.dataset):
        loader = _stratified_loader(loader, num_samples, seed)
    diffusion = construir_difusion(cfg, device)
    sampling_steps = diffusion.timesteps
    if method == "DDIM":
        sampling_steps = int(cfg["modelo"].get("ddim_timesteps", min(50, diffusion.timesteps)))
        if not 1 <= sampling_steps <= diffusion.timesteps:
            raise ValueError(
                "modelo.ddim_timesteps must be between 1 and modelo.diffusion_steps"
            )
    print(
        f"checkpoint={checkpoint} device={device} "
        f"sampling_steps={sampling_steps} samples={num_samples} "
        f"batch_size={batch_size} seed={seed} evaluator={evaluator_checkpoint}"
    )

    real_features = []
    seen = 0
    print("Extracting features from the real test set...")
    with torch.inference_mode():
        for images, _ in tqdm(loader, total=(num_samples + batch_size - 1) // batch_size):
            count = min(images.shape[0], num_samples - seen)
            if count <= 0:
                break
            evaluator_images = _normalizar_para_evaluador(
                images[:count].to(device), source_mean, source_std, evaluator_mean, evaluator_std
            )
            real_features.append(evaluator.forward_features(evaluator_images).cpu())
            seen += count

    generated_features = []
    generated_logits = []
    seen = 0
    print("Generating and extracting features from Gaussian noise...")
    generation_started = time.perf_counter()
    progress = tqdm(total=num_samples, unit="img", desc="Generating samples")
    with torch.inference_mode():
        while seen < num_samples:
            count = min(batch_size, num_samples - seen)
            batch_started = time.perf_counter()
            if cfg["datos"]["nombre"] != "mnist":
                raise ValueError(f"Unsupported dataset for IS/FID evaluation: {cfg['datos']['nombre']}")
            shape = (count, 1, 28, 28)
            generated = (
                diffusion.sample_ddim(model, shape, sampling_steps=sampling_steps)
                if method == "DDIM"
                else diffusion.sample(model, shape)
            )
            evaluator_images = _normalizar_para_evaluador(
                generated, source_mean, source_std, evaluator_mean, evaluator_std
            )
            generated_logits.append(evaluator(evaluator_images).cpu())
            generated_features.append(evaluator.forward_features(evaluator_images).cpu())
            seen += count
            batch_seconds = time.perf_counter() - batch_started
            progress.update(count)
            progress.set_postfix(
                batch_s=f"{batch_seconds:.1f}",
                total_s=f"{time.perf_counter() - generation_started:.1f}",
            )
    progress.close()
    print(f"generated {seen} samples in {time.perf_counter() - generation_started:.1f}s")
    real_features_tensor = torch.cat(real_features)
    generated_features_tensor = torch.cat(generated_features)
    generated_logits_tensor = torch.cat(generated_logits)
    inception_mean, inception_std = _inception_score(generated_logits_tensor)
    print(f"IS: {inception_mean:.4f} +/- {inception_std:.4f}")
    fid = _fid(real_features_tensor, generated_features_tensor)
    print(f"FID: {fid:.4f}")
    return {"IS": inception_mean, "IS/std": inception_std, "FID": fid}


def main() -> int:
    """Evaluate generated images against the local MNIST test set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluator-checkpoint", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    inference = parser.add_mutually_exclusive_group(required=True)
    inference.add_argument("--DDPM", action="store_true")
    inference.add_argument("--DDIM", action="store_true")
    args = parser.parse_args()
    evaluar(
        args.checkpoint, args.evaluator_checkpoint, args.num_samples, args.batch_size,
        args.device, args.seed, "DDIM" if args.DDIM else "DDPM",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
