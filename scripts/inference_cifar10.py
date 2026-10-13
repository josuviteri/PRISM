"""Generate CIFAR-10 images from Gaussian noise with a saved DDPM."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

try:
    from .ddpm_utils import (
        a_rgb,
        cargar_modelo,
        construir_difusion,
        construir_test_loader_con_batch,
        guardar_pngs,
        obtener_normalizacion,
    )
except ImportError:
    from ddpm_utils import (
        a_rgb,
        cargar_modelo,
        construir_difusion,
        construir_test_loader_con_batch,
        guardar_pngs,
        obtener_normalizacion,
    )


def main() -> int:
    """Generate and save images from a checkpoint's EMA model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("generated"))
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.num_samples < 1 or args.batch_size < 1:
        parser.error("--num-samples and --batch-size must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    device = torch.device(device_name)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, cfg = cargar_modelo(args.checkpoint, device)
    loader = construir_test_loader_con_batch(cfg, args.batch_size)
    mean, std = obtener_normalizacion(loader)
    diffusion = construir_difusion(cfg, device)
    print(
        f"checkpoint={args.checkpoint} device={device} "
        f"diffusion_steps={diffusion.timesteps} samples={args.num_samples} "
        f"batch_size={args.batch_size} seed={args.seed}"
    )

    written = 0
    started = time.perf_counter()
    progress = tqdm(total=args.num_samples, unit="img", desc="Generating images")
    while written < args.num_samples:
        count = min(args.batch_size, args.num_samples - written)
        batch_started = time.perf_counter()
        generated = diffusion.sample(model, (count, 3, 32, 32))
        guardar_pngs(a_rgb(generated, mean, std), args.output_dir, written)
        written += count
        batch_seconds = time.perf_counter() - batch_started
        progress.update(count)
        progress.set_postfix(batch_s=f"{batch_seconds:.1f}", total_s=f"{time.perf_counter() - started:.1f}")
    progress.close()

    print(f"generated {written} images in {args.output_dir} in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
