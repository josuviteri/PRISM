"""Generate MNIST images from Gaussian noise with a saved DDPM."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

try:
    from .ddpm_utils import (
        _save_image_grid,
        cargar_modelo,
        construir_difusion,
        escala_de_guiado,
        etiquetas_de_muestreo,
    )
except ImportError:
    from ddpm_utils import (
        _save_image_grid,
        cargar_modelo,
        construir_difusion,
        escala_de_guiado,
        etiquetas_de_muestreo,
    )


def main() -> int:
    """Generate and save images from a checkpoint's EMA model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("generated_diffusion"))
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--denoising-steps",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of reverse-diffusion timesteps for DDIM. Defaults to 50, "
            "or the checkpoint's timestep count when it is lower."
        ),
    )
    parser.add_argument("--label", type=int, default=None, metavar="C",
                        help="Generate only class C (class-conditional checkpoints).")
    parser.add_argument("--sin-etiquetas", action="store_true",
                        help="Sample with the null class instead of class labels.")
    parser.add_argument("--guidance-scale", type=float, default=None,
                        help="CFG scale w (default: value stored in the checkpoint config).")
    inference = parser.add_mutually_exclusive_group(required=True)
    inference.add_argument("--DDPM", action="store_true", help="Use the DDPM inference method for sampling")
    inference.add_argument("--DDIM", action="store_true", help="Use the DDIM inference method for sampling")
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
    diffusion = construir_difusion(cfg, device)
    try:
        guidance = escala_de_guiado(cfg, not args.sin_etiquetas, args.guidance_scale)
        etiquetas_de_muestreo(cfg, 0, 1, device, not args.sin_etiquetas, args.label)
    except ValueError as error:
        parser.error(str(error))
    if args.denoising_steps is not None:
        if not args.DDIM:
            parser.error("--denoising-steps is only supported with --DDIM")
        if not 1 <= args.denoising_steps <= diffusion.timesteps:
            parser.error(
                "--denoising-steps must be between 1 and the checkpoint's "
                f"timestep count ({diffusion.timesteps})"
            )
    if args.DDIM:
        sampling_steps = args.denoising_steps or min(50, diffusion.timesteps)
    else:
        sampling_steps = diffusion.timesteps
    print(
        f"checkpoint={args.checkpoint} device={device} "
        f"diffusion_steps={sampling_steps} samples={args.num_samples} "
        f"batch_size={args.batch_size} seed={args.seed} method={'DDIM' if args.DDIM else 'DDPM'}"
    )

    written = 0
    generated_batches = []
    started = time.perf_counter()
    progress = tqdm(total=args.num_samples, unit="img", desc="Generating images")
    while written < args.num_samples:
        count = min(args.batch_size, args.num_samples - written)
        batch_started = time.perf_counter()
        shape = (count, 1, 28, 28) if cfg["datos"]["nombre"] == "mnist" else (count, 3, 32, 32)
        labels = etiquetas_de_muestreo(cfg, written, count, device, not args.sin_etiquetas, args.label)
        if args.DDIM:
            generated = diffusion.sample_ddim(
                model, shape, sampling_steps=sampling_steps, labels=labels, guidance_scale=guidance
            )
        else:
            generated = diffusion.sample(model, shape, labels=labels, guidance_scale=guidance)
        generated_batches.append(generated)
        written += count
        batch_seconds = time.perf_counter() - batch_started
        progress.update(count)
        progress.set_postfix(batch_s=f"{batch_seconds:.1f}", total_s=f"{time.perf_counter() - started:.1f}")
    progress.close()

    grid_path = args.output_dir / "grid.png"
    _save_image_grid(torch.cat(generated_batches), grid_path)
    print(f"generated {written} images in {grid_path} in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
