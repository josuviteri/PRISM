#!/usr/bin/env python3
"""
Punto de entrada unico del proyecto.

    python run.py --smoke                        # pipeline entero en miniatura, < 5 min en CPU
    python run.py --config configs/default.yaml  # entrenamiento completo, < 20 min en T4
    python run.py --config configs/default.yaml --solo-evaluar

Este fichero es deliberadamente corto: orquesta, no implementa. La logica vive
en src/. Si run.py crece por encima de un centenar de lineas, algo tendria que
haber bajado a un modulo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.seed_demo import fijar_seed     # noqa: E402
from src.utils.tracking import abrir_run, registrar  # noqa: E402
from src.data.loaders import construir_loaders  # noqa: E402
from src.training import diffusion_loop # noqa: E402
from scripts import evaluate_diffusion_mnist, evaluate_diffusion_cifar10  # noqa: E402

# mnist evaluator
from src.training import mnist_evaluator_loop  # noqa: E402

from src.app import serve  # noqa: E402

def cargar_config(ruta: Path) -> dict:
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"),
                    help="fichero de configuracion")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true",
                      help="pipeline entero en miniatura")
    mode.add_argument("--solo-evaluar", action="store_true",
                      help = "carga el checkpoint y evalua respecto a la loss function (soft metrics), sin entrenar")
    mode.add_argument("--solo-metricas", action="store_true",
                      help = "carga el checkpoint y evalua respecto a FID e IS (hard metrics), sin entrenar")
    mode.add_argument("--serve", action="store_true",
                      help="lanza la interfaz web de exploracion de checkpoints")
    
    args = ap.parse_args()

    if args.serve:
        return serve.main([])

    ruta = Path("configs/smoke.yaml") if args.smoke else args.config
    if not ruta.is_file():
        print(f"error: no existe {ruta}", file=sys.stderr)
        return 2
    
    cfg = cargar_config(ruta)

    # 1 · reproducibilidad, antes que nada
    fijar_seed(cfg["experimento"]["seed"])

    if args.solo_metricas:
        metric_cfg = cfg.get("metricas")
        if not isinstance(metric_cfg, dict):
            print("error: falta la seccion 'metricas' en la configuracion", file=sys.stderr)
            return 2
        dataset = cfg.get("datos", {}).get("nombre")
        checkpoint = Path(metric_cfg.get("checkpoint", ""))
        if not checkpoint.is_file():
            print(f"error: no existe el checkpoint {checkpoint}", file=sys.stderr)
            return 2
        method = str(metric_cfg.get("metodo", "DDPM")).upper()
        try:
            if dataset == "mnist":
                evaluator_checkpoint = Path(metric_cfg.get("evaluator_checkpoint", ""))
                if not evaluator_checkpoint.is_file():
                    print(f"error: no existe el checkpoint evaluador {evaluator_checkpoint}", file=sys.stderr)
                    return 2
            elif dataset != "cifar10":
                print(f"error: dataset no soportado para metricas: {dataset}", file=sys.stderr)
                return 2
        except TypeError:
            print("error: la configuracion de metricas contiene una ruta invalida", file=sys.stderr)
            return 2

        run = abrir_run(cfg, evaluar=True)
        try:
            common = {
                "checkpoint": checkpoint,
                "num_samples": int(metric_cfg.get("num_samples", 10000)),
                "batch_size": int(metric_cfg.get("batch_size", 64)),
                "device_name": str(metric_cfg.get("dispositivo", "auto")),
                "seed": int(cfg["experimento"]["seed"]),
                "method": method,
            }
            if dataset == "mnist":
                metricas = evaluate_diffusion_mnist.evaluar(
                    evaluator_checkpoint=evaluator_checkpoint, **common
                )
            else:
                metricas = evaluate_diffusion_cifar10.evaluar(**common)
            registrar(run, metricas)
            print("\n=== metricas ===")
            for key, value in metricas.items():
                print(f"{key}: {value:.4f}")
        finally:
            if run is not None:
                run.finish()
        return 0

    mnist_evaluator = cfg["modelo"]["nombre"] == "lenet_mnist_evaluator"

    # 2 · seguimiento del experimento
    run = abrir_run(cfg)

    # 3 · datos
    train_loader, val_loader = construir_loaders(cfg)

    # 4 · modelo
    if mnist_evaluator:
        modelo = mnist_evaluator_loop.construir_modelo(cfg)
    else:
        modelo = diffusion_loop.construir_modelo(cfg)

    # 5 · entrenar y evaluar
    checkpoint = Path(cfg["entrenamiento"].get("directorio_checkpoints", "checkpoints")) / f'{cfg["experimento"]["nombre"]}.pt'
    if args.solo_evaluar:
        if not checkpoint.is_file():
            print(f"error: no existe el checkpoint {checkpoint}", file=sys.stderr)
            return 2
        estado = torch.load(checkpoint, map_location="cpu", weights_only=True)
        modelo.load_state_dict(estado["ema"])
        modelo.to(diffusion_loop._device(cfg))
    else:
        if mnist_evaluator:
            mnist_evaluator_loop.entrenar(modelo, train_loader, val_loader, cfg, run)
        else:
            diffusion_loop.entrenar(modelo, train_loader, val_loader, cfg, run)
    
    if mnist_evaluator:
        metricas = mnist_evaluator_loop.evaluar(modelo, val_loader, cfg)
    else:
        metricas = diffusion_loop.evaluar(modelo, val_loader, cfg)

    # 6 · el smoke test TIENE que imprimir una metrica: es la prueba de que el
    #     pipeline llega hasta el final. El checker lo comprueba.
    print("\n=== metricas ===")
    for k, v in metricas.items():
        print(f"{k}: {v:.4f}")

    if run is not None:
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
