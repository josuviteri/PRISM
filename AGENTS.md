# Contexto del proyecto

## Qué es esto

PRISM replica DDPM, DDIM y Classifier-Free Guidance (CFG) sobre MNIST y
CIFAR-10 con un U-Net en PyTorch. Hay configuraciones incondicionales
(`cifar10.yaml`, `mnist.yaml`) y guiadas por clase (`guided_cifar10.yaml`,
`guided_mnist.yaml`), un clasificador LeNet que actúa de evaluador para FID/IS
de MNIST (`mnist_evaluator.yaml`) y un aplicativo web (`src/app/`).

## Restricciones del curso

- **Regla 5/20:** `python run.py --smoke` en menos de 5 min en CPU;
  el entrenamiento completo en menos de 20 min en una T4 de Colab gratuito.
- Todo entrenamiento se registra en Weights & Biases.
- Las semillas se fijan **solo** en `src/utils/seed_demo.py` (pendiente de
  renombrar cuando se limpien los ficheros de plantilla).
- Los hiperparámetros viven **solo** en `configs/`. Ninguno escrito a fuego.

## Cómo se ejecuta

```bash
source .venv/bin/activate
python run.py --smoke                                   # pipeline entero en miniatura
python run.py --config configs/cifar10.yaml             # entrenamiento completo (también mnist, guided_*, mnist_evaluator)
python run.py --config configs/cifar10.yaml --solo-evaluar   # loss de validación desde el checkpoint, sin entrenar
python run.py --config configs/cifar10.yaml --solo-metricas  # FID/IS desde el checkpoint (sección `metricas:`)
python run.py --serve                                   # aplicativo web
pytest tests/ -q
python tools/tiiac_check_demo.py .                      # checker del curso
```

`--smoke` selecciona `configs/smoke.yaml`: usa solo 64 imágenes, un modelo
pequeño, 10 pasos de difusión y como máximo 10 batches. Sirve para comprobar
que carga datos, construye el U-Net, entrena, evalúa y escribe un checkpoint;
no sirve para medir calidad generativa.

Los hiperparámetros se cambian en `configs/`, no en el código. `dispositivo:
auto` elige CUDA si está disponible y CPU en caso contrario.

Los checkpoints se guardan en `checkpoints/<experimento.nombre>.pt`; la carpeta
está ignorada por Git.

- Difusión: pesos normales (`model`), pesos EMA (`ema`) y configuración usada.
  `--solo-evaluar` carga `ema`.
- Evaluador MNIST (`format: mnist_evaluator_v1`): `model`, `config`,
  `normalizacion` y métricas; **no tiene EMA**. `--solo-evaluar` carga `model`.

W&B: las métricas se registran por época (`train/*`, `val/*`) y el checkpoint
final se sube como Artifact de tipo `model` con el nombre del experimento. La
run se llama `<experimento.nombre>` (y `<experimento.nombre>_eval` en
`--solo-metricas`). Configura `wandb.entidad`, deja `wandb.modo: online` y
ejecuta `wandb login` antes de entrenar. `configs/smoke.yaml` usa
`wandb.modo: offline`; los tests usan `wandb.modo: disabled`.

El entorno de desarrollo validado usa `torch==2.14.1` (build `+cu130`) sobre una
GPU NVIDIA RTX 5070 (Python 3.11 en CI). La instalación de esa versión puede
requerir el índice de paquetes CUDA que corresponda a la máquina; no sustituirla
automáticamente por Torch CPU.

## Convenciones

- Sin rutas absolutas, nunca.
- Toda función pública lleva docstring y anotaciones de tipo.
- **La normalización usa estadísticas de entrenamiento, no del conjunto completo.**
  Lo contrario es una fuga de datos: no da error e infla la métrica.
- El `data augmentation` se aplica a train, jamás a validación ni a test.
- `src/app/` no importa nada de `src/training/`: el aplicativo solo carga un
  checkpoint y responde.
- **`guidance_scale`:** `1.0` = solo predicción condicional (sin guidance) y
  `0.0` = predicción incondicional pura. En modelos sin clases
  (`num_classes: 1`) la escala se ignora.

## Lo que NO hay que tocar

- Los splits de `src/data/loaders.py` (`construir_datasets`): están congelados
  desde el hito H3.
- CIFAR-10 y MNIST se leen con Pillow/IDX propio en `loaders.py`, sin
  `torchvision.datasets`. `torchvision` solo se usa para los pesos de
  Inception-v3 en la evaluación FID/IS de CIFAR-10.

## Estado actual

- Hecho: entrenamiento y evaluación DDPM/DDIM de CIFAR-10 y MNIST, CFG con
  clase null y dropout de condición, evaluador LeNet para MNIST, métricas
  FID/IS, aplicativo web, logging de métricas y Artifacts en W&B.
- En curso: ampliar tests funcionales del pipeline y completar CI.
- Pendiente: métricas con DDPM sobre el test completo (coste de cómputo);
  renombrar los módulos `*_demo.py` (al renombrar, actualizar los imports).
