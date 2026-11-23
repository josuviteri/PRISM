# Contexto del proyecto

<!--
Fichero de contexto para asistentes de IA (Claude Code, Copilot, Cursor…).
Se explica en la sesión A1 del 9 de noviembre.

Dos criterios para escribirlo:
  1. Va lo que le dirías a alguien que se incorpora hoy al proyecto.
  2. Si te encuentras repitiendo la misma corrección tres veces, esa
     corrección va aquí.

Dos páginas se leen enteras. Quince, no.
-->

## Qué es esto

PRISM replica DDPM y DDIM, y añade Classifier-Free Guidance (CFG) para MNIST y
CIFAR-10. El pipeline PyTorch es común para ambos datasets y admite dos
denoisers: el U-Net original y un Diffusion Transformer (DiT). El U-Net sigue
siendo el modelo por defecto para no romper checkpoints antiguos; el DiT se
selecciona por configuración.

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
python run.py --smoke                          # pipeline entero en miniatura
python run.py --config configs/default.yaml    # entrenamiento completo
python run.py --config configs/default.yaml --solo-evaluar
python run.py --config configs/smoke_dit.yaml  # smoke del pipeline DiT
pytest tests/ -q
python tools/tiiac_check.py .                  # cuando exista el checker
```

`--smoke` selecciona `configs/smoke.yaml`: usa solo 64 imágenes, un modelo
pequeño, 10 pasos de difusión y como máximo 10 batches. Sirve para comprobar
que carga datos, construye el U-Net, entrena, evalúa y escribe un checkpoint;
no sirve para medir calidad generativa.

El entrenamiento completo usa `configs/default.yaml`. Los hiperparámetros se
cambian ahí, no en el código. `dispositivo: auto` elige CUDA si está
disponible y CPU en caso contrario.

Para el DiT hay configuraciones propias: `configs/dit_cifar10.yaml`,
`configs/guided_dit_cifar10.yaml`, `configs/dit_mnist.yaml` y
`configs/guided_dit_mnist.yaml`. Las variantes `guided_*` entrenan con clase
nula y CFG. `configs/smoke_dit.yaml` ejecuta el pipeline DiT en miniatura.

Los checkpoints se guardan en `checkpoints/<experimento.nombre>.pt`. El fichero
contiene los pesos normales, los pesos EMA y la configuración usada. La carpeta
está ignorada por Git. Cuando W&B está activo, el checkpoint final también se
sube como Artifact de tipo `model`; las métricas se registran por época.
Configura `wandb.entidad`, deja `wandb.modo: online` y ejecuta `wandb login`
antes del entrenamiento. Para smoke/tests usa `wandb.modo: disabled`.

El entorno de desarrollo validado usa `torch==2.14.1+cu130` sobre una GPU
NVIDIA RTX 5070. La instalación de esa versión puede requerir el índice de
paquetes CUDA que corresponda a la máquina; no sustituirla automáticamente por
Torch CPU.

## Convenciones

- Sin rutas absolutas, nunca.
- Toda función pública lleva docstring y anotaciones de tipo.
- **La normalización usa estadísticas de entrenamiento, no del conjunto completo.**
  Lo contrario es una fuga de datos: no da error e infla la métrica.
- El `data augmentation` se aplica a train, jamás a validación ni a test.
- La construcción del modelo vive en `src/models/factory.py`: selecciona U-Net
  o DiT según `modelo.arquitectura` (o un token `dit` en `modelo.nombre`).
  `src/app/` importa esa factoría, nunca `src/training/`.
- El DiT se configura con `hidden_size`, `depth`, `num_heads`, `patch_size`,
  `mlp_ratio` y `frequency_embedding_size` (256 por defecto). `hidden_size`
  debe ser divisible por `num_heads` y por 4
  (posiciones sinusoidales); `patch_size` debe dividir el lado de la imagen
  (28 en MNIST, 32 en CIFAR-10).
- `src/app/` no importa nada de `src/training/`: el aplicativo solo carga un
  checkpoint y responde.

## Lo que NO hay que tocar

- `src/data/splits.py`: los splits están congelados desde el hito H3.
- No reintroducir `torchvision` en el loader actual: CIFAR-10 se lee con
  Pillow para evitar dependencia de backend.

## Estado actual

- Hecho: DDPM/DDIM y CFG para CIFAR-10 y MNIST con U-Net; entrenamiento,
  evaluación y checkpoint EMA; logging de métricas y Artifacts de W&B.
- Hecho: DiT con patchify, posiciones sinusoidales, bloques adaLN-Zero,
  condicionamiento por clase, clase nula y CFG; configs para ambos datasets,
  smoke propio y tests unitarios/integración.
- En curso: redactar tests funcionales del pipeline y completar CI.
- Pendiente: generar muestras durante evaluación en todas las configuraciones;
  completar `src/app`; revisar los bugs conocidos de configs incondicionales y
  del evaluador MNIST; al modificar los módulos `*_demo.py`, renombrarlos a la
  hora de importarlos en otros ficheros.

## Para añadir una arquitectura

Añadir el módulo en `src/models/`, exponer un constructor con la misma interfaz
`forward(x, t, y=None)` y el atributo `num_classes`, registrarlo en
`src/models/factory.py` y crear una configuración con
`modelo.arquitectura: <nombre>`. Antes de entrenar, añadir tests de forma para
MNIST y CIFAR-10, condicionamiento, CFG y sampling.

Para CFG, el modelo debe recibir `y` opcional, reservar una clase null y
aplicar dropout de condición durante entrenamiento. En sampling hay que
calcular predicción condicional e incondicional y combinarlas con la escala
CFG configurada.
