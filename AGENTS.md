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

PRISM replica DDPM y prepara una ampliación con Classifier-Free Guidance (CFG)
para MNIST y CIFAR-10. La base ejecutable actual es un DDPM PyTorch
incondicional para CIFAR-10 usando las imágenes PNG y `trainLabels.csv` locales.
MNIST y el conditioning por clase todavía están pendientes.

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
- `src/app/` no importa nada de `src/training/`: el aplicativo solo carga un
  checkpoint y responde. El servicio ejecutable todavía necesita migrarse de
  `serve_demo.py` al checkpoint DDPM actual.

## Lo que NO hay que tocar

- `src/data/splits.py`: los splits están congelados desde el hito H3.
- No reintroducir `torchvision` en el loader actual: CIFAR-10 se lee con
  Pillow para evitar dependencia de backend.

## Estado actual

- Hecho: entrenamiento, evaluación y checkpoint EMA de CIFAR-10 en PyTorch;
  smoke validado en GPU; logging de métricas y Artifacts de W&B preparado.
- En curso: redactar tests funcionales del pipeline y completar CI.
- Pendiente: implementar loader/modelo/loop de MNIST; añadir etiquetas de clase
  al U-Net y al loop para CFG; generar muestras durante evaluación; completar
  `src/app`; al modificar los módulos `*_demo.py`, renombrarlos a la hora de
  importarlos en otros ficheros .

## Para añadir MNIST

Implementar en `src/data/loaders.py` el dataset local IDX o su exportación PNG,
con imágenes `(1, 28, 28)`, y eliminar el `NotImplementedError` de
`construir_datasets`. Después añadir una configuración `configs/mnist.yaml`, un
constructor de modelo con `in_ch=1`, y un loop equivalente que comparta la
difusión, EMA, evaluación, checkpoint y tracking. Añadir primero tests de
forma, split, normalización y un smoke MNIST antes de entrenar.

Para CFG, el modelo debe recibir `y` opcional, reservar una clase null y
aplicar dropout de condición durante entrenamiento. En sampling hay que
calcular predicción condicional e incondicional y combinarlas con la escala
CFG configurada. Es una extensión posterior, no está implementada en el
checkpoint CIFAR actual.
