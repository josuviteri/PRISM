# CLEAR
### _Probabilistic Reverse-diffusion for Image Synthesis with Multimodal guidance_
_Research project based on DDPM and CFG papers._

Replication  of **Jonathan Ho, Ajay Jain y Pieter Abbeel, 2020 ·  DDPM**, **Jonathan Ho & Tim Salimans**, 2022 · (CFG)**, over **MNIST** and **CIFAR-10** datasets,   
with a web UI for experimental guided generation on top.

- **DDPM Paper:** <https://arxiv.org/pdf/2006.11239>
- **CFG Paper:** <https://arxiv.org/pdf/2207.12598>
- **Reference of DDPM Implementation:** (https://github.com/hojonathanho/diffusion)
- **Experiment Tracking:** <enlace al proyecto de W&B — obligatorio>
- **Authory:** Josu Viteri, Gotzon Viteri

## Resultado

| Métrica | Paper | Nuestra réplica | Escala |
|---|---|---|---|
| <accuracy> | <0.72> | <0.68> | <CIFAR-10, subset de 5.000> |

<Una frase honesta sobre la diferencia. Una desviación bien explicada puntúa
igual que un acierto; una desviación sin explicar, no.>

## Instalación

```bash
git clone <url>
cd <repo>
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Datos

```bash
python scripts/get_data.py
```

Descarga <dataset> de <origen oficial> y verifica el SHA-256. Ocupa <N> GB.
Licencia: <licencia del dataset>.

## Ejecución

```bash
# Smoke test: el pipeline entero en miniatura. Menos de 5 minutos en CPU.
python run.py --smoke

# Entrenamiento completo. Menos de 20 minutos en una T4 de Colab gratuito.
python run.py --config configs/default.yaml

# El aplicativo, sobre el checkpoint entrenado.
python -m src.app.serve --checkpoint checkpoints/mejor.pt
```

## Estructura

```
configs/      Todos los hiperparámetros. Ninguno escrito a fuego en el código
src/data/     Carga, splits y transformaciones
src/models/   La arquitectura replicada
src/training/ Bucle de entrenamiento y evaluación
src/app/      La capa de servicio: carga un checkpoint y responde
scripts/      Descarga de datos y utilidades
tests/        Tests de forma, invariancia y reproducibilidad
run.py        Punto de entrada único
```

## Reproducibilidad

- Semilla fijada en `src/utils/seed.py`, valor en `configs/`.
- Dependencias con versión exacta en `requirements.txt`.
- Presupuesto de cómputo declarado: **<tiempo real> en <hardware>**.
- Qué se recortó respecto al paper y qué se pierde con ello: <explicación>.

## Decisiones donde el paper no especifica

<La sección que distingue replicar de copiar. Lista lo que tuvisteis que
decidir vosotros y con qué criterio. Sale de la ficha de replicación.>

| Ambigüedad del paper | Qué decidimos | Por qué |
|---|---|---|
| <no dice el schedule de lr> | <cosine con warmup de 5 épocas> | <es lo habitual en la familia de trabajos citada> |

## Tests

```bash
pytest tests/ -q
```

## Uso de IA

Documentado en [`AI_USAGE.md`](AI_USAGE.md).
