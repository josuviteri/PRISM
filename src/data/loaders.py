"""Carga de datos y splits (solo PyTorch / torchvision, sin TensorFlow).

REGLAS QUE NO SE ROMPEN (y que el revisor del PR comprueba):
  1. El split se hace ANTES de cualquier transformacion ajustada a los datos.
  2. La normalizacion usa estadisticas de ENTRENAMIENTO, nunca del conjunto
     completo. Lo contrario es una fuga: no da error e infla tu metrica.
  3. El data augmentation se aplica a train, jamas a validacion ni a test.

Configuracion esperada (cfg["datos"]):
    nombre       : "placeholder" | "mnist" | "cifar10"   (def. "placeholder")
    ruta         : carpeta de descarga/cache              (def. "./data")
    batch_size   : int
    num_workers  : int                                    (def. 0)
    val_frac     : fraccion de train para validacion      (def. 0.1)
    seed         : semilla del split                      (def. 0)
    augment      : bool, solo afecta a train              (def. False)
    subset       : int opcional, recorta SOLO train
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# Dataset -> (clase torchvision, tamano de imagen, canales)
_DATASETS = {
    "mnist": (datasets.MNIST, 28, 1),
    "cifar10": (datasets.CIFAR10, 32, 3),
}


class DatasetPlaceholder(Dataset):
    """Dataset sintetico para que `python run.py --smoke` funcione desde el
    minuto cero. Sustituyelo por el dataset de tu paper cuando toque.

    OJO: train y val deben usar semillas distintas; con la misma semilla
    serian datos identicos y la validacion no valdria nada.
    """

    def __init__(self, n: int = 512, dim: int = 32, clases: int = 10, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, dim, generator=g)
        self.y = torch.randint(0, clases, (n,), generator=g)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def _estadisticas(ds_crudo: Dataset, indices, batch_size: int = 1000):
    """Media y desviacion por canal calculadas SOLO sobre `indices` (train).

    Acumula sumas por lotes para no cargar todo el dataset en memoria.
    `ds_crudo` debe devolver tensores (C, H, W) en [0, 1] sin normalizar.
    """
    loader = DataLoader(Subset(ds_crudo, indices), batch_size=batch_size)
    suma = sumsq = 0.0
    n_pix = 0
    for x, _ in loader:
        suma = suma + x.sum(dim=(0, 2, 3))
        sumsq = sumsq + (x ** 2).sum(dim=(0, 2, 3))
        n_pix += x.shape[0] * x.shape[2] * x.shape[3]
    media = suma / n_pix
    std = (sumsq / n_pix - media ** 2).clamp_min(1e-12).sqrt()
    return media.tolist(), std.tolist()


def _transforms(nombre: str, media, std, augment: bool):
    """Devuelve (transform_train, transform_eval). El augmentation solo va en train."""
    _, size, _ = _DATASETS[nombre]
    base = [transforms.ToTensor(), transforms.Normalize(media, std)]

    aug = []
    if augment:
        aug.append(transforms.RandomCrop(size, padding=size // 8))
        if nombre == "cifar10":          # un volteo cambiaria el significado de un digito
            aug.append(transforms.RandomHorizontalFlip())

    return transforms.Compose(aug + base), transforms.Compose(base)


def _split_indices(n: int, val_frac: float, seed: int):
    """Split reproducible train/val sobre los indices del set de entrenamiento."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = int(round(n * val_frac))
    return perm[n_val:], perm[:n_val]       # (train_idx, val_idx)


# --------------------------------------------------------------------------
# API publica
# --------------------------------------------------------------------------
def construir_datasets(cfg: dict):
    """Devuelve (train_ds, val_ds, test_ds). test_ds es None para el placeholder."""
    dcfg = cfg["datos"]
    nombre = dcfg.get("nombre", "placeholder")

    if nombre == "placeholder":
        train_ds = DatasetPlaceholder(seed=0)
        val_ds = DatasetPlaceholder(n=128, seed=1)
        test_ds = None
    elif nombre in _DATASETS:
        cls, _, _ = _DATASETS[nombre]
        ruta = dcfg.get("ruta", "./data")
        seed = dcfg.get("seed", 0)

        # Regla 1: primero el split (solo indices, no se ajusta nada todavia).
        crudo = cls(ruta, train=True, download=True, transform=transforms.ToTensor())
        train_idx, val_idx = _split_indices(len(crudo), dcfg.get("val_frac", 0.1), seed)

        # Regla 2: estadisticas SOLO con los indices de train.
        media, std = _estadisticas(crudo, train_idx)
        tf_train, tf_eval = _transforms(nombre, media, std, dcfg.get("augment", False))

        # Regla 3: dos vistas del mismo set, cada una con su transform.
        # Train lleva augmentation; val y test, no.
        train_ds = Subset(cls(ruta, train=True, transform=tf_train), train_idx)
        val_ds = Subset(cls(ruta, train=True, transform=tf_eval), val_idx)
        test_ds = cls(ruta, train=False, download=True, transform=tf_eval)
    else:
        raise ValueError(
            f"Dataset '{nombre}' no disponible. Opciones: placeholder, {', '.join(_DATASETS)}"
        )

    # El subset se aplica DESPUES del split y solo a train, para no cambiar
    # la distribucion de validacion.
    if dcfg.get("subset"):
        n = min(int(dcfg["subset"]), len(train_ds))
        train_ds = Subset(train_ds, range(n))

    return train_ds, val_ds, test_ds


def construir_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """Devuelve (train_loader, val_loader) segun la configuracion."""
    dcfg = cfg["datos"]
    train_ds, val_ds, _ = construir_datasets(cfg)

    comun = dict(batch_size=dcfg["batch_size"],
                 num_workers=dcfg.get("num_workers", 0))
    return (DataLoader(train_ds, shuffle=True, **comun),
            DataLoader(val_ds, shuffle=False, **comun))


def construir_test_loader(cfg: dict) -> DataLoader:
    """Loader de test. Usalo una sola vez, al final, no para elegir hiperparametros."""
    dcfg = cfg["datos"]
    _, _, test_ds = construir_datasets(cfg)
    if test_ds is None:
        raise ValueError("El dataset 'placeholder' no tiene conjunto de test.")
    return DataLoader(test_ds, shuffle=False, batch_size=dcfg["batch_size"],
                      num_workers=dcfg.get("num_workers", 0))