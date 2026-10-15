"""Carga de datos y splits (solo PyTorch / torchvision, sin TensorFlow).

REGLAS QUE NO SE ROMPEN (y que el revisor del PR comprueba):
  1. El split se hace ANTES de cualquier transformacion ajustada a los datos.
  2. CIFAR-10 se normaliza con media y desviacion (0.5, 0.5, 0.5),
      dejando los tensores en el rango esperado por DDPM: [-1, 1].
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

import csv
import random
import struct
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset


class _Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, value):
        for transform in self.transforms:
            value = transform(value)
        return value


class _ToTensor:
    def __call__(self, image: Image.Image) -> torch.Tensor:
        values = torch.tensor(list(image.getdata()), dtype=torch.float32)
        channels = len(image.getbands())
        return values.reshape(image.height, image.width, channels).permute(2, 0, 1) / 255.0


class _Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.mean) / self.std


class _RandomHorizontalFlip:
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image


class _RandomCrop:
    def __init__(self, size: int, padding: int):
        self.size = size
        self.padding = padding

    def __call__(self, image: Image.Image) -> Image.Image:
        padded = Image.new(
            image.mode,
            (image.width + 2 * self.padding, image.height + 2 * self.padding),
        )
        padded.paste(image, (self.padding, self.padding))
        left = random.randint(0, padded.width - self.size)
        top = random.randint(0, padded.height - self.size)
        return padded.crop((left, top, left + self.size, top + self.size))


class LocalCIFAR10(Dataset):
    """CIFAR-10 PNG export used by this project, with labels from CSV."""

    def __init__(self, root: str | Path, train: bool, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.image_dir = self.root / ("train" if train else "test")
        self.files = sorted(self.image_dir.glob("*.png"), key=lambda path: int(path.stem))
        self.labels = {}
        if train:
            with (self.root / "trainLabels.csv").open(newline="", encoding="utf-8") as file:
                self.labels = {int(row["id"]): row["label"] for row in csv.DictReader(file)}
            classes = sorted(set(self.labels.values()))
            self.class_to_idx = {name: index for index, name in enumerate(classes)}

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.class_to_idx[self.labels[int(path.stem)]] if self.labels else -1
        return image, label

class LocalMNIST(Dataset):
    """Local MNIST dataset backed by the standard uncompressed IDX files."""

    _FILES = {
        True: ("train-images.idx3-ubyte", "train-labels.idx1-ubyte"),
        False: ("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte"),
    }

    def __init__(self, root: str | Path, train: bool, transform=None):
        self.root = Path(root)
        self.transform = transform
        image_name, label_name = self._FILES[train]
        self.images = self._read_images(self.root / image_name)
        self.labels = self._read_labels(self.root / label_name, len(self.images))

    @staticmethod
    def _read_images(path: Path) -> torch.Tensor:
        data = path.read_bytes()
        if len(data) < 16:
            raise ValueError(f"MNIST image file is too short: {path}")
        magic, count, rows, columns = struct.unpack(">IIII", data[:16])
        if magic != 2051 or rows != 28 or columns != 28:
            raise ValueError(f"invalid MNIST image header in {path}")
        expected = 16 + count * rows * columns
        if len(data) != expected:
            raise ValueError(f"MNIST image count does not match file size: {path}")
        return torch.frombuffer(bytearray(data), dtype=torch.uint8, offset=16).clone().reshape(
            count, rows, columns
        )

    @staticmethod
    def _read_labels(path: Path, expected_count: int) -> torch.Tensor:
        data = path.read_bytes()
        if len(data) < 8:
            raise ValueError(f"MNIST label file is too short: {path}")
        magic, count = struct.unpack(">II", data[:8])
        if magic != 2049 or count != expected_count or len(data) != 8 + count:
            raise ValueError(f"invalid MNIST label file: {path}")
        labels = torch.frombuffer(bytearray(data), dtype=torch.uint8, offset=8).clone()
        if torch.any(labels > 9):
            raise ValueError(f"MNIST labels must be in [0, 9]: {path}")
        return labels.to(torch.long)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index].numpy(), mode="L")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index].item()


_DATASETS = {"mnist": (LocalMNIST, 28, 1), "cifar10": (LocalCIFAR10, 32, 3)}


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


def _transforms(nombre: str, augment: bool, mean=None, std=None):
    """Devuelve (transform_train, transform_eval). El augmentation solo va en train."""
    _, size, channels = _DATASETS[nombre]
    mean = (0.5,) * channels if mean is None else mean
    std = (0.5,) * channels if std is None else std
    base = [_ToTensor(), _Normalize(mean, std)]

    aug = []
    if augment:
        aug.append(_RandomCrop(size, padding=size // 8))
        if nombre == "cifar10":          # un volteo cambiaria el significado de un digito
            aug.append(_RandomHorizontalFlip())

    return _Compose(aug + base), _Compose(base)


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
        ruta = dcfg.get("ruta", dcfg.get("raiz", "./data"))
        seed = dcfg.get("seed", 0)

        # Regla 1: primero el split (solo indices, no se ajusta nada todavia).
        crudo = cls(ruta, train=True, transform=_ToTensor())
        train_idx, val_idx = _split_indices(len(crudo), dcfg.get("val_frac", 0.1), seed)

        if nombre == "mnist":
            mean, std = _estadisticas(crudo, train_idx)
        else:
            mean, std = None, None
        tf_train, tf_eval = _transforms(nombre, dcfg.get("augment", False), mean, std)

        # Regla 3: dos vistas del mismo set, cada una con su transform.
        # Train lleva augmentation; val y test, no.
        train_ds = Subset(cls(ruta, train=True, transform=tf_train), train_idx)
        val_ds = Subset(cls(ruta, train=True, transform=tf_eval), val_idx)
        test_ds = cls(ruta, train=False, transform=tf_eval)
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
                 num_workers=dcfg.get("num_workers", 0),
                 pin_memory=dcfg.get("pin_memory", False))
    return (DataLoader(train_ds, shuffle=True, **comun),
            DataLoader(val_ds, shuffle=False, **comun))


def construir_test_loader(cfg: dict) -> DataLoader:
    """Loader de test. Usalo una sola vez, al final, no para elegir hiperparametros."""
    dcfg = cfg["datos"]
    _, _, test_ds = construir_datasets(cfg)
    if test_ds is None:
        raise ValueError("El dataset 'placeholder' no tiene conjunto de test.")
    return DataLoader(test_ds, shuffle=False, batch_size=dcfg["batch_size"],
                      num_workers=dcfg.get("num_workers", 0),
                      pin_memory=dcfg.get("pin_memory", False))