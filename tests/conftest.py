"""Small local datasets used by tests; CI does not need downloaded data."""

from __future__ import annotations

import csv
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _write_idx(root: Path, prefix: str, images: np.ndarray, labels: np.ndarray) -> None:
    image_name = "train-images.idx3-ubyte" if prefix == "train" else "t10k-images.idx3-ubyte"
    label_name = "train-labels.idx1-ubyte" if prefix == "train" else "t10k-labels.idx1-ubyte"
    (root / image_name).write_bytes(
        struct.pack(">IIII", 2051, len(images), 28, 28) + images.astype(np.uint8).tobytes()
    )
    (root / label_name).write_bytes(
        struct.pack(">II", 2049, len(labels)) + labels.astype(np.uint8).tobytes()
    )


@pytest.fixture
def mnist_root(tmp_path: Path) -> Path:
    """Create a deterministic miniature MNIST IDX directory."""
    root = tmp_path / "mnist"
    root.mkdir()
    train = np.zeros((20, 28, 28), dtype=np.uint8)
    train[:, 7:21, 7:21] = np.arange(20, dtype=np.uint8)[:, None, None] * 10 + 40
    test = np.full((5, 28, 28), 128, dtype=np.uint8)
    _write_idx(root, "train", train, np.arange(20) % 10)
    _write_idx(root, "test", test, np.arange(5) % 10)
    return root


@pytest.fixture
def cifar_root(tmp_path: Path) -> Path:
    """Create a miniature CIFAR PNG export and matching labels CSV."""
    root = tmp_path / "cifar-10"
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir()
    with (root / "trainLabels.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("id", "label"))
        for index in range(1, 21):
            writer.writerow((index, f"class{index % 2}"))
            pixels = np.full((32, 32, 3), index * 10, dtype=np.uint8)
            Image.fromarray(pixels).save(root / "train" / f"{index}.png")
    for index in range(1, 6):
        Image.fromarray(np.full((32, 32, 3), index * 20, dtype=np.uint8)).save(
            root / "test" / f"{index}.png"
        )
    return root
