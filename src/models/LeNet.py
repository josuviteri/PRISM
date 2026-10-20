"""Small MNIST classifier used as the feature extractor for IS and FID."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class LeNet(nn.Module):
    """Improved LeNet-style classifier for one-channel 28x28 MNIST images.

    ``forward_features`` returns the 128-dimensional representation used by
    FID-MNIST.  ``forward`` returns the ten class logits used by IS-MNIST.
    """

    def __init__(
        self,
        channels: Sequence[int] = (32, 64, 128),
        embedding_dim: int = 128,
        num_classes: int = 10,
    ) -> None:
        """Initialise the convolutional feature extractor and digit head."""
        super().__init__()
        if len(channels) != 3:
            raise ValueError("channels must contain exactly three widths")
        if channels[-1] != embedding_dim:
            raise ValueError("the final convolution width must equal embedding_dim")
        if embedding_dim <= 0 or num_classes <= 1:
            raise ValueError("embedding_dim must be positive and num_classes must exceed one")

        first, second, third = channels
        self.features = nn.Sequential(
            nn.Conv2d(1, first, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(first),
            nn.ReLU(inplace=True),
            nn.Conv2d(first, second, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(second),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(second, third, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(third),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.channels = tuple(channels)

    def forward_features(self, images: Tensor) -> Tensor:
        """Return the per-image embedding immediately before the classifier."""
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError("LeNet expects images with shape (N, 1, H, W)")
        return self.pool(self.features(images)).flatten(1)

    def forward(self, images: Tensor) -> Tensor:
        """Return unnormalised logits for the ten MNIST digit classes."""
        return self.classifier(self.forward_features(images))

    def architecture_config(self) -> dict[str, int | list[int]]:
        """Return the architecture metadata persisted with an evaluator checkpoint."""
        return {
            "in_channels": 1,
            "channels": list(self.channels),
            "embedding_dim": self.embedding_dim,
            "num_classes": self.num_classes,
        }


def model(cfg: dict) -> LeNet:
    """Build a LeNet evaluator from the ``modelo`` configuration section."""
    model_cfg = cfg["modelo"]
    if model_cfg.get("nombre") != "lenet_mnist_evaluator":
        raise ValueError("modelo.nombre must be 'lenet_mnist_evaluator'")
    return LeNet(
        channels=tuple(model_cfg["channels"]),
        embedding_dim=model_cfg["embedding_dim"],
        num_classes=model_cfg["num_classes"],
    )
