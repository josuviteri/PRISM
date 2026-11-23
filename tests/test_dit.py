"""Unit and integration tests for the Diffusion Transformer (DiT)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from src.app.diffusion_service import DiffusionExplorer, _build_model, discover_checkpoints
from src.models.dit import DiT
from src.models.unet import UNet
from src.training.diffusion_loop import construir_modelo
from src.utils.diffusion_utils import GaussianDiffusion


def _small_dit(*, channels: int = 3, num_classes: int = 1) -> DiT:
    """Return a tiny DiT that keeps CPU tests fast."""
    return DiT(
        in_ch=channels,
        out_ch=channels,
        hidden_size=32,
        depth=1,
        num_heads=4,
        patch_size=4,
        mlp_ratio=2.0,
        dropout=0.0,
        num_classes=num_classes,
    )


def _activate_conditioning(model: DiT) -> None:
    """Make adaLN and the output head non-zero so class conditioning is observable.

    DiT zero-initialises adaLN-Zero and the final projection on purpose, so at
    initialisation every conditioning path is gated off. Perturbing those
    weights lets a unit test verify the conditioning wiring without training.
    """
    with torch.no_grad():
        for block in model.blocks:
            block.adaLN_modulation[1].weight.normal_(0.0, 0.02)
            block.adaLN_modulation[1].bias.normal_(0.0, 0.02)
        model.final_layer.adaLN_modulation[1].weight.normal_(0.0, 0.02)
        model.final_layer.adaLN_modulation[1].bias.normal_(0.0, 0.02)
        model.final_layer.linear.weight.normal_(0.0, 0.02)
        model.final_layer.linear.bias.normal_(0.0, 0.02)


@pytest.mark.parametrize(("channels", "size"), [(3, 32), (1, 28)])
def test_dit_preserves_image_shape_and_is_differentiable(channels, size):
    """CIFAR-10 and MNIST shapes survive patchify/unpatchify and backprop."""
    model = _small_dit(channels=channels)
    images = torch.randn(2, channels, size, size)
    timesteps = torch.tensor([0, 3])
    output = model(images, timesteps)
    assert output.shape == images.shape
    assert torch.isfinite(output).all()

    loss = (output - images).pow(2).mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_dit_rejects_images_not_divisible_by_patch_size():
    """A patch grid must cover the full image without cropping or padding."""
    model = _small_dit(channels=1)
    with pytest.raises(ValueError, match="divisible by patch_size"):
        model(torch.randn(1, 1, 30, 30), torch.zeros(1, dtype=torch.long))


def test_dit_uses_the_null_class_when_labels_are_omitted():
    """Conditional DiT maps ``y=None`` to the reserved null class."""
    model = _small_dit(num_classes=10)
    _activate_conditioning(model)
    images = torch.randn(2, 3, 32, 32)
    timesteps = torch.zeros(2, dtype=torch.long)

    implicit = model(images, timesteps)
    explicit = model(images, timesteps, torch.full((2,), 10, dtype=torch.long))
    assert torch.allclose(implicit, explicit)


def test_dit_class_conditioning_changes_the_prediction():
    """Different labels reach the transformer through adaLN conditioning."""
    model = _small_dit(num_classes=10)
    _activate_conditioning(model)
    images = torch.randn(2, 3, 32, 32)
    timesteps = torch.zeros(2, dtype=torch.long)

    first = model(images, timesteps, torch.tensor([0, 0]))
    second = model(images, timesteps, torch.tensor([1, 1]))
    assert not torch.allclose(first, second)


def test_dit_validates_class_labels():
    """Out-of-range and non-long labels fail with the same contract as the U-Net."""
    model = _small_dit(num_classes=10)
    images = torch.randn(2, 3, 32, 32)
    timesteps = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match=r"class labels must be in"):
        model(images, timesteps, torch.tensor([0, 11]))
    with pytest.raises(ValueError, match="dtype torch.long"):
        model(images, timesteps, torch.tensor([0.0, 1.0]))


def test_dit_supports_classifier_free_guidance_prediction():
    """CFG combines the conditional and null predictions exactly as expected."""
    model = _small_dit(channels=1, num_classes=3)
    _activate_conditioning(model)
    diffusion = GaussianDiffusion(2, 0.01, 0.02, torch.device("cpu"))
    images = torch.randn(2, 1, 8, 8)
    timesteps = torch.zeros(2, dtype=torch.long)
    labels = torch.tensor([0, 1])

    prediction = diffusion._predict_noise(
        model, images, timesteps, labels, guidance_scale=2.0
    )
    conditional = model(images, timesteps, labels)
    unconditional = model(images, timesteps, torch.full_like(labels, 3))
    expected = unconditional + 2.0 * (conditional - unconditional)
    assert torch.allclose(prediction, expected)
    assert torch.isfinite(prediction).all()


def test_dit_training_loss_accepts_label_dropout():
    """The shared DDPM loss can train a conditional DiT with null-class dropout."""
    model = _small_dit(channels=1, num_classes=3)
    diffusion = GaussianDiffusion(2, 0.01, 0.02, torch.device("cpu"))
    images = torch.randn(4, 1, 8, 8)
    labels = torch.tensor([0, 1, 2, 0])
    loss = diffusion.loss(
        model,
        images,
        generator=torch.Generator().manual_seed(5),
        labels=labels,
        p_uncond=0.5,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_dit_can_be_sampled_with_ddim_and_guidance():
    """The shared sampler works with a DiT denoiser without special cases."""
    model = _small_dit(channels=1, num_classes=3)
    _activate_conditioning(model)
    diffusion = GaussianDiffusion(2, 0.01, 0.02, torch.device("cpu"))
    generator = torch.Generator().manual_seed(7)
    samples = diffusion.sample_ddim(
        model,
        (2, 1, 8, 8),
        sampling_steps=2,
        generator=generator,
        labels=torch.tensor([0, 1]),
        guidance_scale=2.0,
    )
    assert samples.shape == (2, 1, 8, 8)
    assert torch.isfinite(samples).all()
    assert samples.min() >= -1.0 and samples.max() <= 1.0


def test_model_dispatch_keeps_unet_as_default():
    """Configs without an architecture key keep building the existing U-Net."""
    config = {
        "datos": {"nombre": "mnist"},
        "modelo": {
            "nombre": "ddpm_mnist",
            "base_channels": 8,
            "channel_multipliers": [1],
            "residual_blocks": 1,
            "attention_resolutions": [],
            "dropout": 0.0,
        },
    }
    assert isinstance(construir_modelo(config), UNet)


@pytest.mark.parametrize(
    ("dataset", "channels", "size", "name"),
    [("mnist", 1, 28, "dit_mnist"), ("cifar10", 3, 32, "guided_dit_cifar10")],
)
def test_model_dispatch_builds_dit_for_both_datasets(dataset, channels, size, name):
    """Explicit architecture selection builds DiT for both image formats."""
    config = {
        "datos": {"nombre": dataset},
        "modelo": {
            "nombre": name,
            "arquitectura": "dit",
            "hidden_size": 32,
            "depth": 1,
            "num_heads": 4,
            "patch_size": 2,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "num_classes": 10,
        },
    }
    model = construir_modelo(config)
    assert isinstance(model, DiT)
    assert model.num_classes == 10
    output = model(
        torch.randn(2, channels, size, size),
        torch.zeros(2, dtype=torch.long),
        torch.tensor([0, 1]),
    )
    assert output.shape == (2, channels, size, size)


def test_model_dispatch_recognizes_dit_in_the_model_name():
    """A ``dit`` token in ``modelo.nombre`` selects DiT without an explicit key."""
    config = {
        "datos": {"nombre": "mnist"},
        "modelo": {
            "nombre": "guided_dit_mnist",
            "hidden_size": 32,
            "depth": 1,
            "num_heads": 4,
            "patch_size": 2,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "num_classes": 10,
        },
    }
    assert isinstance(construir_modelo(config), DiT)


def test_app_builds_and_samples_a_dit_checkpoint(tmp_path: Path):
    """The checkpoint-only app can load and sample a DiT without training imports."""
    config = {
        "datos": {"nombre": "cifar10"},
        "modelo": {
            "nombre": "dit_cifar10",
            "arquitectura": "dit",
            "hidden_size": 32,
            "depth": 1,
            "num_heads": 4,
            "patch_size": 4,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "num_classes": 10,
            "diffusion_steps": 2,
            "beta_start": 0.0001,
            "beta_end": 0.02,
        },
        "entrenamiento": {"p_uncond": 0.1, "guidance_scale": 2.0},
    }
    model = _build_model(config)
    assert isinstance(model, DiT)
    checkpoint = tmp_path / "dit.pt"
    torch.save({"ema": model.state_dict(), "config": config}, checkpoint)

    available = discover_checkpoints(tmp_path)
    explorer = DiffusionExplorer(available, torch.device("cpu"))
    noise, samples, trajectory, info, steps = explorer.generate(
        "dit.pt", "ddim", 2, 1, seed=11, label=3, guidance_scale=2.0
    )
    assert noise.size == samples.size == (32, 32)
    assert len(trajectory) == 3
    assert info.num_classes == 10
    assert steps == 2


@pytest.mark.parametrize(
    "config_name",
    ["dit_cifar10", "guided_dit_cifar10", "dit_mnist", "guided_dit_mnist", "smoke_dit"],
)
def test_dit_configs_build_the_expected_architecture(config_name):
    """Every shipped DiT config is loadable and selects the DiT architecture."""
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / f"{config_name}.yaml").read_text(encoding="utf-8"))
    model = construir_modelo(config)
    assert isinstance(model, DiT)
    assert model.num_classes == config["modelo"].get("num_classes", 1)


@pytest.mark.parametrize(
    "config_name",
    ["dit_cifar10", "guided_dit_cifar10", "dit_mnist", "guided_dit_mnist"],
)
def test_full_dit_configs_enable_hard_metrics_during_training(config_name):
    """Non-smoke DiT configs keep the FID/IS cadence added by the hard-metrics feature."""
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / f"{config_name}.yaml").read_text(encoding="utf-8"))
    metrics = config["metricas"]
    assert metrics["cada_epocas"] > 0
    assert metrics["num_samples_intermedias"] >= 2
