"""Diffusion Transformer (DiT) with optional class conditioning for CFG.

The module follows the DiT design of Peebles & Xie (2023): images are split
into non-overlapping patches, the patch tokens go through a stack of
transformer blocks modulated with adaLN-Zero, and a final layer projects each
token back to its patch. Timestep and class embeddings are summed into a single
conditioning vector, so the same blocks support unconditional and
class-conditional training.

The public interface matches ``src.models.unet.UNet``: ``forward(x, t, y=None)``
returns noise with the same shape as ``x`` and ``num_classes`` is exposed for
classifier-free guidance. ``y`` may be omitted for conditional models, in which
case the reserved null class ``num_classes`` is used.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def timestep_embedding(timesteps: Tensor, channels: int) -> Tensor:
  """Sinusoidal timestep embedding, identical in spirit to the U-Net version."""
  half = channels // 2
  frequencies = torch.exp(
    -math.log(10000)
    * torch.arange(half, device=timesteps.device, dtype=torch.float32)
    / half
  )
  values = timesteps.float()[:, None] * frequencies[None]
  embedding = torch.cat([torch.sin(values), torch.cos(values)], dim=1)
  return F.pad(embedding, (0, channels - embedding.shape[1]))


def sincos_position_embedding(
  height: int, width: int, channels: int, device: torch.device
) -> Tensor:
  """Fixed 2-D sine/cosine positions for a ``height x width`` patch grid.

  DiT uses fixed positions instead of learned ones, which keeps the checkpoint
  independent of the training image resolution. ``channels`` must be divisible
  by four so the grid can be split into two sinusoidal halves.
  """
  if channels % 4 != 0:
    raise ValueError("position embedding channels must be divisible by 4")
  quarter = channels // 4
  omega = torch.arange(quarter, device=device, dtype=torch.float32) / quarter
  omega = 1.0 / (10000.0 ** omega)
  grid_h = torch.arange(height, device=device, dtype=torch.float32)
  grid_w = torch.arange(width, device=device, dtype=torch.float32)
  angles_h = grid_h[:, None] * omega[None, :]
  angles_w = grid_w[:, None] * omega[None, :]
  embedding_h = torch.cat([torch.sin(angles_h), torch.cos(angles_h)], dim=1)
  embedding_w = torch.cat([torch.sin(angles_w), torch.cos(angles_w)], dim=1)
  embedding = torch.cat([
    embedding_h[:, None, :].expand(height, width, channels // 2),
    embedding_w[None, :, :].expand(height, width, channels // 2),
  ], dim=-1)
  return embedding.reshape(height * width, channels)


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
  """Applies adaLN shift/scale to a ``(B, N, C)`` token sequence."""
  return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class PatchEmbed(nn.Module):
  """Splits an image into patches and projects each patch to ``hidden_size``."""

  def __init__(self, in_ch: int, hidden_size: int, patch_size: int):
    super().__init__()
    self.patch_size = patch_size
    self.proj = nn.Conv2d(in_ch, hidden_size, kernel_size=patch_size, stride=patch_size)

  def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
    height, width = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
    tokens = self.proj(x).flatten(2).transpose(1, 2)
    return tokens, height, width


class Attention(nn.Module):
  """Multi-head self-attention over patch tokens."""

  def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
    super().__init__()
    self.num_heads = num_heads
    self.head_dim = hidden_size // num_heads
    self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
    self.proj = nn.Linear(hidden_size, hidden_size)
    self.dropout = dropout

  def forward(self, x: Tensor) -> Tensor:
    batch, tokens, channels = x.shape
    qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    query, key, value = qkv[0], qkv[1], qkv[2]
    attended = F.scaled_dot_product_attention(
      query, key, value, dropout_p=self.dropout if self.training else 0.0
    )
    attended = attended.transpose(1, 2).reshape(batch, tokens, channels)
    return self.proj(attended)


class MLP(nn.Module):
  """Transformer feed-forward network with GELU and dropout."""

  def __init__(self, hidden_size: int, mlp_ratio: float, dropout: float = 0.0):
    super().__init__()
    inner_size = int(hidden_size * mlp_ratio)
    self.fc1 = nn.Linear(hidden_size, inner_size)
    self.act = nn.GELU(approximate="tanh")
    self.fc2 = nn.Linear(inner_size, hidden_size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: Tensor) -> Tensor:
    return self.dropout(self.fc2(self.dropout(self.act(self.fc1(x)))))


class DiTBlock(nn.Module):
  """Transformer block with adaLN-Zero conditioning."""

  def __init__(
    self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float = 0.0
  ):
    super().__init__()
    self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.attn = Attention(hidden_size, num_heads, dropout)
    self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.mlp = MLP(hidden_size, mlp_ratio, dropout)
    self.adaLN_modulation = nn.Sequential(
      nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size)
    )
    nn.init.zeros_(self.adaLN_modulation[1].weight)
    nn.init.zeros_(self.adaLN_modulation[1].bias)

  def forward(self, x: Tensor, condition: Tensor) -> Tensor:
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
      self.adaLN_modulation(condition).chunk(6, dim=-1)
    )
    x = x + gate_msa[:, None, :] * self.attn(
      _modulate(self.norm1(x), shift_msa, scale_msa)
    )
    x = x + gate_mlp[:, None, :] * self.mlp(
      _modulate(self.norm2(x), shift_mlp, scale_mlp)
    )
    return x


class FinalLayer(nn.Module):
  """Final adaLN layer that projects tokens back to pixel patches."""

  def __init__(self, hidden_size: int, patch_size: int, out_ch: int):
    super().__init__()
    self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_ch)
    self.adaLN_modulation = nn.Sequential(
      nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size)
    )
    nn.init.zeros_(self.adaLN_modulation[1].weight)
    nn.init.zeros_(self.adaLN_modulation[1].bias)
    nn.init.zeros_(self.linear.weight)
    nn.init.zeros_(self.linear.bias)

  def forward(self, x: Tensor, condition: Tensor) -> Tensor:
    shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
    return self.linear(_modulate(self.norm_final(x), shift, scale))


class DiT(nn.Module):
  """Diffusion Transformer for ``(B, C, H, W)`` images in ``[-1, 1]``.

  ``H`` and ``W`` must be divisible by ``patch_size``. The model is
  class-conditional when ``num_classes > 1``; the reserved null class is
  ``num_classes`` and is selected automatically when ``y`` is omitted.
  """

  def __init__(
    self,
    *,
    in_ch: int = 3,
    out_ch: int = 3,
    hidden_size: int = 256,
    depth: int = 6,
    num_heads: int = 4,
    patch_size: int = 2,
    mlp_ratio: float = 4.0,
    dropout: float = 0.0,
    num_classes: int = 1,
    frequency_embedding_size: int = 256,
  ):
    super().__init__()
    if num_classes < 1:
      raise ValueError("num_classes must be positive")
    if frequency_embedding_size < 2:
      raise ValueError("frequency_embedding_size must be at least 2")
    if patch_size < 1:
      raise ValueError("patch_size must be positive")
    if depth < 1:
      raise ValueError("depth must be positive")
    if mlp_ratio <= 0:
      raise ValueError("mlp_ratio must be positive")
    if num_heads < 1 or hidden_size % num_heads != 0:
      raise ValueError("hidden_size must be divisible by num_heads")
    if hidden_size < 4 or hidden_size % 4 != 0:
      raise ValueError(
        "hidden_size must be a positive multiple of 4 for positional embeddings"
      )

    self.num_classes = num_classes
    self.in_ch = in_ch
    self.out_ch = out_ch
    self.hidden_size = hidden_size
    self.patch_size = patch_size
    self.frequency_embedding_size = frequency_embedding_size

    self.patch_embed = PatchEmbed(in_ch, hidden_size, patch_size)
    self.temb = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size),
    )
    self.class_embedding = (
      nn.Embedding(num_classes + 1, hidden_size) if num_classes > 1 else None
    )
    self.blocks = nn.ModuleList([
      DiTBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(depth)
    ])
    self.final_layer = FinalLayer(hidden_size, patch_size, out_ch)
    self._initialize_weights()

  def _initialize_weights(self) -> None:
    """DiT-style init: Xavier linears, small embeddings, zero adaLN/final output."""
    for module in self.modules():
      if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)
    # Zero-init is applied after the generic pass so the residual branches start
    # as identity and the model predicts zero noise at initialisation.
    for block in self.blocks:
      nn.init.zeros_(block.adaLN_modulation[1].weight)
      nn.init.zeros_(block.adaLN_modulation[1].bias)
    nn.init.zeros_(self.final_layer.adaLN_modulation[1].weight)
    nn.init.zeros_(self.final_layer.adaLN_modulation[1].bias)
    nn.init.zeros_(self.final_layer.linear.weight)
    nn.init.zeros_(self.final_layer.linear.bias)

  def _class_condition(self, x: Tensor, y: Optional[Tensor]) -> Optional[Tensor]:
    """Validates labels and returns their embedding, or ``None`` if unconditional."""
    if self.class_embedding is None:
      return None
    null_label = self.num_classes
    if y is None:
      y = torch.full((x.shape[0],), null_label, device=x.device, dtype=torch.long)
    elif y.ndim != 1 or y.shape[0] != x.shape[0]:
      raise ValueError("expected y with shape (B,)")
    elif y.dtype != torch.long:
      raise ValueError("class labels must have dtype torch.long")
    if torch.any((y < 0) | (y > null_label)):
      raise ValueError(f"class labels must be in [0, {null_label}]")
    return self.class_embedding(y)

  def forward(self, x: Tensor, t: Tensor, y: Optional[Tensor] = None) -> Tensor:
    if x.ndim != 4 or t.ndim != 1 or x.shape[0] != t.shape[0]:
      raise ValueError("expected x with shape (B, C, H, W) and t with shape (B,)")
    if x.shape[-2] % self.patch_size != 0 or x.shape[-1] % self.patch_size != 0:
      raise ValueError(
        f"image height and width must be divisible by patch_size={self.patch_size}"
      )

    condition = self.temb(timestep_embedding(t, self.frequency_embedding_size))
    class_condition = self._class_condition(x, y)
    if class_condition is not None:
      condition = condition + class_condition

    tokens, height, width = self.patch_embed(x)
    position = sincos_position_embedding(
      height, width, self.hidden_size, x.device
    ).to(dtype=tokens.dtype)
    tokens = tokens + position
    for block in self.blocks:
      tokens = block(tokens, condition)
    tokens = self.final_layer(tokens, condition)
    return self.unpatchify(tokens, height, width)

  def unpatchify(self, tokens: Tensor, height: int, width: int) -> Tensor:
    """Rearranges ``(B, H*W, patch*patch*C)`` tokens back to ``(B, C, H, W)``."""
    patch = self.patch_size
    batch = tokens.shape[0]
    tokens = tokens.reshape(batch, height, width, patch, patch, self.out_ch)
    return tokens.permute(0, 5, 1, 3, 2, 4).reshape(
      batch, self.out_ch, height * patch, width * patch
    )


def model(
  *,
  in_ch: int = 3,
  out_ch: int = 3,
  hidden_size: int = 256,
  depth: int = 6,
  num_heads: int = 4,
  patch_size: int = 2,
  mlp_ratio: float = 4.0,
  dropout: float = 0.0,
  num_classes: int = 1,
  frequency_embedding_size: int = 256,
  **kwargs,
) -> DiT:
  """Builds a DiT, ignoring U-Net-only keys so configs can be shared."""
  del kwargs
  return DiT(
    in_ch=in_ch,
    out_ch=out_ch,
    hidden_size=hidden_size,
    depth=depth,
    num_heads=num_heads,
    patch_size=patch_size,
    mlp_ratio=mlp_ratio,
    dropout=dropout,
    num_classes=num_classes,
    frequency_embedding_size=frequency_embedding_size,
  )
