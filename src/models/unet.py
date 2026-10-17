from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def nonlinearity(x: Tensor) -> Tensor:
  return F.silu(x)


def _group_count(channels: int) -> int:
  return min(32, channels) if channels % min(32, channels) == 0 else 1


class ResnetBlock(nn.Module):
  def __init__(self, in_ch: int, out_ch: int, temb_ch: int, dropout: float, conv_shortcut: bool = False):
    super().__init__()
    self.norm1 = nn.GroupNorm(_group_count(in_ch), in_ch)
    self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
    self.temb_proj = nn.Linear(temb_ch, out_ch)
    self.norm2 = nn.GroupNorm(_group_count(out_ch), out_ch)
    self.dropout = nn.Dropout(dropout)
    self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
    self.skip = (nn.Conv2d(in_ch, out_ch, 3, padding=1) if conv_shortcut
                 else nn.Conv2d(in_ch, out_ch, 1)) if in_ch != out_ch else nn.Identity()
    nn.init.zeros_(self.conv2.weight)
    nn.init.zeros_(self.conv2.bias)

  def forward(self, x: Tensor, temb: Tensor) -> Tensor:
    h = self.conv1(nonlinearity(self.norm1(x)))
    h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
    h = self.conv2(self.dropout(nonlinearity(self.norm2(h))))
    return self.skip(x) + h


class AttentionBlock(nn.Module):
  def __init__(self, channels: int):
    super().__init__()
    self.norm = nn.GroupNorm(_group_count(channels), channels)
    self.q = nn.Conv2d(channels, channels, 1)
    self.k = nn.Conv2d(channels, channels, 1)
    self.v = nn.Conv2d(channels, channels, 1)
    self.proj_out = nn.Conv2d(channels, channels, 1)
    nn.init.zeros_(self.proj_out.weight)
    nn.init.zeros_(self.proj_out.bias)

  def forward(self, x: Tensor) -> Tensor:
    batch, channels, height, width = x.shape
    normalized = self.norm(x)
    q = self.q(normalized).flatten(2).transpose(1, 2)
    k = self.k(normalized).flatten(2).transpose(1, 2)
    v = self.v(normalized).flatten(2).transpose(1, 2)
    # Single-head attention. SDPA's default scale is channels ** -0.5, identical to the
    # previous manual bmm/softmax version, so state_dict keys and results are unchanged.
    h = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])[:, 0]
    h = h.transpose(1, 2).reshape(batch, channels, height, width)
    return x + self.proj_out(h)


class UNet(nn.Module):
  """Unconditional U-Net"""

  def __init__(self, *, in_ch: int = 3, ch: int, out_ch: int,
               ch_mult: Sequence[int] = (1, 2, 4, 8), num_res_blocks: int,
               attn_resolutions: Sequence[int], dropout: float = 0.,
               resamp_with_conv: bool = True):
    super().__init__()
    self.ch_mult = tuple(ch_mult)
    self.attn_resolutions = set(attn_resolutions)
    temb_ch = ch * 4
    self.temb = nn.Sequential(nn.Linear(ch, temb_ch), nn.SiLU(), nn.Linear(temb_ch, temb_ch))
    self.conv_in = nn.Conv2d(in_ch, ch, 3, padding=1)
    self.down = nn.ModuleList()
    skip_channels = [ch]
    current_ch = ch
    for level, multiplier in enumerate(self.ch_mult):
      blocks = nn.ModuleList()
      attentions = nn.ModuleList()
      out_channels = ch * multiplier
      for _ in range(num_res_blocks):
        blocks.append(ResnetBlock(current_ch, out_channels, temb_ch, dropout))
        current_ch = out_channels
        attentions.append(AttentionBlock(current_ch))
        skip_channels.append(current_ch)
      downsample = (nn.Conv2d(current_ch, current_ch, 3, stride=2, padding=1)
                    if resamp_with_conv and level != len(self.ch_mult) - 1 else None)
      self.down.append(nn.ModuleDict({'blocks': blocks, 'attentions': attentions,
                  'downsample': downsample or nn.Identity()}))
      if level != len(self.ch_mult) - 1:
        skip_channels.append(current_ch)

    self.mid = nn.ModuleList([ResnetBlock(current_ch, current_ch, temb_ch, dropout),
                              AttentionBlock(current_ch),
                              ResnetBlock(current_ch, current_ch, temb_ch, dropout)])
    self.up = nn.ModuleList()
    for level in reversed(range(len(self.ch_mult))):
      blocks = nn.ModuleList()
      attentions = nn.ModuleList()
      out_channels = ch * self.ch_mult[level]
      for _ in range(num_res_blocks + 1):
        skip_ch = skip_channels.pop()
        blocks.append(ResnetBlock(current_ch + skip_ch, out_channels, temb_ch, dropout))
        current_ch = out_channels
        attentions.append(AttentionBlock(out_channels))
      upsample = (nn.Conv2d(current_ch, current_ch, 3, padding=1)
                  if resamp_with_conv and level != 0 else None)
      self.up.append(nn.ModuleDict({'blocks': blocks, 'attentions': attentions,
                'upsample': upsample or nn.Identity()}))
    self.norm_out = nn.GroupNorm(_group_count(current_ch), current_ch)
    self.conv_out = nn.Conv2d(current_ch, out_ch, 3, padding=1)
    nn.init.zeros_(self.conv_out.weight)
    nn.init.zeros_(self.conv_out.bias)

  @staticmethod
  def timestep_embedding(t: Tensor, channels: int) -> Tensor:
    half = channels // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    values = t.float()[:, None] * frequencies[None]
    embedding = torch.cat([torch.sin(values), torch.cos(values)], dim=1)
    return F.pad(embedding, (0, channels - embedding.shape[1]))

  def forward(self, x: Tensor, t: Tensor, y: Optional[Tensor] = None) -> Tensor:
    del y
    if x.ndim != 4 or t.ndim != 1 or x.shape[0] != t.shape[0]:
      raise ValueError('expected x with shape (B, C, H, W) and t with shape (B,)')
    temb = self.temb(self.timestep_embedding(t, self.conv_in.out_channels))
    h = self.conv_in(x)
    hs = [h]
    for level, stage in enumerate(self.down):
      for block, attention in zip(stage['blocks'], stage['attentions']):
        h = block(h, temb)
        if h.shape[-1] in self.attn_resolutions:
          h = attention(h)
        hs.append(h)
      if level != len(self.down) - 1:
        h = stage['downsample'](h)
        hs.append(h)
    h = self.mid[0](h, temb)
    h = self.mid[1](h)
    h = self.mid[2](h, temb)
    for level, stage in enumerate(self.up):
      for block, attention in zip(stage['blocks'], stage['attentions']):
        h = block(torch.cat([h, hs.pop()], dim=1), temb)
        if h.shape[-1] in self.attn_resolutions:
          h = attention(h)
      if level != len(self.up) - 1:
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = stage['upsample'](h)
    return self.conv_out(nonlinearity(self.norm_out(h)))


def model(*, in_ch: int = 3, ch: int, out_ch: int, ch_mult=(1, 2, 4, 8),
          num_res_blocks: int, attn_resolutions, dropout: float = 0.,
          resamp_with_conv: bool = True, num_classes: int = 1, **kwargs) -> UNet:
  if num_classes != 1:
    raise ValueError('only unconditional models are supported')
  del kwargs
  return UNet(in_ch=in_ch, ch=ch, out_ch=out_ch, ch_mult=ch_mult,
              num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions,
              dropout=dropout, resamp_with_conv=resamp_with_conv)