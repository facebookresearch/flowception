# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Part of this implementation is adapted from https://github.com/facebookresearch/DiT
# which is released under NonCommercial-4.0 license

# Part of this implementation is adapted from https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
# which is released under MIT license

# Part of this implementation is adapted from https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
# which is released under MIT license

import math
from typing import Optional

import torch
import torch.nn.functional as F

from einops import rearrange
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from torch.nn.attention.flex_attention import BlockMask, flex_attention, create_block_mask

from torch import nn, Tensor

import rotary


def bias_dropout_add_scale(
    x: Tensor, scale: Tensor, residual: Optional[Tensor], prob: float, training: bool
) -> Tensor:
    return residual + scale * F.dropout(x, p=prob, training=training)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            x = F.layer_norm(x.float(), [self.dim])

        return x * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: float = 0.001,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

    @staticmethod
    def timestep_embedding(time: Tensor, dim: int, max_period: float = 0.001) -> Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=time.device)
        args = time[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, time: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(
            time=time, dim=self.frequency_embedding_size, max_period=self.max_period
        )
        t_emb = self.mlp(t_freq)
        return t_emb


class DDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        cond_dim: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        attn_type: str = "sdpa",
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be devisable by n_heads"

        self.n_heads = n_heads
        self.dim = dim
        self.dropout = dropout
        self.attn_type = attn_type

        self.head_dim = self.dim // self.n_heads

        self.norm1 = LayerNorm(dim=dim)

        self.qw = nn.Linear(dim, dim, bias=False)
        self.kw = nn.Linear(dim, dim, bias=False)
        self.vw = nn.Linear(dim, dim, bias=False)

        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim=dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: Tensor, rotary_cos_sin: Tensor, c: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=-1)

        x_skip = x
        x = modulate(x=self.norm1(x), shift=shift_msa, scale=scale_msa)

        q = self.qw(x)
        k = self.kw(x)
        v = self.vw(x)

        q, k, v = (
            item.view(batch_size, seq_len, self.n_heads, self.head_dim)
            for item in (q, k, v)
        )

        with torch.amp.autocast("cuda", enabled=False):
            cos, sin = rotary_cos_sin
        original_dtype = q.dtype

        q = rotary.apply_rotary_emb_torch(
            x=q.float(), cos=cos.float(), sin=sin.float()
        ).to(original_dtype)
        k = rotary.apply_rotary_emb_torch(
            x=k.float(), cos=cos.float(), sin=sin.float()
        ).to(original_dtype)

        q, k, v = (item.transpose(1, 2) for item in (q, k, v))

        if self.attn_type == "sdpa":
            x = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=mask)
        elif self.attn_type == "flex":
            assert isinstance(mask, BlockMask)
            x = flex_attention(q, k, v, block_mask=mask)
        else:
            raise ValueError(f"Unknown attention type {self.attn_type}")

        x = rearrange(x, "b h s d -> b s (h d)", b=batch_size)
        x = bias_dropout_add_scale(
            x=self.attn_out(x),
            scale=gate_msa,
            residual=x_skip,
            prob=self.dropout,
            training=self.training,
        )
        x = bias_dropout_add_scale(
            x=self.mlp(modulate(x=self.norm2(x), shift=shift_mlp, scale=scale_mlp)),
            scale=gate_mlp,
            residual=x,
            prob=self.dropout,
            training=self.training,
        )

        return x


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, 2*hidden_size)
        self.vel_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, out_channels)
        )
        self.lambda_ins_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        
        
    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = modulate(x=self.norm_final(x), shift=shift, scale=scale)
        x = self.linear(x)
        
        vel = self.vel_head(x[:, :, :self.hidden_size])
        
        lambda_ins = self.lambda_ins_head(x[:, :, self.hidden_size:(2*self.hidden_size)])
    
        return vel, lambda_ins


class Transformer(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        self.config = config

        self.embedder = nn.Linear(config.input_dim, config.hidden_size)

        self.time_embedding = TimestepEmbedder(
            hidden_size=config.cond_dim, max_period=config.max_period
        )
        self.rotary_emb = rotary.Rotary(dim=config.hidden_size // config.n_heads, base=config.rope_base)

        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    dim=config.hidden_size,
                    n_heads=config.n_heads,
                    cond_dim=config.cond_dim,
                    dropout=config.dropout,
                    attn_type="sdpa",
                )
                for _ in range(config.n_blocks)
            ]
        )

        self.output_layer = DDitFinalLayer(
            hidden_size=config.hidden_size,
            out_channels=config.out_channels,
            cond_dim=config.cond_dim,
        )

    def forward(self, x_t: Tensor, time: Tensor, mask=None) -> Tensor:
        x = self.embedder(x_t)
        b, d = time.shape
        tau_f = rearrange(time, "b d-> (b d)")
        
        c = F.silu(self.time_embedding(time=tau_f))
        c = rearrange(c, "(b d) h -> b d h", b=b, d=d)
        
        rotary_cos_sin = self.rotary_emb(x=x)

        mask = torch.logical_or(torch.logical_and(mask[:, :, None], mask[:, None, :]),
                                    torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device))
        mask = mask[:, None, :, :]
        
        with torch.amp.autocast("cuda", dtype=torch.float32):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x=x, rotary_cos_sin=rotary_cos_sin, c=c, mask=mask)
            x = self.output_layer(x=x, c=c)

        return x