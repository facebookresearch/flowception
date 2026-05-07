# Copyright 2025 The Lightricks team and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LTX Video Transformer Wrapper with support for:
- LTX 0.9.5 (2B)
- LTX 0.9.7 (13B dev/distilled)  
- LTX 0.9.8 (13B dev/distilled, 2B distilled)

Key architecture differences:
- 2B models: 28 layers, 64 attention_head_dim, 2048 cross_attention_dim
- 13B models: 48 layers, 128 attention_head_dim, 4096 cross_attention_dim
"""

import inspect
import logging
import math
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)

from huggingface_hub import snapshot_download

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, deprecate, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PixArtAlphaTextProjection
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle, RMSNorm


logger = logging.get_logger(__name__)


class LTXModelVersion(str, Enum):
    """Available LTX model versions."""
    LTXV_2B_095 = "ltxv-2b-0.9.5"
    LTXV_2B_096_DEV = "ltxv-2b-0.9.6-dev"
    LTXV_2B_096_DISTILLED = "ltxv-2b-0.9.6-distilled"
    LTXV_2B_098_DISTILLED = "ltxv-2b-0.9.8-distilled"
    LTXV_13B_097_DEV = "ltxv-13b-0.9.7-dev"
    LTXV_13B_097_DISTILLED = "ltxv-13b-0.9.7-distilled"
    LTXV_13B_098_DEV = "ltxv-13b-0.9.8-dev"
    LTXV_13B_098_DISTILLED = "ltxv-13b-0.9.8-distilled"


# Model configurations for different versions
LTX_MODEL_CONFIGS = {
    # 2B models (0.9.5, 0.9.6, 0.9.8-distilled)
    "2b": {
        "in_channels": 128,
        "out_channels": 128,
        "patch_size": 1,
        "patch_size_t": 1,
        "num_attention_heads": 32,
        "attention_head_dim": 64,
        "cross_attention_dim": 2048,
        "num_layers": 28,
        "caption_channels": 4096,
    },
    # 13B models (0.9.7, 0.9.8)
    "13b": {
        "in_channels": 128,
        "out_channels": 128,
        "patch_size": 1,
        "patch_size_t": 1,
        "num_attention_heads": 32,
        "attention_head_dim": 128,
        "cross_attention_dim": 4096,
        "num_layers": 48,
        "caption_channels": 4096,
    },
}

# Maps model registry key → (hf_repo_id, subfolder inside the repo)
_HF_PRETRAINED_MAP = {
    # 2B variants
    "ltx2b":              ("Lightricks/LTX-Video", "ltxv-2b-0.9.5/model_files"),
    "ltx2b-distilled":    ("Lightricks/LTX-Video", "ltxv-2b-0.9.5/model_files"),
    "ltx2b-p2-distilled": ("Lightricks/LTX-Video", "ltxv-2b-0.9.5/model_files"),
    "ltx2b-0.9.6-dev":        ("Lightricks/LTX-Video", "ltxv-2b-0.9.6-dev/model_files"),
    "ltx2b-0.9.6-distilled":  ("Lightricks/LTX-Video", "ltxv-2b-0.9.6-distilled/model_files"),
    "ltx2b-0.9.8-distilled":  ("Lightricks/LTX-Video", "ltxv-2b-0.9.8-distilled/model_files"),
    # 13B variants
    "ltx13b-distilled":       ("Lightricks/LTX-Video", "ltxv-13b-0.9.8-distilled/model_files"),
    "ltx13b-0.9.7-dev":       ("Lightricks/LTX-Video", "ltxv-13b-0.9.7-dev/model_files"),
    "ltx13b-0.9.7-distilled": ("Lightricks/LTX-Video", "ltxv-13b-0.9.7-distilled/model_files"),
    "ltx13b-0.9.8-dev":       ("Lightricks/LTX-Video", "ltxv-13b-0.9.8-dev/model_files"),
}


def _fetch_pretrained_checkpoint(model_key: str) -> str:
    """Download the pretrained LTX weights from HuggingFace and return the local directory path."""
    if model_key not in _HF_PRETRAINED_MAP:
        raise ValueError(f"No pretrained HF weights registered for model key '{model_key}'. "
                         f"Known keys: {list(_HF_PRETRAINED_MAP)}")
    repo_id, subfolder = _HF_PRETRAINED_MAP[model_key]
    logger.info(f"Downloading pretrained weights for '{model_key}' from {repo_id}/{subfolder} ...")
    local_dir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{subfolder}/*"],
    )
    return str(local_dir + "/" + subfolder)


def _make_token_mask_from_frame_mask(
    frame_mask: torch.Tensor,
    tokens_per_frame: int,
    include_frame_token: bool,
) -> torch.Tensor:
    """
    Returns token_valid: (B, S_total) bool
    Token order assumed: [frame0 tokens..., (frame0 ftok), frame1 tokens..., (frame1 ftok), ...]
    """
    if frame_mask.dtype != torch.bool:
        frame_mask = frame_mask.to(torch.bool)

    B, F = frame_mask.shape
    tpf_total = tokens_per_frame + (1 if include_frame_token else 0)
    token_valid = frame_mask[:, :, None].expand(B, F, tpf_total).reshape(B, F * tpf_total)
    return token_valid


def _pad_rope_for_frame_tokens(
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_frames: int,
    tokens_per_frame: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Insert per-frame token positions using the center spatial position of each frame.
    This gives frame tokens position (t, H/2, W/2) so they're close to their frame's tokens.
    """
    B, S_img, C = cos.shape
    assert S_img == num_frames * tokens_per_frame, f"RoPE len mismatch: {S_img} vs {num_frames*tokens_per_frame}"

    cos4 = cos.view(B, num_frames, tokens_per_frame, C)
    sin4 = sin.view(B, num_frames, tokens_per_frame, C)

    # Use center spatial position for each frame token
    center_idx = tokens_per_frame // 2
    cos_frame = cos4[:, :, center_idx:center_idx+1, :]  # (B, num_frames, 1, C)
    sin_frame = sin4[:, :, center_idx:center_idx+1, :]  # (B, num_frames, 1, C)

    cos_out = torch.cat([cos4, cos_frame], dim=2).reshape(B, num_frames * (tokens_per_frame + 1), C)
    sin_out = torch.cat([sin4, sin_frame], dim=2).reshape(B, num_frames * (tokens_per_frame + 1), C)
    
    return cos_out, sin_out


def apply_rotary_emb(x, freqs):
    cos, sin = freqs
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out


class LTXVideoAttnProcessor:
    """
    Attention processor for LTX Video models.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if is_torch_version("<", "2.0"):
            raise ValueError(
                "LTX attention processors require a minimum PyTorch version of 2.0."
            )

    def __call__(
        self,
        attn: "LTXAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class LTXAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = LTXVideoAttnProcessor
    _available_processors = [LTXVideoAttnProcessor]

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        kv_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = True,
        cross_attention_dim: Optional[int] = None,
        out_bias: bool = True,
        qk_norm: str = "rms_norm_across_heads",
        processor=None,
    ):
        super().__init__()
        if qk_norm != "rms_norm_across_heads":
            raise NotImplementedError("Only 'rms_norm_across_heads' is supported.")

        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = query_dim
        self.heads = heads

        norm_eps = 1e-5
        norm_elementwise_affine = True
        self.norm_q = torch.nn.RMSNorm(dim_head * heads, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.norm_k = torch.nn.RMSNorm(dim_head * kv_heads, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = torch.nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.to_v = torch.nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.to_out = torch.nn.ModuleList([])
        self.to_out.append(torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(torch.nn.Dropout(dropout))

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        unused_kwargs = [k for k, _ in kwargs.items() if k not in attn_parameters]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, **kwargs)


class LTXVideoRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        dim: int,
        base_num_frames: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        patch_size: int = 1,
        patch_size_t: int = 1,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.base_num_frames = base_num_frames
        self.base_height = base_height
        self.base_width = base_width
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.theta = theta

    def _prepare_video_coords(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        rope_interpolation_scale: Tuple[torch.Tensor, float, float],
        device: torch.device,
    ) -> torch.Tensor:
        grid_h = torch.arange(height, dtype=torch.float32, device=device)
        grid_w = torch.arange(width, dtype=torch.float32, device=device)
        grid_f = torch.arange(num_frames, dtype=torch.float32, device=device)
        grid = torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
        grid = torch.stack(grid, dim=0)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)

        if rope_interpolation_scale is not None:
            grid[:, 0:1] = grid[:, 0:1] * rope_interpolation_scale[0] * self.patch_size_t / self.base_num_frames
            grid[:, 1:2] = grid[:, 1:2] * rope_interpolation_scale[1] * self.patch_size / self.base_height
            grid[:, 2:3] = grid[:, 2:3] * rope_interpolation_scale[2] * self.patch_size / self.base_width

        grid = grid.flatten(2, 4).transpose(1, 2)
        return grid

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        rope_interpolation_scale: Optional[Tuple[torch.Tensor, float, float]] = None,
        video_coords: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.size(0)

        if video_coords is None:
            grid = self._prepare_video_coords(
                batch_size,
                num_frames,
                height,
                width,
                rope_interpolation_scale=rope_interpolation_scale,
                device=hidden_states.device,
            )
        else:
            grid = torch.stack(
                [
                    video_coords[:, 0] / self.base_num_frames,
                    video_coords[:, 1] / self.base_height,
                    video_coords[:, 2] / self.base_width,
                ],
                dim=-1,
            )

        start = 1.0
        end = self.theta
        freqs = self.theta ** torch.linspace(
            math.log(start, self.theta),
            math.log(end, self.theta),
            self.dim // 6,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        freqs = freqs * math.pi / 2.0
        freqs = freqs * (grid.unsqueeze(-1) * 2 - 1)
        freqs = freqs.transpose(-1, -2).flatten(2)

        cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

        if self.dim % 6 != 0:
            cos_padding = torch.ones_like(cos_freqs[:, :, : self.dim % 6])
            sin_padding = torch.zeros_like(cos_freqs[:, :, : self.dim % 6])
            cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
            sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)

        return cos_freqs, sin_freqs


@maybe_allow_in_graph
class LTXVideoTransformerBlock(nn.Module):
    """
    Transformer block for LTX Video models.
    Supports both 2B and 13B architectures.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        qk_norm: str = "rms_norm_across_heads",
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn1 = LTXAttention(
            query_dim=dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            cross_attention_dim=None,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
        )

        self.norm2 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn2 = LTXAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
        )

        self.ff = FeedForward(dim, activation_fn=activation_fn)

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        self_attention_mask: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.size(0)
        norm_hidden_states = self.norm1(hidden_states)

        num_ada_params = self.scale_shift_table.shape[0]
        ada_values = self.scale_shift_table[None, None].to(temb.device) + temb.reshape(
            batch_size, temb.size(1), num_ada_params, -1
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=self_attention_mask,
            image_rotary_emb=image_rotary_emb,
        )

        if token_mask is not None:
            attn_hidden_states = attn_hidden_states * token_mask.unsqueeze(-1).to(attn_hidden_states.dtype)

        hidden_states = hidden_states + attn_hidden_states * gate_msa

        attn_hidden_states = self.attn2(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=None,
            attention_mask=encoder_attention_mask,
        )

        if token_mask is not None:
            attn_hidden_states = attn_hidden_states * token_mask.unsqueeze(-1).to(attn_hidden_states.dtype)

        hidden_states = hidden_states + attn_hidden_states
        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp

        return hidden_states


@maybe_allow_in_graph
class LTXVideoTransformer3DModel(
    ModelMixin, ConfigMixin, AttentionMixin, FromOriginalModelMixin, PeftAdapterMixin, CacheMixin
):
    """
    LTX Video Transformer model supporting both 2B and 13B architectures.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["norm"]
    _repeated_blocks = ["LTXVideoTransformerBlock"]
    _cp_plan = {
        "": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "encoder_attention_mask": ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
        },
        "rope": {
            0: ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
            1: ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 128,
        patch_size: int = 1,
        patch_size_t: int = 1,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        cross_attention_dim: int = 2048,
        num_layers: int = 28,
        activation_fn: str = "gelu-approximate",
        qk_norm: str = "rms_norm_across_heads",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = 4096,
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        use_frame_tokens: bool = False,
        use_insertion_head: bool = False,
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels
        inner_dim = num_attention_heads * attention_head_dim

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t  # keep for future; we’ll do spatial only here

        in_proj_dim = in_channels * (patch_size * patch_size)  # spatial patching only
        out_proj_dim = out_channels * (patch_size * patch_size)

        self.proj_in = nn.Linear(in_proj_dim, inner_dim)
        
        self.use_frame_tokens = use_frame_tokens
        self.use_insertion_head = use_insertion_head

        if self.use_frame_tokens:
            self.frame_token = nn.Parameter(torch.zeros(1, 1, inner_dim), requires_grad=True)
        else:
            self.frame_token = None

        if self.use_insertion_head:
            self.insertion_head = nn.Linear(inner_dim, 1)
        else:
            self.insertion_head = None

        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.time_embed = AdaLayerNormSingle(inner_dim, use_additional_conditions=False)

        self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)

        self.rope = LTXVideoRotaryPosEmbed(
            dim=inner_dim,
            base_num_frames=20,
            base_height=2048,
            base_width=2048,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            theta=10000.0,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                LTXVideoTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    qk_norm=qk_norm,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    attention_out_bias=attention_out_bias,
                    eps=norm_eps,
                    elementwise_affine=norm_elementwise_affine,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = nn.LayerNorm(inner_dim, eps=1e-6, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_proj_dim)

        self.gradient_checkpointing = False

    def _patchify_spatial(self, x, num_frames, height, width):
        # x: [B, S, C] where S = num_frames * height * width
        ps = self.patch_size
        if ps == 1:
            return x, height, width

        B, S, C = x.shape
        assert S == num_frames * height * width, "Token length mismatch"
        assert height % ps == 0 and width % ps == 0, f"H,W must be divisible by patch_size={ps}"

        x = x.view(B, num_frames, height, width, C)

        Hp, Wp = height // ps, width // ps
        # [B, D, Hp, ps, Wp, ps, C] -> [B, D, Hp, Wp, ps, ps, C]
        x = x.view(B, num_frames, Hp, ps, Wp, ps, C).permute(0, 1, 2, 4, 3, 5, 6)
        # merge patch pixels into channel dim
        x = x.reshape(B, num_frames * Hp * Wp, C * ps * ps)
        return x, Hp, Wp


    def _unpatchify_spatial(self, x, num_frames, Hp, Wp):
        # x: [B, num_frames*Hp*Wp, out_channels*ps*ps]
        ps = self.patch_size
        if ps == 1:
            return x

        B, S, Cout_ps2 = x.shape
        assert S == num_frames * Hp * Wp, "Patched token length mismatch"
        out_channels = Cout_ps2 // (ps * ps)
        assert out_channels * ps * ps == Cout_ps2, "Bad output channel multiple"

        x = x.view(B, num_frames, Hp, Wp, ps, ps, out_channels)  # [B,D,Hp,Wp,ps,ps,Cout]
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()          # [B,D,Hp,ps,Wp,ps,Cout]
        x = x.view(B, num_frames, Hp * ps, Wp * ps, out_channels) # [B,D,H,W,Cout]
        x = x.view(B, num_frames * (Hp * ps) * (Wp * ps), out_channels)
        return x


    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        rope_interpolation_scale: Optional[Union[Tuple[float, float, float], torch.Tensor]] = None,
        video_coords: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        frame_mask: Optional[torch.Tensor] = None,
        return_frame_logits: bool = False,
    ) -> torch.Tensor:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

        if num_frames is None:
            raise ValueError("num_frames must be provided for LTXVideoTransformer3DModel.forward().")

        batch_size = hidden_states.size(0)

       
        hidden_states, rope_h, rope_w = self._patchify_spatial(hidden_states, num_frames, height, width)
        tokens_per_frame = rope_h * rope_w


        # IMPORTANT: now tokens_per_frame is rope_h * rope_w (not height * width)
        # and RoPE should be computed on patch grid sizes
        image_rotary_emb = self.rope(
            hidden_states,
            num_frames=num_frames,
            height=rope_h,
            width=rope_w,
            rope_interpolation_scale=rope_interpolation_scale,
            video_coords=video_coords,
        )


        if self.use_frame_tokens:
            cos, sin = image_rotary_emb
            cos, sin = _pad_rope_for_frame_tokens(cos, sin, num_frames=num_frames, tokens_per_frame=tokens_per_frame)
            image_rotary_emb = (cos, sin)

        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        hidden_states = self.proj_in(hidden_states)

        frame_logits = None
        if self.use_frame_tokens:
            hs4 = hidden_states.view(batch_size, num_frames, tokens_per_frame, -1)
            ftok = self.frame_token.to(hidden_states.dtype).expand(batch_size, num_frames, 1, -1)
            hidden_states = torch.cat([hs4, ftok], dim=2).reshape(batch_size, num_frames * (tokens_per_frame + 1), -1)
            S_total = hidden_states.shape[1]
        else:
            S_total = hidden_states.shape[1]

        if timestep.ndim == 2:
            if timestep.shape[1] != num_frames:
                raise ValueError(f"timestep has F={timestep.shape[1]} but num_frames={num_frames}.")

            temb_f, embedded_f = self.time_embed(
                timestep.reshape(-1),
                batch_size=batch_size * num_frames,
                hidden_dtype=hidden_states.dtype,
            )
            temb_f = temb_f.view(batch_size, num_frames, -1)
            embedded_f = embedded_f.view(batch_size, num_frames, -1)

            tpf_total = tokens_per_frame + (1 if self.use_frame_tokens else 0)
            temb = temb_f[:, :, None, :].expand(batch_size, num_frames, tpf_total, temb_f.shape[-1]).reshape(
                batch_size, num_frames * tpf_total, temb_f.shape[-1]
            )

            embedded_timestep = embedded_f[:, 0:1, :]
        else:
            temb, embedded_timestep = self.time_embed(
                timestep.flatten(),
                batch_size=batch_size,
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(batch_size, -1, temb.size(-1))
            embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        token_mask = None
        self_attention_mask = None
        if frame_mask is not None:
            token_mask = _make_token_mask_from_frame_mask(
                frame_mask=frame_mask,
                tokens_per_frame=tokens_per_frame,
                include_frame_token=self.use_frame_tokens,
            )

            # self_attention_mask = (1.0 - token_mask.to(hidden_states.dtype)) * -100000.0
            # self_attention_mask = self_attention_mask.unsqueeze(1)
            
            mask_value = torch.finfo(hidden_states.dtype).min
            self_attention_mask = torch.where(token_mask, 0.0, mask_value).to(hidden_states.dtype)

            

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    encoder_attention_mask,
                    self_attention_mask,
                    token_mask,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    encoder_attention_mask=encoder_attention_mask,
                    self_attention_mask=self_attention_mask,
                    token_mask=token_mask,
                )

        scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift

        if self.use_frame_tokens:
            hs4 = hidden_states.view(batch_size, num_frames, tokens_per_frame + 1, -1)
            frame_tok_states = hs4[:, :, -1, :]

            if self.use_insertion_head:
                frame_logits = self.insertion_head(frame_tok_states)

            hidden_states = hs4[:, :, :tokens_per_frame, :].reshape(batch_size, num_frames * tokens_per_frame, -1)

            if frame_mask is not None:
                spatial_token_mask = _make_token_mask_from_frame_mask(
                    frame_mask=frame_mask,
                    tokens_per_frame=tokens_per_frame,
                    include_frame_token=False,
                )
                hidden_states = hidden_states * spatial_token_mask.unsqueeze(-1).to(hidden_states.dtype)

        output = self.proj_out(hidden_states)
        output = self._unpatchify_spatial(output, num_frames=num_frames, Hp=rope_h, Wp=rope_w)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            if return_frame_logits:
                return (output, frame_logits)
            return (output,)

        out = Transformer2DModelOutput(sample=output)
        if return_frame_logits:
            out.frame_logits = frame_logits
        return out


class FlowceptionV3_LTXWrapper(nn.Module):
    """
    API-compatible wrapper around LTXVideoTransformer3DModel.
    Supports both 2B and 13B model variants.
    
    Args:
        model_size: "2b" or "13b" to select model architecture
        checkpoint_path: Path to the safetensors checkpoint
        use_frame_tokens: Whether to use frame tokens for insertion
        use_insertion_head: Whether to use insertion head
    """

    def __init__(
        self,
        in_channels: int = 4,
        patch_size: int = 2,
        act_checkpoint: bool = False,
        fps: float = 24.0,
        # LTX-specific
        model_size: str = "2b",  # "2b" or "13b"
        ltx_in_channels: int = 128,
        ltx_out_channels: Optional[int] = None,
        ltx_caption_channels: int = 4096,
        use_frame_tokens: bool = True,
        use_insertion_head: bool = True,
        checkpoint_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        self.learn_sigma = False
        self.in_channels = in_channels
        self.model_size = model_size

        if ltx_out_channels is None:
            ltx_out_channels = ltx_in_channels

        self.ltx_in_channels = ltx_in_channels
        self.ltx_out_channels = ltx_out_channels
        
        self.fps = fps

        # Get config for model size
        if model_size not in LTX_MODEL_CONFIGS:
            raise ValueError(f"model_size must be '2b' or '13b', got {model_size}")
        
        config = LTX_MODEL_CONFIGS[model_size]

        # Create LTX transformer with appropriate config
        self.ltx = LTXVideoTransformer3DModel(
            in_channels=in_channels,
            out_channels=in_channels,
            patch_size=patch_size,
            patch_size_t=config["patch_size_t"],
            num_attention_heads=config["num_attention_heads"],
            attention_head_dim=config["attention_head_dim"],
            cross_attention_dim=config["cross_attention_dim"],
            num_layers=config["num_layers"],
            activation_fn="gelu-approximate",
            qk_norm="rms_norm_across_heads",
            norm_elementwise_affine=False,
            norm_eps=1e-6,
            caption_channels=config["caption_channels"],
            attention_bias=True,
            attention_out_bias=True,
            use_frame_tokens=use_frame_tokens,
            use_insertion_head=use_insertion_head,
        )
        self.ltx.train()

        if act_checkpoint:
            self.ltx.enable_gradient_checkpointing()
        
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        # Fallback logits if insertion head disabled
        self.fallback_logits = nn.Linear(ltx_out_channels, 1)


    def _load_checkpoint(self, checkpoint_path: str):
        from pathlib import Path
        import glob
        from safetensors.torch import load_file

        checkpoint_path = Path(checkpoint_path)

        # --- load safetensors (your existing logic) ---
        if checkpoint_path.is_dir():
            shard_files = sorted(glob.glob(str(checkpoint_path / "diffusion_pytorch_model-*.safetensors")))
            if shard_files:
                ltx_state = {}
                for shard_file in shard_files:
                    ltx_state.update(load_file(shard_file))
            else:
                single_file = checkpoint_path / "diffusion_pytorch_model.safetensors"
                ltx_state = load_file(str(single_file))
        else:
            ltx_state = load_file(str(checkpoint_path))

        # --- NEW: filter mismatched shapes ---
        model_state = self.ltx.state_dict()
        filtered = {}
        bad = []
        for k, v in ltx_state.items():
            if k in model_state and tuple(v.shape) == tuple(model_state[k].shape):
                filtered[k] = v
            elif k in model_state:
                bad.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            # else: unexpected key -> will be reported by strict=False

        if bad:
            logger.warning(f"[ckpt] Skipping {len(bad)} keys with shape mismatch. Examples:")
            for k, s_ckpt, s_model in bad[:10]:
                logger.warning(f"  {k}: ckpt={s_ckpt} model={s_model}")

        result = self.ltx.load_state_dict(filtered, strict=False)
        logger.info(f"Load result - Missing: {len(result.missing_keys)}, Unexpected: {len(result.unexpected_keys)}")


    def forward(
        self,
        sample,
        timestep,
        class_labels,
        encoder_hidden_states=None,
        seg_start=-1,
        seg_end=99999,
        blur_sigma=-1.0,
        return_means=False,
        context_frames=None,
        frame_mask=None,
        fps: float = 24.0,
        timestep_in_sigma: bool = True,
        **kwargs,
    ):
        """
        Forward pass.
        
        Args:
            sample: [B, D, C, H, W] (AE latent)
            timestep: [B] or [B, D] - sigma values in [0, 1] where 1=noise, 0=clean
            class_labels: [B, S_text, caption_channels] (text tokens)
            frame_mask: [B, D] bool/0-1
            fps: frames per second for ROPE scaling
            timestep_in_sigma: If True (default), timestep is sigma in [0,1].
                              If False, timestep is already scaled to [0, 1000].
                              
        Note on Flow Matching / Rectified Flow:
            - LTX uses flow matching where the model predicts velocity v = ε - x₀
            - The noisy sample is x_t = (1-σ)x₀ + σε
            - The scheduler update is x_{t-dt} = x_t - dt * v
            - Timesteps are sigma values: 1.0 = pure noise, 0.0 = clean
            - The transformer expects timestep * 1000 (scaled to [0, 1000])
        """
        B, D, C, H, W = sample.shape
        if C != self.ltx_in_channels:
            raise ValueError(f"Expected sample channels C={self.ltx_in_channels}, got {C}.")

        # Build token sequence: [B, S_img, C] where S_img = D*H*W
        tok = sample.permute(0, 1, 3, 4, 2).contiguous().view(B, D * H * W, C)

        text = class_labels
        if text is None:
            raise ValueError("class_labels must be provided as text tokens.")

        encoder_attention_mask = tok.new_ones((B, text.shape[1]), dtype=torch.long)

        if frame_mask is not None and frame_mask.dtype != torch.bool:
            frame_mask = frame_mask.to(torch.bool)

        # Compute ROPE scaling factors
        # temporal_compression_ratio = self.temporal_scale #8.0
        # spatial_compression_ratio = self.spatial_scale #32.0
        temporal_compression_ratio = 8.0
        spatial_compression_ratio = 32.0
        rope_interpolation_scale = [
            temporal_compression_ratio / self.fps,
            spatial_compression_ratio,
            spatial_compression_ratio,
        ]

        # Convert timestep to LTX format
        # LTX expects timestep in [0, 1000] where 1000 = pure noise, 0 = clean
        # if timestep_in_sigma:
        #     # timestep is sigma in [0, 1], scale to [0, 1000]
        #     ltx_timestep = (1.0 - timestep) * 1000.0
        # else:
        #     # timestep is already scaled
        #     ltx_timestep = timestep
        ltx_timestep = (1.0 - timestep) * 1000.0

        out = self.ltx(
            hidden_states=tok,
            encoder_hidden_states=text,
            timestep=ltx_timestep,
            encoder_attention_mask=encoder_attention_mask,
            num_frames=D,
            height=H,
            width=W,
            rope_interpolation_scale=rope_interpolation_scale,
            frame_mask=frame_mask,
            return_frame_logits=True,
            return_dict=True,
        )

        tok_out = out.sample
        feat = tok_out.view(B, D, H, W, tok_out.shape[-1]).permute(0, 4, 1, 2, 3).contiguous()

        # For flow matching, the model output IS the velocity (v = ε - x₀)
        vel = -feat

        if not hasattr(out, "frame_logits") or out.frame_logits is None:
            raise RuntimeError(
                "frame_logits is missing. Ensure: use_frame_tokens=True, use_insertion_head=True, "
                "and return_frame_logits=True."
            )
        logits = out.frame_logits

        if not return_means:
            return vel, logits

        repa_out = None
        means = []
        means_y = []
        return vel, logits, repa_out, means, means_y


# Factory functions for different model sizes
def LTX2B(**kwargs):
    """Create LTX-2B model initialised from LTX-Video 0.9.5 weights."""
    checkpoint_path = kwargs.pop("checkpoint_path", None)
    fetch_pretrained = kwargs.pop("fetch_pretrained", False)
    if fetch_pretrained and not checkpoint_path:
        checkpoint_path = _fetch_pretrained_checkpoint("ltx2b-distilled")

    model = FlowceptionV3_LTXWrapper(
        model_size="2b",
        patch_size=1,
        depth_patch_size=1,
        hidden_size=2048,
        depth=28,
        num_heads=32,
        mlp_ratio=4.0,
        class_dropout_prob=0.0,
        learn_sigma=False,
        add_y_emb=False,
        ltx_in_channels=128,
        ltx_out_channels=None,
        checkpoint_path=checkpoint_path,
        **kwargs
    )
    
    return model


def LTX2B_P2(**kwargs):
    """Create LTX-2B model with patch_size=2, initialised from LTX-Video 0.9.5 weights."""
    checkpoint_path = kwargs.pop("checkpoint_path", None)
    fetch_pretrained = kwargs.pop("fetch_pretrained", False)
    if fetch_pretrained and not checkpoint_path:
        checkpoint_path = _fetch_pretrained_checkpoint("ltx2b-p2-distilled")

    model = FlowceptionV3_LTXWrapper(
        model_size="2b",
        patch_size=2,
        depth_patch_size=1,
        hidden_size=2048,
        depth=28,
        num_heads=32,
        mlp_ratio=4.0,
        class_dropout_prob=0.0,
        learn_sigma=False,
        add_y_emb=False,
        ltx_in_channels=16,
        ltx_out_channels=None,
        checkpoint_path=checkpoint_path,
        **kwargs
    )
    
    return model


def LTX2B_096_Dev(**kwargs):
    """Create LTX-2B model initialised from LTX-Video 0.9.6-dev weights."""
    checkpoint_path = kwargs.pop("checkpoint_path", None)
    fetch_pretrained = kwargs.pop("fetch_pretrained", False)
    if fetch_pretrained and not checkpoint_path:
        checkpoint_path = _fetch_pretrained_checkpoint("ltx2b-0.9.6-dev")

    model = FlowceptionV3_LTXWrapper(
        model_size="2b",
        patch_size=1,
        depth_patch_size=1,
        hidden_size=2048,
        depth=28,
        num_heads=32,
        mlp_ratio=4.0,
        class_dropout_prob=0.0,
        learn_sigma=False,
        add_y_emb=False,
        ltx_in_channels=128,
        ltx_out_channels=None,
        checkpoint_path=checkpoint_path,
        **kwargs
    )
    return model


def LTX2B_096_Distilled(**kwargs):
    """Create LTX-2B model initialised from LTX-Video 0.9.6-distilled weights."""
    checkpoint_path = kwargs.pop("checkpoint_path", None)
    fetch_pretrained = kwargs.pop("fetch_pretrained", False)
    if fetch_pretrained and not checkpoint_path:
        checkpoint_path = _fetch_pretrained_checkpoint("ltx2b-0.9.6-distilled")

    model = FlowceptionV3_LTXWrapper(
        model_size="2b",
        patch_size=1,
        depth_patch_size=1,
        hidden_size=2048,
        depth=28,
        num_heads=32,
        mlp_ratio=4.0,
        class_dropout_prob=0.0,
        learn_sigma=False,
        add_y_emb=False,
        ltx_in_channels=128,
        ltx_out_channels=None,
        checkpoint_path=checkpoint_path,
        **kwargs
    )
    return model


def LTX2B_098_Distilled(**kwargs):
    """Create LTX-2B model initialised from LTX-Video 0.9.8-distilled weights."""
    checkpoint_path = kwargs.pop("checkpoint_path", None)
    fetch_pretrained = kwargs.pop("fetch_pretrained", False)
    if fetch_pretrained and not checkpoint_path:
        checkpoint_path = _fetch_pretrained_checkpoint("ltx2b-0.9.8-distilled")

    model = FlowceptionV3_LTXWrapper(
        model_size="2b",
        patch_size=1,
        depth_patch_size=1,
        hidden_size=2048,
        depth=28,
        num_heads=32,
        mlp_ratio=4.0,
        class_dropout_prob=0.0,
        learn_sigma=False,
        add_y_emb=False,
        ltx_in_channels=128,
        ltx_out_channels=None,
        checkpoint_path=checkpoint_path,
        **kwargs
    )
    return model


def LTX13B(**kwargs):
    """Create LTX-13B model initialised from LTX-Video 0.9.8-distilled weights."""
    checkpoint_path = kwargs.pop("checkpoint_path", None)
    fetch_pretrained = kwargs.pop("fetch_pretrained", False)
    if fetch_pretrained and not checkpoint_path:
        checkpoint_path = _fetch_pretrained_checkpoint("ltx13b-distilled")

    model = FlowceptionV3_LTXWrapper(
        model_size="13b",
        patch_size=1,
        depth_patch_size=1,
        hidden_size=4096,  # 32 heads * 128 dim
        depth=48,
        num_heads=32,
        mlp_ratio=4.0,
        class_dropout_prob=0.0,
        learn_sigma=False,
        add_y_emb=False,
        ltx_in_channels=128,
        ltx_out_channels=None,
        checkpoint_path=checkpoint_path,
        **kwargs
    )
    
    return model


def LTX13B_097_Dev(**kwargs):
    """Create LTX-13B model initialised from LTX-Video 0.9.7-dev weights."""
    checkpoint_path = kwargs.pop("checkpoint_path", None)
    fetch_pretrained = kwargs.pop("fetch_pretrained", False)
    if fetch_pretrained and not checkpoint_path:
        checkpoint_path = _fetch_pretrained_checkpoint("ltx13b-0.9.7-dev")

    model = FlowceptionV3_LTXWrapper(
        model_size="13b",
        patch_size=1,
        depth_patch_size=1,
        hidden_size=4096,
        depth=48,
        num_heads=32,
        mlp_ratio=4.0,
        class_dropout_prob=0.0,
        learn_sigma=False,
        add_y_emb=False,
        ltx_in_channels=128,
        ltx_out_channels=None,
        checkpoint_path=checkpoint_path,
        **kwargs
    )
    return model


def LTX13B_097_Distilled(**kwargs):
    """Create LTX-13B model initialised from LTX-Video 0.9.7-distilled weights."""
    checkpoint_path = kwargs.pop("checkpoint_path", None)
    fetch_pretrained = kwargs.pop("fetch_pretrained", False)
    if fetch_pretrained and not checkpoint_path:
        checkpoint_path = _fetch_pretrained_checkpoint("ltx13b-0.9.7-distilled")

    model = FlowceptionV3_LTXWrapper(
        model_size="13b",
        patch_size=1,
        depth_patch_size=1,
        hidden_size=4096,
        depth=48,
        num_heads=32,
        mlp_ratio=4.0,
        class_dropout_prob=0.0,
        learn_sigma=False,
        add_y_emb=False,
        ltx_in_channels=128,
        ltx_out_channels=None,
        checkpoint_path=checkpoint_path,
        **kwargs
    )
    return model


def LTX13B_098_Dev(**kwargs):
    """Create LTX-13B model initialised from LTX-Video 0.9.8-dev weights."""
    checkpoint_path = kwargs.pop("checkpoint_path", None)
    fetch_pretrained = kwargs.pop("fetch_pretrained", False)
    if fetch_pretrained and not checkpoint_path:
        checkpoint_path = _fetch_pretrained_checkpoint("ltx13b-0.9.8-dev")

    model = FlowceptionV3_LTXWrapper(
        model_size="13b",
        patch_size=1,
        depth_patch_size=1,
        hidden_size=4096,
        depth=48,
        num_heads=32,
        mlp_ratio=4.0,
        class_dropout_prob=0.0,
        learn_sigma=False,
        add_y_emb=False,
        ltx_in_channels=128,
        ltx_out_channels=None,
        checkpoint_path=checkpoint_path,
        **kwargs
    )
    return model


# Model registry
ltxn_98_models = {
    # 2B — trained with LTX-Video 0.9.5
    "ltx2b":              LTX2B,
    "ltx2b-distilled":    LTX2B,
    "ltx2b-p2-distilled": LTX2B_P2,
    # 2B — newer base weights
    "ltx2b-0.9.6-dev":        LTX2B_096_Dev,
    "ltx2b-0.9.6-distilled":  LTX2B_096_Distilled,
    "ltx2b-0.9.8-distilled":  LTX2B_098_Distilled,
    # 13B — trained with LTX-Video 0.9.8-distilled
    "ltx13b-distilled":       LTX13B,
    "ltx13b-0.9.7-dev":       LTX13B_097_Dev,
    "ltx13b-0.9.7-distilled": LTX13B_097_Distilled,
    "ltx13b-0.9.8-dev":       LTX13B_098_Dev,
}


# Convenience function to create model by version
def create_ltx_model(
    version: Union[str, LTXModelVersion],
    checkpoint_path: Optional[str] = None,
    **kwargs
) -> FlowceptionV3_LTXWrapper:
    """
    Create LTX model by version string or enum.
    
    Args:
        version: Model version (e.g., "ltxv-2b-0.9.5", "ltxv-13b-0.9.8-dev")
        checkpoint_path: Path to checkpoint file
        **kwargs: Additional arguments passed to the model
        
    Returns:
        FlowceptionV3_LTXWrapper instance
    """
    if isinstance(version, LTXModelVersion):
        version = version.value
    
    version_lower = version.lower()
    
    # Determine model size from version string
    if "13b" in version_lower:
        factory = LTX13B
    elif "2b" in version_lower:
        factory = LTX2B
    else:
        raise ValueError(f"Cannot determine model size from version: {version}")
    
    return factory(checkpoint_path=checkpoint_path, **kwargs)
