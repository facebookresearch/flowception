import math
import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from einops import rearrange

try:
    from modules.size_embed import JointTimestepEmbedderND
except (ImportError, ModuleNotFoundError):
    JointTimestepEmbedderND = None  # not used by FlowceptionV1
from modules.video.video_modules.depth_modules import (
    PixelShuffle3d,
    PixelUnshuffle3d,
    FusedRMSNorm,
    RMSNorm,
)  # , modulate
from modules.video.video_modules.video_rope import (
    Qwen2VLRotaryEmbedding,
    apply_multimodal_rotary_pos_emb,
    generate_position_ids,
)
from timm.models.vision_transformer import Mlp

from einops import rearrange, repeat
from collections import defaultdict
from functools import partial

import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            FusedRMSNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=20):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
            These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = 256.0 * t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # if dim % 2:
        #     embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class MultiTextEmbedder(nn.Module):
    def __init__(self, text_dim_1, text_dim_2, hidden_size):
        super().__init__()
        self.emb1 = nn.Sequential(
            nn.Sequential(
                nn.Linear(text_dim_1, hidden_size),
                FusedRMSNorm(hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
        )

        self.emb2 = nn.Sequential(
            nn.Sequential(
                nn.Linear(text_dim_2, hidden_size),
                FusedRMSNorm(hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
        )

    def forward(self, emb):
        x1 = self.emb1(emb[0])
        x2 = self.emb2(emb[1])
        return torch.cat([x1, x2], dim=1)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
        mp_size: int = 1,
    ):
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert hidden_dim % mp_size == 0

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x3 = self.w3(x)
        output = self.w2(F.silu(x1) * x3)
        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        in_init_std = init_std or (self.dim ** (-0.5))
        out_init_std = init_std or (self.hidden_dim ** (-0.5))
        in_init_std = in_init_std
        out_init_std = out_init_std / factor
        for w in [self.w1, self.w3]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=in_init_std,
                a=-3 * in_init_std,
                b=3 * in_init_std,
            )
        nn.init.trunc_normal_(
            self.w2.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )


class SpatioTemporalCrossAttention(nn.Module):
    """
    rope_mode:
      - "global": apply RoPE over the full channel dim (original behavior) and RMSNorm over C
      - "per_head": apply RoPE independently per head (on head_dim) and RMSNorm per head
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        rope_theta=10000,
        device="cpu",
        rope_mode: str = "per_head",  # <--- NEW
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.rope_mode = rope_mode.lower()
        assert self.rope_mode in ("global", "per_head"), "rope_mode must be 'global' or 'per_head'"

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_linear = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)

        # ---------- Norms ----------
        if self.rope_mode == "global":
            # single RMSNorm over all channels
            self.q_norm = FusedRMSNorm(dim)
            self.k_norm = FusedRMSNorm(dim)
            self.q_norm_head = None
            self.k_norm_head = None
        else:
            # per-head RMSNorm
            self.q_norm_head = RMSNorm(self.head_dim)
            self.k_norm_head = RMSNorm(self.head_dim)
            self.q_norm = None
            self.k_norm = None

        self.qc_scale = 1.0

        # ---------- Rotary embeddings ----------
        if self.rope_mode == "global":
            # need C divisible for Qwen2VL-style mRoPE partition
            assert dim % 6 == 0, "dim must be divisible by 6 for global mRoPE"
            self.rotary_emb_global = Qwen2VLRotaryEmbedding(dim=dim, device=device, base=rope_theta)
            self.mrope_section_global = [dim // 2 - 2 * (dim // 6)] + [dim // 6] * 2
            self.rotary_emb_head = None
            self.mrope_section_head = None
        else:
            # per-head requires head_dim divisible by 6
            # assert self.head_dim % 6 == 0, "head_dim must be divisible by 6 for per-head mRoPE"
            self.rotary_emb_head = Qwen2VLRotaryEmbedding(dim=self.head_dim, device=device, base=rope_theta)

            self.mrope_section_head = [
                self.head_dim // 2 - (3 * self.head_dim // 8),
                (3 * self.head_dim // 16),
                (3 * self.head_dim // 16),
            ]
            self.rotary_emb_global = None
            self.mrope_section_global = None

        # guidance / extras
        self.use_e_att = False
        self.step_size = 1
        self.num_steps = 1
        self.potential_w = 1.0

        self.use_sag = False
        self.sag_dummy = nn.Identity()

    # --- replace _rmsnorm_per_head entirely ---
    def _rmsnorm_per_head(self, x: torch.Tensor, norm: nn.Module, B: int, T: int) -> torch.Tensor:
        """
        x: [B, T, C] with C = H*D
        norm: RMSNorm(D)
        returns: [B, T, C]
        """
        H, D = self.num_heads, self.head_dim
        x = rearrange(x, "b t (h d) -> (b t h) d", h=H, d=D)
        x = norm(x)  # normalize over the head_dim only
        x = rearrange(x, "(b t h) d -> b t (h d)", b=B, t=T, h=H)
        return x

    def _apply_rope_global(self, qi, ki, v_img, position_ids, B, rope_len):
        """
        qi/ki: [B, s1, C]; apply mRoPE to first 'rope_len' image tokens globally over C
        """
        # split the image part into [spatial | extra] tokens
        qi_img_spatial, qi_img_extra = qi[:, :rope_len], qi[:, rope_len:]
        ki_img_spatial, ki_img_extra = ki[:, :rope_len], ki[:, rope_len:]

        # cos/sin from positions; match previous calling convention
        cos, sin = self.rotary_emb_global(v_img, position_ids.repeat(1, B, 1))

        qi_img_spatial, ki_img_spatial = apply_multimodal_rotary_pos_emb(
            qi_img_spatial, ki_img_spatial, cos, sin, self.mrope_section_global
        )
        # stitch back
        qi = torch.cat([qi_img_spatial, qi_img_extra], dim=1)
        ki = torch.cat([ki_img_spatial, ki_img_extra], dim=1)
        return qi, ki

    # inside SpatioTemporalCrossAttention

    # --- replace _apply_rope_per_head ---
    def _apply_rope_per_head(self, qi, ki, vi, position_ids, B, rope_len):
        H, D = self.num_heads, self.head_dim
        S = min(rope_len, qi.size(1), ki.size(1), vi.size(1))

        qi_s, qi_e = qi[:, :S], qi[:, S:]
        ki_s, ki_e = ki[:, :S], ki[:, S:]

        qi_h = rearrange(qi_s, "b s (h d) -> (b h) s d", h=H, d=D)
        ki_h = rearrange(ki_s, "b s (h d) -> (b h) s d", h=H, d=D)
        v_h = rearrange(vi[:, :S, :], "b s (h d) -> (b h) s d", h=H, d=D)

        cos, sin = self.rotary_emb_head(v_h, position_ids.repeat(1, B * H, 1))
        qi_h, ki_h = apply_multimodal_rotary_pos_emb(qi_h, ki_h, cos, sin, self.mrope_section_head)

        qi_s = rearrange(qi_h, "(b h) s d -> b s (h d)", b=B, h=H)
        ki_s = rearrange(ki_h, "(b h) s d -> b s (h d)", b=B, h=H)

        return torch.cat([qi_s, qi_e], dim=1), torch.cat([ki_s, ki_e], dim=1)

    def forward(self, x, cond, attn_mask, position_ids, seps, seqlen, blur_sigma=-1, frame_mask=None):
        B, N, C = x.shape
        M = cond.shape[1]
        s1, s2 = seps  # number of image tokens, number of text tokens

        q = self.q_linear(x)  # [B, N, C]
        kv = self.kv_linear(cond)  # [B, M, 2C]
        kv = kv.reshape(B, M, 2, C)
        k, v = kv.unbind(2)  # [B, M, C], [B, M, C]

        # ---------- RMSNorm ----------
        if self.rope_mode == "global":
            q = self.q_norm(q)
            k = self.k_norm(k)
        else:
            q = self._rmsnorm_per_head(q, self.q_norm_head, B, N)
            k = self._rmsnorm_per_head(k, self.k_norm_head, B, M)

        # split q/k/v into [image | text]
        qi, qc = q[:, :s1, :], q[:, s1:, :]
        ki, kc = k[:, :s1, :], k[:, s1:, :]
        vi, vc = v[:, :s1, :], v[:, s1:, :]

        # ---------- RoPE on image spatial tokens only ----------
        rope_len = position_ids.shape[-1]  # D * HW (number of spatial image tokens)
        if self.rope_mode == "global":
            qi, ki = self._apply_rope_global(qi, ki, vi, position_ids, B, rope_len)
        else:
            qi, ki = self._apply_rope_per_head(qi, ki, vi, position_ids, B, rope_len)

        # stitch back image+text for q/k/v
        q = torch.cat([qi, qc], dim=1)
        k = torch.cat([ki, kc], dim=1)
        v = torch.cat([vi, vc], dim=1)

        # reshape for attention: [B, H, T, D]
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if blur_sigma > 0:
            q[:, :, :s1] = q[:, :, :s1] * blur_sigma

        # ---------- build SDPA mask (True = masked) ----------
        sdpa_mask = None
        if (attn_mask is not None) or (frame_mask is not None):
            if attn_mask is not None:
                allow_topology = attn_mask.to(torch.bool)[None, None, :, :].expand(B, self.num_heads, -1, -1)
            else:
                allow_topology = torch.ones(B, self.num_heads, N, M, dtype=torch.bool, device=q.device)

            if frame_mask is not None:
                tokens_per_frame = seqlen
                depth = s1 // tokens_per_frame
                img_valid = frame_mask[:, :, None].expand(B, depth, tokens_per_frame).reshape(B, s1)
                txt_valid = torch.ones(B, s2, dtype=torch.bool, device=q.device)
                token_valid = torch.cat([img_valid, txt_valid], dim=1)  # [B, N] (and M)
                allow_q = token_valid[:, None, :, None]
                allow_k = token_valid[:, None, None, :]
                allow = allow_topology & allow_q & allow_k
            else:
                allow = allow_topology
            sdpa_mask = allow

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        ckpt_act=False,
        attention_mask="causal_t2i",
        device="cpu",
        rope_theta=10000,
        rope_mode="global",
        **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm1y = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SpatioTemporalCrossAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=False,
            rope_theta=rope_theta,
            rope_mode=rope_mode,
            device=device,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * 4)  # mlp_ratio)
        self.ckpt_act = ckpt_act

        self.mlp = FeedForward(
            dim=hidden_size,
            hidden_dim=mlp_hidden_dim,
            multiple_of=256,
            ffn_dim_multiplier=None,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            FusedRMSNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        self.adaLN_ymodulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            FusedRMSNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True),
        )
        self.attention_mask = attention_mask

    def causal_first_frame_bidirectional_mask(self, x, y):
        s, hw = x.shape[-3:-1]
        sy = 77
        temporal_mask = torch.zeros(s, hw, s, hw).to(x.device)
        for i in range(s):
            for j in range(i, s):
                temporal_mask[j, :, i, :] = 1.0

        d, hw = x.shape[1:3]
        att_mask = rearrange(temporal_mask, "s1 sp1 s2 sp2 -> (s1 sp1) (s2 sp2)")

        # Add a new block of zeros to the attention mask for the text tokens
        att_mask_y = torch.zeros(att_mask.shape[0] + sy, att_mask.shape[1] + sy, device=att_mask.device)

        # Set the values in the new block to allow the text tokens to attend to themselves and the first frame to attend to the text tokens
        att_mask_y[: att_mask.shape[0], : att_mask.shape[1]] = att_mask
        att_mask_y[att_mask.shape[0] :, :hw] = 1.0  # Allow text tokens to attend to the first frame
        att_mask_y[att_mask.shape[0] :, att_mask.shape[1] :] = (
            1.0  # Allow text tokens to attend to themselves
        )
        att_mask_y[:hw, att_mask.shape[1] :] = 1.0  # Allow the first frame to attend to the text tokens

        x_seq = rearrange(x, "b d hw c -> b (d hw) c")
        xy = torch.cat([x_seq, y], dim=1)
        return xy, att_mask_y.bool()

    def causal_text_to_image_mask(self, x, y):
        s, hw = x.shape[-3:-1]
        sy = 77
        temporal_mask = torch.zeros(s, hw, s, hw).to(x.device)
        for i in range(s):
            for j in range(i, s):
                temporal_mask[j, :, i, :] = 1.0

        d, hw = x.shape[1:3]
        att_mask = rearrange(temporal_mask, "s1 sp1 s2 sp2 -> (s1 sp1) (s2 sp2)")

        # Add a new block of zeros to the attention mask for the text tokens
        att_mask_y = torch.zeros(att_mask.shape[0] + sy, att_mask.shape[1] + sy, device=att_mask.device)

        # Set the values in the new block to allow the text tokens to influence all frames, but prevent the frames from influencing the text tokens
        att_mask_y[: att_mask.shape[0], : att_mask.shape[1]] = att_mask
        att_mask_y[att_mask.shape[0] :, : att_mask.shape[1]] = (
            0.0  # Allow text tokens to influence all frames
        )
        att_mask_y[att_mask.shape[0] :, att_mask.shape[1] :] = (
            1.0  # Allow text tokens to attend to themselves
        )
        att_mask_y[: att_mask.shape[0], att_mask.shape[1] :] = (
            1.0  # Prevent frames from influencing text tokens
        )

        x_seq = rearrange(x, "b d hw c -> b (d hw) c")
        xy = torch.cat([x_seq, y], dim=1)
        return xy, att_mask_y.bool()

    def full_image_mask(self, x, y):
        s, hw = x.shape[-3:-1]
        sy = y.shape[-2]  # 77

        temporal_mask = torch.zeros(s, hw, s, hw).to(x.device)
        for i in range(s):
            for j in range(s):
                temporal_mask[j, :, i, :] = 1.0

        d, hw = x.shape[1:3]
        att_mask = rearrange(temporal_mask, "s1 sp1 s2 sp2 -> (s1 sp1) (s2 sp2)")

        # Add a new block of zeros to the attention mask for the text tokens
        att_mask_y = torch.zeros(att_mask.shape[0] + sy, att_mask.shape[1] + sy, device=att_mask.device)

        # Set the values in the new block to allow the text tokens to influence all frames, but prevent the frames from influencing the text tokens
        att_mask_y[: att_mask.shape[0], : att_mask.shape[1]] = att_mask
        att_mask_y[att_mask.shape[0] :, : att_mask.shape[1]] = (
            0.0  # Prevent frames from influencing text tokens
        )
        att_mask_y[att_mask.shape[0] :, att_mask.shape[1] :] = (
            1.0  # Allow text tokens to attend to themselves
        )
        att_mask_y[: att_mask.shape[0], att_mask.shape[1] :] = (
            1.0  # Allow text tokens to influence all frames
        )

        x_seq = rearrange(x, "b d hw c -> b (d hw) c")
        xy = torch.cat([x_seq, y], dim=1)
        return xy, att_mask_y.bool()

    def full_mask(self, x, y):
        s, hw = x.shape[-3:-1]
        sy = y.shape[-2]  # 77

        temporal_mask = torch.ones(s, hw, s, hw).to(x.device)

        d, hw = x.shape[1:3]
        att_mask = rearrange(temporal_mask, "s1 sp1 s2 sp2 -> (s1 sp1) (s2 sp2)")
        att_mask_y = torch.ones(att_mask.shape[0] + sy, att_mask.shape[1] + sy, device=att_mask.device)

        x_seq = rearrange(x, "b d hw c -> b (d hw) c")
        xy = torch.cat([x_seq, y], dim=1)
        return xy, att_mask_y.bool()

    def prepare_spatiotemporal_attention(self, x, y):
        if self.attention_mask == "causal_t2i":
            return self.causal_text_to_image_mask(x, y)
        elif self.attention_mask == "full":
            return self.full_image_mask(x, y)
        elif self.attention_mask == "all":
            return self.full_mask(x, y)
        else:
            raise ValueError(
                f"Unknown attention_mask type: '{self.attention_mask}'. Expected 'causal_t2i' or 'full'."
            )

    def _inner_forward(
        self,
        x,
        c,
        y,
        position_ids,
        blur_sigma=-1.0,
        frame_mask=None,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(
            6, dim=2
        )

        shift_y, scale_y, gate_y = self.adaLN_ymodulation(y).chunk(3, dim=2)

        x_mod = modulate(self.norm1(x), shift_msa, scale_msa)
        y_mod = self.norm1y(y) * (1 + scale_y) + shift_y

        bs, depth, seqlen, hidden_dim = x_mod.shape
        x_att, mask_att = self.prepare_spatiotemporal_attention(x, y)

        q_att = x_att

        seps = (x_mod.shape[1] * x_mod.shape[2], y_mod.shape[1])

        attn = self.attn(
            x=x_att,
            cond=q_att,
            attn_mask=mask_att,
            position_ids=position_ids,
            seps=seps,
            seqlen=seqlen,
            blur_sigma=blur_sigma,
            frame_mask=frame_mask,
        )

        dx, dy = attn[:, : seps[0]], attn[:, seps[0] :]
        dx = rearrange(dx, "b (d hw) c -> b d hw c", d=depth, hw=seqlen)

        if frame_mask is not None:
            valid_img = frame_mask[:, :, None].expand(-1, -1, seqlen)  # [B, D, HW+1]
            valid_img = valid_img[..., None]  # [B, D, HW+1, 1]
            dx = dx * valid_img

        x = x + gate_msa.unsqueeze(2) * dx
        x = x + gate_mlp.unsqueeze(2) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        y = y + gate_y * dy

        return x, y

    def forward(
        self,
        x,
        c,
        y,
        freqs_cis,
        blur_sigma=-1.0,
        frame_mask=None,
    ):
        if not self.training or not self.ckpt_act:
            return self._inner_forward(x, c, y, freqs_cis, blur_sigma, frame_mask=frame_mask)
        else:
            return cp.checkpoint(self._inner_forward, x, c, y, freqs_cis, blur_sigma, frame_mask=frame_mask)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, depth_upscale_factor, add_refiner=True):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.l1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            FusedRMSNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

        self.shuffle = PixelShuffle3d(
            upscale_factor=patch_size,
            depth_upscale_factor=depth_upscale_factor,
        )
        self.out = nn.Conv3d(
            int(hidden_size // (depth_upscale_factor * patch_size**2)),
            out_channels,
            kernel_size=(1, 1, 1),
            stride=1,
            bias=True,
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = F.silu(modulate(self.norm_final(x), shift, scale))
        x = self.l1(x)

        # rearrange as spatial grid.
        h = w = int(x.shape[2] ** 0.5)
        x = rearrange(x, "b d (h w) c -> b c d h w", h=h, w=w)
        x = self.shuffle(x)

        x = self.out(x)

        return x


class PatchEmbed(nn.Module):
    def __init__(
        self, input_size, patch_size, depth, in_channels, hidden_size, bias=True, hidden_multiplier=1
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.unshuffle = PixelUnshuffle3d(upscale_factor=patch_size, depth_upscale_factor=depth)
        self.proj = nn.Conv3d(
            in_channels * depth * patch_size**2, hidden_size, kernel_size=(1, 1, 1), bias=bias
        )

    def forward(self, x):
        x = self.unshuffle(x)
        x = self.proj(x)
        x = rearrange(x, "b c d h w -> b d (h w) c")
        return x


class FlowceptionV1(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        depth_patch_size=3,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        text_encoder_dim=None,
        text_encoder_dim_2=None,
        embed_type="default",
        act_checkpoint=False,
        rope_theta=10000,
        add_refiner=True,
        attention_mask="causal_t2i",
        device="cpu",
        repa_layer=-1,
        repa_dim=1024,
        rope_mode="global",
        h_mult=200,
        w_mult=200,
        t_mult=20,
        add_y_emb=True,
        **kwargs,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.h_mult = h_mult
        self.w_mult = w_mult
        self.t_mult = t_mult

        self.x_embedder = PatchEmbed(
            input_size,
            patch_size=patch_size,
            in_channels=in_channels * 2,
            hidden_size=hidden_size,
            depth=depth_patch_size,
            bias=True,
        )
        self.t_embedder = TimestepEmbedder(hidden_size, frequency_embedding_size=512)

        self.num_classes = num_classes
        if isinstance(num_classes, int):
            # class conditioning
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
            self.y_emb = nn.Sequential(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    FusedRMSNorm(hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
            )
        elif self.num_classes == "text_cond":
            assert text_encoder_dim is not None
            # learnable projection for text conditioning.
            self.y_embedder = None
            if text_encoder_dim_2 is None:
                self.y_emb = nn.Sequential(
                    nn.Sequential(
                        nn.Linear(text_encoder_dim, hidden_size, bias=False),
                        FusedRMSNorm(hidden_size),
                        nn.SiLU(),
                        nn.Linear(hidden_size, hidden_size, bias=False),
                    )
                )
            else:
                self.y_emb = MultiTextEmbedder(
                    text_dim_1=text_encoder_dim,
                    text_dim_2=text_encoder_dim_2,
                    hidden_size=hidden_size,
                )
        else:
            raise ValueError

        # Rotary positional embeddings for images.
        self.embed_type = embed_type

        seqlen = (input_size // patch_size) ** 2

        self.seqlen = seqlen
        self.num_heads = num_heads

        # Add rotary positional embeddings here.
        self.position_ids = None

        self.lambda_ins_tokens = nn.Parameter(torch.randn(1, 1, hidden_size), requires_grad=True)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    act_ckpt=False,
                    attention_mask=attention_mask,
                    rope_theta=rope_theta,
                    device=device,
                    rope_mode=rope_mode,
                )
                for _ in range(depth)
            ]
        )

        self.repa_layer = repa_layer
        if repa_layer >= 0:
            self.repa_proj = nn.Linear(hidden_size, repa_dim)

        self.final_layer = FinalLayer(
            hidden_size,
            patch_size,
            self.out_channels,
            depth_upscale_factor=depth_patch_size,
            add_refiner=add_refiner,
        )
        self.lambda_ins_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            FusedRMSNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

        self.initialize_weights()

    def enable_xformers_memory_efficient_attention(
        self,
    ):
        warnings.warn("enable_xformers_memory_efficient_attention is not implemented for FlowceptionV1.")

    def initialize_weights(self):
        # Initialize transformer layers:

        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize label embedding table:
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[3].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for k, block in enumerate(self.blocks):
            nn.init.normal_(block.attn.proj.weight, std=0.02)
            nn.init.constant_(block.attn.proj.bias, 0.0)

            # nn.init.constant_(block.norm1.weight, 0.0)
            # nn.init.constant_(block.norm1y.weight, 0.0)

            # nn.init.constant_(block.norm1.bias, 0.0)
            # nn.init.constant_(block.norm1y.bias, 0.0)

            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            nn.init.constant_(block.adaLN_ymodulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_ymodulation[-1].bias, 0)

            # block.mlp.reset_parameters()

        nn.init.constant_(self.final_layer.out.weight, 0.02)
        nn.init.constant_(self.final_layer.out.bias, 0)

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
        **kwargs,
    ):
        """
        Forward pass of DiT.
        x: (N, D, C, H, W) tensor of spatial inputs (frames or latent representations of frames)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # sample:         [B, D, C, H, W]
        # context_frames: [B, K, C, H, W] or None

        # 1) To PatchEmbed, you want [B, C, D, H, W]
        x = sample.permute(0, 2, 1, 3, 4).contiguous()  # -> [B, C, D, H, W]

        context_length = 0
        if context_frames is not None:
            # context_frames is already [B, K, C, H, W] coming in
            # Convert to [B, C, K, H, W] to match x’s channel-first 3D conv
            ctx = context_frames.permute(0, 2, 1, 3, 4).contiguous()  # -> [B, C, K, H, W]
            K = ctx.shape[2]
            context_length = K

            # Build the "context-only" stream with K frames filled, rest zeros
            # x has shape [B, C, D, H, W], so make a zeros volume same shape
            xz = torch.zeros_like(x)
            xz[:, :, :K] = ctx  # place the K context frames

            # Two-stream input: [context_stream | current_stream]
            # channels become 2*C (your PatchEmbed uses in_channels*2)
            x = torch.cat([xz, x], dim=1)  # concat on channel axis

        t = timestep
        y = class_labels

        x = self.x_embedder(x)  # (N, T, D), where T = H * W / patch_size ** 2

        b, d = t.shape
        t = rearrange(t, "b d-> (b d)")
        t = self.t_embedder(t)
        t = rearrange(t, "(b d) h -> b d h", b=b, d=d)

        B, D, HW, C = x.shape
        frame_tok = self.lambda_ins_tokens.expand(B, D, 1, C)  # [B, D, 1, C]
        x = torch.cat([x, frame_tok], dim=2)  # [B, D, HW+1, C]

        if self.y_embedder is not None:
            y = self.y_embedder(y, self.training)
        if isinstance(y, torch.Tensor) and y.ndim == 2:
            y = y.unsqueeze(1)  # (N, S, D)

        y = self.y_emb(y)
        c = t + y.mean(1, keepdim=True)  ##+ c_add

        # precompute position ids and register them as buffers.
        _, depth, _, height, width = sample.shape
        # depth += context_length
        if self.position_ids is None or self.position_ids.shape[-1] != depth * (height // self.patch_size) * (
            width // self.patch_size
        ):
            self.position_ids = generate_position_ids(
                depth=depth,
                height=height // self.patch_size,
                width=width // self.patch_size,
                device=x.device,
                h_mult=self.h_mult,
                w_mult=self.w_mult,
                t_mult=self.t_mult,
            )

        means = []
        means_y = []

        repa_out = None

        for i, block in enumerate(self.blocks):
            if i > seg_start and i < seg_end:
                x, y = block(x, c, y, self.position_ids, blur_sigma, frame_mask=frame_mask)  # (N, T, D)
            else:
                x, y = block(x, c, y, self.position_ids, -1.0, frame_mask=frame_mask)  # (N, T, D)

            means.append(x.norm(2))
            means_y.append(y.norm(2))

            if self.repa_layer >= 0 and i == self.repa_layer:
                repa_out = self.repa_proj(x)[:, context_length:]
                rh = int(self.input_size / self.patch_size)
                rw = int(self.input_size / self.patch_size)
                repa_out = rearrange(repa_out, "b d (h w) c -> b c d h w", h=rh, w=rw)

        x_img = x[:, :, :-1, :]  # spatial tokens for decoding
        x_cls = x[:, :, -1, :]  # per-frame token, shape [B, D, C]

        vel = self.final_layer(x_img, c)  # [B, out_ch, D_up, H, W]
        logits = self.lambda_ins_head(x_cls)  # [B, D, 1]

        if not return_means:
            return vel, logits
        else:
            return vel, logits, repa_out, means, means_y


def FlowceptionV1_W_2(**kwargs):
    return FlowceptionV1(depth=38, hidden_size=2736, patch_size=2, num_heads=38, **kwargs)


def FlowceptionV1_G_2(**kwargs):
    return FlowceptionV1(depth=28, hidden_size=3584, patch_size=2, num_heads=28, **kwargs)


def FlowceptionV1_H_2(**kwargs):
    return FlowceptionV1(depth=38, hidden_size=1536, patch_size=2, num_heads=24, **kwargs)


def FlowceptionV1_H_1(**kwargs):
    return FlowceptionV1(depth=38, hidden_size=1536, patch_size=1, num_heads=24, **kwargs)


def FlowceptionV1_XL_2(**kwargs):
    return FlowceptionV1(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def FlowceptionV1_XL_1(**kwargs):
    return FlowceptionV1(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)


def FlowceptionV1_Tiny_1(**kwargs):
    """Tiny model for debugging on toy data (e.g. 3×3 coloring videos).

    ~2M params — runs comfortably on a single GPU with large batch sizes.
    patch_size=1 because the spatial resolution is only 3×3.
    depth_patch_size is overridden to 1 (identity VAE, no temporal patchify).
    hidden_size=240 so that dim is divisible by 6 (global mRoPE) and 8 (num_heads).
    """
    kwargs.setdefault("depth_patch_size", 1)
    return FlowceptionV1(
        depth=6,
        hidden_size=240,
        patch_size=1,
        num_heads=8,
        **kwargs,
    )


FlowceptionV1_models = {
    "FlowceptionV1-W/2": FlowceptionV1_W_2,
    "FlowceptionV1-G/2": FlowceptionV1_G_2,
    "FlowceptionV1-H/2": FlowceptionV1_H_2,
    "FlowceptionV1-H/1": FlowceptionV1_H_1,
    "FlowceptionV1-XL/2": FlowceptionV1_XL_2,
    "FlowceptionV1-XL/1": FlowceptionV1_XL_1,
    "FlowceptionV1-Tiny/1": FlowceptionV1_Tiny_1,
}
