
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


def _default_inv_freq(dim: int, base: float, device):
    """Compute the default RoPE inverse-frequency band (no dynamic scaling)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim))
    return inv_freq


class Qwen2VLRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
    ):
        super().__init__()
        self.rope_type = rope_type
        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings
        self.attention_scaling = 1.0

        inv_freq = _default_inv_freq(dim, base, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer(
                "inv_freq", inv_freq, persistent=False
            ) 
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len
        ):  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block. In contrast to other models, Qwen2_VL has different position ids for thw grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
        # import torch; torch.set_printoptions(profile='full')
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=None):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension seperately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1)

    if unsqueeze_dim:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_m_modify_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """
    VideoRoPE-style: combines h+w, puts temporal last (low frequency).
    """
    # Combine h+w into one section
    mrope_section = [mrope_section[0], mrope_section[1] + mrope_section[2]]
    mrope_section = mrope_section * 2
    # Reverse: puts temporal last (low freq)
    mrope_section = mrope_section[::-1]

    index = 0
    result_cos = []
    result_sin = []

    # Interleave h and w (spatial), then temporal
    for i, section in enumerate(mrope_section):
        if i % 2 == 0:  # Spatial section (h and w interleaved)
            for j in range(section):
                row = 1 if j % 2 == 0 else 2  # Alternate h (1) and w (2)
                result_cos.append(cos[row, ..., index : index + 1])
                result_sin.append(sin[row, ..., index : index + 1])
                index += 1
        else:  # Temporal section
            result_cos.append(cos[0, ..., index : index + section])
            result_sin.append(sin[0, ..., index : index + section])
            index += section

    cos = torch.cat(result_cos, dim=-1).unsqueeze(dim=unsqueeze_dim)
    sin = torch.cat(result_sin, dim=-1).unsqueeze(dim=unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def generate_position_ids(depth, height, width, device, t_mult=20, h_mult=200, w_mult=200):
    ids = torch.zeros(3, depth, height, width, device=device, requires_grad=False)
    for t in range(depth):
        ids[0, t, :, :] = t * t_mult
    for h in range(height):
        ids[1, :, h, :] = h * int(h_mult * 8 / height)
    for w in range(width):
        ids[2, :, :, w] = w * int(w_mult * 8 / width)
    position_ids = rearrange(ids, "o t h w -> o (t h w)").unsqueeze(1)  # .repeat(1, batch_size, 1)

    return position_ids
