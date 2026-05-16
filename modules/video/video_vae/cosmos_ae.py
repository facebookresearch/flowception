import torch
import torch.nn as nn
from dataclasses import dataclass
from diffusers.utils import BaseOutput

try:
    from cosmos_tokenizer.video_lib import CausalVideoTokenizer
except ImportError:
    CausalVideoTokenizer = None
import torch.nn.functional as F


@dataclass
class DecoderOutput(BaseOutput):
    r"""
    Output of decoding method.

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The decoded output sample from the last layer of the model.
    """

    sample: torch.FloatTensor


class DummyLatentDist:
    def __init__(self, latents):
        self.latents = latents

    def forward(self):
        return self.latents

    def sample(
        self,
    ):
        return self.forward()

    def mode(
        self,
    ):
        return self.forward()


@dataclass
class DecoderLatentDist(BaseOutput):
    r"""
    Output of decoding method.

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The decoded output sample from the last layer of the model.
    """

    latent_dist: DummyLatentDist


class CosmosVAE(nn.Module):
    def __init__(self, model_name, weights_dir, device, dtype="bf16"):
        super().__init__()
        self.encoder = torch.compile(
            CausalVideoTokenizer(checkpoint_enc=f"{weights_dir}/{model_name}/encoder.jit", device=device)
        )
        self.decoder = torch.compile(
            CausalVideoTokenizer(checkpoint_dec=f"{weights_dir}/{model_name}/decoder.jit", device=device)
        )

    def encode(self, x, temporal_chunk=False):
        return DecoderLatentDist(DummyLatentDist(self.encoder.encode(x)[0]))

    def decode(self, x, temporal_chunk=False):
        return DecoderOutput(self.decoder.decode(x))


from dataclasses import dataclass
from diffusers.utils import BaseOutput

# ------------------ diffusers-style outputs ------------------


@dataclass
class DecoderOutput(BaseOutput):
    sample: torch.FloatTensor  # [B, C, T, H, W] in [-1,1] (or whatever your Cosmos returns)


class _DummyLatentDist:
    def __init__(self, latents: torch.Tensor):
        self._latents = latents

    def forward(self):
        return self._latents

    def sample(self):
        return self._latents

    def mode(self):
        return self._latents


@dataclass
class DecoderLatentDist(BaseOutput):
    latent_dist: _DummyLatentDist  # holder to mimic diffusers VAE API


def space_to_depth_latents(z: torch.Tensor, s: int) -> torch.Tensor:
    """
    z: [B, C, T, H, W]  (latents from Cosmos encoder)
    returns: [B, C*s*s, T, H//s, W//s]
    """
    if s == 1:
        return z
    B, C, T, H, W = z.shape
    assert H % s == 0 and W % s == 0, f"latent H/W must be divisible by s={s}"
    # Use pixel_unshuffle per time-slice (F.pixel_unshuffle expects [B,C,H,W])
    z2 = z.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)  # [B*T, C, H, W]
    z2 = F.pixel_unshuffle(z2, downscale_factor=s)  # [B*T, C*s*s, H//s, W//s]
    z2 = z2.view(B, T, C * s * s, H // s, W // s).permute(0, 2, 1, 3, 4).contiguous()
    return z2


def depth_to_space_latents(z: torch.Tensor, s: int) -> torch.Tensor:
    """
    inverse of space_to_depth_latents
    z: [B, C*s*s, T, H//s, W//s]  -> [B, C, T, H, W]
    """
    if s == 1:
        return z
    B, C2, T, H2, W2 = z.shape
    assert C2 % (s * s) == 0, f"channels must be divisible by s^2={s * s}"
    C = C2 // (s * s)
    z2 = z.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C2, H2, W2)  # [B*T, C*s*s, H2, W2]
    z2 = F.pixel_shuffle(z2, upscale_factor=s)  # [B*T, C, H2*s, W2*s]
    z2 = z2.view(B, T, C, H2 * s, W2 * s).permute(0, 2, 1, 3, 4).contiguous()
    return z2


class CosmosVAE_Space2Depth(nn.Module):
    """
    Cosmos VAE + optional extra spatial downsample on LATENTS via pixel unshuffle/shuffle.
    Effective spatial downsample = 8 * s_extra.
    - encode(): runs Cosmos, then space-to-depth on latents
    - decode(): depth-to-space on latents, then Cosmos decoder
    This is lossless (invertible) and requires no retraining.
    """

    def __init__(self, base_vae: CosmosVAE, s_extra: int = 4):
        super().__init__()
        assert s_extra >= 1 and int(s_extra) == s_extra
        self.vae = base_vae
        self.s_extra = int(s_extra)
        self.effective_spatial_factor = self.vae.spatial_factor * self.s_extra

    @torch.no_grad()
    def encode(self, x, temporal_chunk: bool = False) -> DecoderLatentDist:
        """
        x: [B, C, T, H, W], H/W must be divisible by (8 * s_extra)
        returns latent_dist.sample(): [B, C_lat*s_extra^2, T_lat, H/(8*s_extra), W/(8*s_extra)]
        """
        B, C, T, H, W = x.shape
        eff = self.effective_spatial_factor
        if (H % eff) or (W % eff):
            raise ValueError(f"Input H/W must be divisible by 8*s_extra={eff}, got H={H}, W={W}")
        # base encode
        z = self.vae.encode(x, temporal_chunk=temporal_chunk).latent_dist.sample()  # [B, C_lat, T, H/8, W/8]
        # extra S2D on latents
        z_s2d = space_to_depth_latents(z, self.s_extra)  # [B, C_lat*s^2, T, H/(8*s), W/(8*s)]
        return DecoderLatentDist(_DummyLatentDist(z_s2d))

    @torch.no_grad()
    def decode(self, z_s2d, temporal_chunk: bool = False) -> DecoderOutput:
        """
        z_s2d: [B, C_lat*s_extra^2, T_lat, H/(8*s_extra), W/(8*s_extra)]
        returns: [B, C, T, H, W]
        """
        # invert S2D on latents
        z = depth_to_space_latents(z_s2d, self.s_extra)  # [B, C_lat, T, H/8, W/8]
        x = self.vae.decode(z, temporal_chunk=temporal_chunk).sample
        return DecoderOutput(x)
