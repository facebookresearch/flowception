import torch.nn as nn
import torch
import torch._dynamo as dynamo

from einops import rearrange


class PixelShuffle3d(nn.Module):
    def __init__(self, upscale_factor: int, depth_upscale_factor: int):
        super().__init__()
        self.r = upscale_factor
        self.rd = depth_upscale_factor

    def forward(self, x):
        B, C, D, H, W = x.shape
        r, rd = self.r, self.rd
        assert (C % (r * r * rd)) == 0, f"channels {C} not divisible by {r * r * rd}"
        Cout = C // (r * r * rd)
        # rearrange is compile-friendly with SymInt shapes
        # (B, Cout*rd*r*r, D, H, W) -> (B, Cout, D*rd, H*r, W*r)
        y = rearrange(x, "b (c rd r1 r2) d h w -> b c (d rd) (h r1) (w r2)", c=Cout, rd=rd, r1=r, r2=r)
        return y


class PixelUnshuffle3d(nn.Module):
    def __init__(self, upscale_factor: int, depth_upscale_factor: int):
        super().__init__()
        self.r = upscale_factor
        self.rd = depth_upscale_factor

    def forward(self, x):
        B, C, D, H, W = x.shape
        r, rd = self.r, self.rd
        assert D % rd == 0 and H % r == 0 and W % r == 0, "incompatible sizes for unshuffle"
        # (B, C, D, H, W) -> (B, C*rd*r*r, D/rd, H/r, W/r)
        y = rearrange(x, "b c (d rd) (h r1) (w r2) -> b (c rd r1 r2) d h w", rd=rd, r1=r, r2=r)
        return y


from liger_kernel.ops.rms_norm import LigerRMSNormFunction
from torch import Tensor, nn


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class FusedRMSNorm(RMSNorm):
    def forward(self, x: Tensor):
        return LigerRMSNormFunction.apply(
            x,
            self.scale,
            1e-6,
            0.0,
            "llama",
            False,
        )


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)
