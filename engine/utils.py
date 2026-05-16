"""Shared utility functions for Flowception training engines."""

import inspect
import math
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def vae_accepts_latent_mask(vae) -> bool:
    """Check if the VAE decoder supports a latent_mask argument."""
    if hasattr(vae, "decoder"):
        try:
            return "latent_mask" in inspect.signature(vae.decoder.forward).parameters
        except Exception:
            pass
    return False


def randomly_select_slice(x):
    """Randomly select either first or last D-1 frames from a [B,C,D,H,W] tensor."""
    B, C, D, H, W = x.shape
    start_idx = torch.randint(0, 2, (B,), device=x.device)
    offsets = torch.arange(D - 1, device=x.device)
    indices = start_idx[:, None] + offsets[None, :]
    indices = indices[:, None, :, None, None].expand(-1, C, -1, H, W)
    return torch.gather(x, dim=2, index=indices)


def get_ema_loss_heatmap_to_wandb(
    ema_tracker,
    step=None,
    tag="ema/loss_heatmap",
    num_bins=5,
    num_frames=8,
    accelerator=None,
):
    """Build a wandb heatmap image from EMA loss tracker (bin × frame)."""
    mat = np.full((num_bins, num_frames), np.nan)
    for bin_idx in range(num_bins):
        for frame_idx in range(num_frames):
            if bin_idx in ema_tracker and frame_idx in ema_tracker[bin_idx]:
                mat[bin_idx, frame_idx] = ema_tracker[bin_idx][frame_idx]

    fig, ax = plt.subplots(figsize=(num_frames, num_bins // 2 + 1))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Timestep Bin")
    ax.set_title("EMA Loss (Bin x Frame)")

    for i in range(num_bins):
        for j in range(num_frames):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )
    plt.colorbar(im, ax=ax, label="EMA Loss")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    return {tag: wandb.Image(img)}


def update_ema_loss_bins(ema_tracker, loss_tensor, timesteps, frame_axis=1, alpha=0.95):
    """Update EMA-tracked losses per timestep bin and frame index."""
    ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    bin_ids = torch.bucketize(
        timesteps,
        boundaries=torch.tensor([r[1] for r in ranges], device=timesteps.device),
    )
    for bin_idx in range(len(ranges)):
        bin_mask = bin_ids == bin_idx
        if not bin_mask.any():
            continue
        for t in range(loss_tensor.shape[frame_axis]):
            frame_losses = loss_tensor[bin_mask, t]
            if frame_losses.numel() == 0:
                continue
            mean_loss = frame_losses.mean().item()
            if bin_idx not in ema_tracker:
                ema_tracker[bin_idx] = {}
            if t not in ema_tracker[bin_idx]:
                ema_tracker[bin_idx][t] = mean_loss
            else:
                ema_tracker[bin_idx][t] = alpha * ema_tracker[bin_idx][t] + (1 - alpha) * mean_loss
    return ema_tracker


def sample_tau_g_edge_equalized(batch_size: int, device, eps: float = 1e-2) -> torch.Tensor:
    """Sample tau_g in [eps, 2-eps] with density q(beta) ∝ 1/min(beta, 2-beta)."""
    B = batch_size
    u_side = torch.rand(B, 1, device=device)
    u = torch.rand(B, 1, device=device)
    log_inv_eps = torch.log(torch.tensor(1.0 / eps, device=device))
    beta_left = eps * torch.exp(u * log_inv_eps)
    beta = torch.where(u_side < 0.5, beta_left, 2.0 - beta_left)
    return beta


def logit_normal_weight(t, mu=0.0, sigma=1.0, eps=1e-6):
    """Compute logit-normal importance weight for timestep t."""
    t = t.clamp(eps, 1.0 - eps)
    logit_t = torch.log(t) - torch.log1p(-t)
    z = (logit_t - mu) / sigma
    w = torch.exp(-0.5 * z * z) / (
        sigma * torch.sqrt(torch.tensor(2.0 * torch.pi, device=t.device)) * t * (1 - t)
    )
    w = w.clamp(min=0.05, max=None)
    return w


def scale_snr(u, sigma_scale):
    """Rescale uniform samples by signal-to-noise ratio."""
    return u / (1 + (sigma_scale - 1) * (1 - u))


def frame_cov_full_masked(x: torch.Tensor, m: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Mask-aware per-video full channel covariance over frames + spatial dims.

    Args:
        x: [B, F, C, H, W]
        m: [B, F] boolean frame mask (True=valid)
    Returns:
        cov: [B, C, C]
    """
    B, F, C, H, W = x.shape
    v = x.permute(0, 2, 1, 3, 4).reshape(B, C, F * H * W)
    mask4 = m.bool()[:, :, None, None].float()
    mask5 = mask4[:, None].expand(-1, 1, -1, H, W)
    mask_flat = mask5.reshape(B, 1, F * H * W)
    denom = mask_flat.sum(dim=-1).clamp_min(1.0)
    num = (v * mask_flat).sum(dim=-1)
    mean = num / denom
    v_centered = v - mean.unsqueeze(-1)
    v_weighted = v_centered * mask_flat
    cov = torch.bmm(v_weighted, v_centered.transpose(1, 2))
    cov = cov / denom.unsqueeze(-1)
    eye = torch.eye(C, device=x.device, dtype=x.dtype).unsqueeze(0)
    return cov + eps * eye


def downsample_video(img: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """Downsample [B,C,T,H,W] video to [B,C,T,out_h,out_w]."""
    B, C, T, H, W = img.shape
    if (H, W) == (out_h, out_w):
        return img
    frames = img.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W).contiguous()
    frames = torch.nn.functional.interpolate(
        frames, size=(out_h, out_w), mode="bilinear", align_corners=False
    )
    return frames.reshape(B, T, C, out_h, out_w).permute(0, 2, 1, 3, 4).contiguous()


def _min_zoom_to_avoid_padding(theta_rad: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Compute the minimum zoom factor to avoid padding after rotation by theta_rad."""
    c = torch.cos(theta_rad).abs()
    s = torch.sin(theta_rad).abs()
    Hf = float(H)
    Wf = float(W)
    z = torch.maximum(c + s * (Hf / Wf), c + s * (Wf / Hf))
    return torch.maximum(z, torch.ones_like(z))


class GPUBatchVideoAug:
    """GPU-based batched video augmentation: hflip, spatial affine, color jitter."""

    def __init__(
        self,
        hflip_p=0.5,
        scale_range=(1.0, 1.3),
        rot_deg=15.0,
        trans_px=8,
        padding_mode="border",
        align_corners=False,
        brightness=0.15,
        contrast=0.35,
        saturation=0.3,
        hue_deg=3.0,
        gamma=0.15,
        color_p=0.5,
        spatial_p=1.0,
    ):
        self.hflip_p = hflip_p
        self.scale_range = scale_range
        self.rot_deg = rot_deg
        self.trans_px = trans_px
        self.padding_mode = padding_mode
        self.align_corners = align_corners

        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue_deg = hue_deg
        self.gamma = gamma
        self.color_p = color_p
        self.spatial_p = spatial_p

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        """
        x: [B,C,T,H,W] in [-1,1], on GPU
        returns augmented x, same shape/range
        """
        B, C, T, H, W = x.shape
        device = x.device
        dtype = x.dtype

        # --------- HFLIP (batched select) ----------
        if self.hflip_p > 0:
            flip = (torch.rand((B,), generator=gen, device=device) < self.hflip_p).view(B, 1, 1, 1, 1)
            x_flip = x.flip(-1)
            x = torch.where(flip, x_flip, x)

        # --------- SPATIAL AFFINE (single grid_sample over B*T) ----------
        if self.spatial_p > 0 and torch.rand((), generator=gen, device=device).item() < self.spatial_p:
            zoom = torch.empty((B,), device=device, dtype=torch.float32).uniform_(
                self.scale_range[0], self.scale_range[1], generator=gen
            )

            ang_deg = (torch.rand((B,), device=device, generator=gen) * 2 - 1) * float(self.rot_deg)
            ang_fwd = ang_deg * (math.pi / 180.0)

            z_min = _min_zoom_to_avoid_padding(ang_fwd, H, W)
            zoom = torch.maximum(zoom, z_min * 1.01)

            a = 1.0 / zoom

            ang = -ang_fwd
            ca = torch.cos(ang)
            sa = torch.sin(ang)

            cropW = W / zoom
            cropH = H / zoom

            x0 = torch.rand((B,), device=device, generator=gen) * (W - cropW)
            y0 = torch.rand((B,), device=device, generator=gen) * (H - cropH)

            cx = x0 + 0.5 * cropW
            cy = y0 + 0.5 * cropH

            if self.align_corners:
                txn = 2.0 * cx / (W - 1) - 1.0
                tyn = 2.0 * cy / (H - 1) - 1.0
            else:
                txn = (2.0 * cx + 1.0) / W - 1.0
                tyn = (2.0 * cy + 1.0) / H - 1.0

            theta = torch.zeros((B, 2, 3), device=device, dtype=torch.float32)
            theta[:, 0, 0] = ca * a
            theta[:, 0, 1] = -sa * a
            theta[:, 1, 0] = sa * a
            theta[:, 1, 1] = ca * a
            theta[:, 0, 2] = txn
            theta[:, 1, 2] = tyn

            frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W).contiguous()
            theta_bt = theta[:, None].expand(B, T, 2, 3).reshape(B * T, 2, 3).contiguous()

            grid = F.affine_grid(theta_bt.to(dtype), frames.size(), align_corners=self.align_corners)
            frames = F.grid_sample(
                frames,
                grid,
                mode="bilinear",
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )
            x = frames.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

        # --------- COLOR (vectorized, per-clip params, no flicker) ----------
        if self.color_p > 0 and torch.rand((), generator=gen, device=device).item() < self.color_p:
            y = (x * 0.5 + 0.5).clamp(0.0, 1.0).to(torch.float32)

            b = (torch.rand((B,), generator=gen, device=device) * 2 - 1) * float(self.brightness)
            c = 1.0 + (torch.rand((B,), generator=gen, device=device) * 2 - 1) * float(self.contrast)
            s = 1.0 + (torch.rand((B,), generator=gen, device=device) * 2 - 1) * float(self.saturation)
            h = (
                (torch.rand((B,), generator=gen, device=device) * 2 - 1)
                * float(self.hue_deg)
                * (math.pi / 180.0)
            )
            g = 1.0 + (torch.rand((B,), generator=gen, device=device) * 2 - 1) * float(self.gamma)

            b = b.view(B, 1, 1, 1, 1)
            c = c.view(B, 1, 1, 1, 1)
            s = s.view(B, 1, 1, 1, 1)
            g = g.view(B, 1, 1, 1, 1)

            y = y.pow(g).clamp(0, 1)
            y = y + b
            mean = y.mean(dim=(1, 2, 3, 4), keepdim=True)
            y = (y - mean) * c + mean

            if C == 3:
                gray = (0.299 * y[:, 0] + 0.587 * y[:, 1] + 0.114 * y[:, 2]).unsqueeze(1)
                y = gray + s * (y - gray)

                cosh = torch.cos(h).view(B, 1, 1, 1)
                sinh = torch.sin(h).view(B, 1, 1, 1)
                R, G, Bc = y[:, 0], y[:, 1], y[:, 2]
                Yl = 0.299 * R + 0.587 * G + 0.114 * Bc
                I = 0.596 * R - 0.274 * G - 0.322 * Bc
                Q = 0.211 * R - 0.523 * G + 0.312 * Bc
                I2 = I * cosh - Q * sinh
                Q2 = I * sinh + Q * cosh
                R2 = Yl + 0.956 * I2 + 0.621 * Q2
                G2 = Yl - 0.272 * I2 - 0.647 * Q2
                B2 = Yl - 1.106 * I2 + 1.703 * Q2
                y = torch.stack([R2, G2, B2], dim=1)

            y = y.clamp(0.0, 1.0).to(dtype)
            x = y * 2.0 - 1.0

        return x
