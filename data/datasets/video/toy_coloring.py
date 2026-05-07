"""Toy coloring-video dataset for debugging and small-scale experiments.

Generates variable-length 3×3 RGB "videos" where boundary pixels rotate through
a colour palette.  No files on disk are required — everything is synthesised on
the fly, making this dataset ideal for unit tests and quick integration checks.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class Datapoint(dict):
    """Dict subclass with attribute access — compatible with both
    PyTorch's default_collate (sees a dict) and the engine (uses dot-access)."""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def generate_projected_coloring_video_batch(
    batch_size: int = 16,
    T_max: int = 30,
    device: str = "cpu",
    padding_idx: float = 3.0,
    min_length: int = 5,
    max_length: int = 30,
    valid_stride_choices: tuple = (1, 2, 3),
):
    """Generate variable-length 3×3 coloring videos with rotating boundary colours."""
    boundary_path = torch.tensor([0, 1, 2, 5, 8, 7, 6, 3], device=device)
    flat_to_ij = torch.tensor(
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
        device=device,
    )
    center_idx = 4

    valid_lengths = torch.arange(min_length, max_length + 1, step=5, device=device)
    video_lengths = valid_lengths[torch.randint(0, len(valid_lengths), (batch_size,), device=device)]
    stride_choices = torch.tensor(valid_stride_choices, device=device)
    strides = stride_choices[torch.randint(0, len(stride_choices), (batch_size,), device=device)]

    c1 = torch.rand(batch_size, 3, device=device)[..., None]
    c2 = torch.rand(batch_size, 3, device=device)[..., None]
    dx = torch.linspace(0, 1, steps=9, device=device)[None, None, :]
    color_seq = (c1 * dx + c2 * (1 - dx)).permute(0, 2, 1)  # [B, 9, 3]
    boundary_colors = color_seq[:, :8, :]  # [B, 8, 3]
    center_colors = color_seq[:, 8, :]  # [B, 3]

    videos = torch.full(
        (batch_size, T_max, 3, 3, 3),
        fill_value=padding_idx,
        device=device,
        dtype=torch.float32,
    )

    t = torch.arange(T_max, device=device).unsqueeze(0).expand(batch_size, -1)
    mask = t < video_lengths.unsqueeze(1)
    shift = (t * strides.unsqueeze(1)) % 8

    base_color_idx = torch.arange(8, device=device)
    all_color_rolls = torch.stack([torch.roll(base_color_idx, -s) for s in range(8)], dim=0)
    rolled_color_idx = all_color_rolls[shift]  # [B, T, 8]

    for pi in range(8):
        grid_idx = boundary_path[pi]
        i, j = flat_to_ij[grid_idx]
        color_idx = rolled_color_idx[:, :, pi]
        gathered_color = torch.gather(boundary_colors, 1, color_idx.unsqueeze(-1).expand(-1, -1, 3))
        target = videos[:, :, :, i, j]
        target.masked_scatter_(mask.unsqueeze(-1), gathered_color[mask])

    i, j = flat_to_ij[center_idx]
    center_expanded = center_colors.unsqueeze(1).expand(-1, T_max, -1)
    center_target = videos[:, :, :, i, j]
    center_target.masked_scatter_(mask.unsqueeze(-1), center_expanded[mask])

    return videos, mask, video_lengths, strides


class ToyColoringDataset(Dataset):
    """On-the-fly toy coloring-video dataset.

    Each sample is a short variable-length 3×3 RGB video normalised to [-1, 1].
    The dataset returns ``Datapoint`` dicts compatible with the Flowception
    training pipeline, so it can be used as a drop-in replacement for real
    video datasets.

    Parameters
    ----------
    num_frames : int
        Maximum number of frames per video (T_max).
    min_length, max_length : int
        Range of *actual* video lengths (sampled uniformly from multiples of 5).
    height, width : int
        Spatial resolution — kept for API compatibility but always 3×3 internally.
    num_start_frames : int
        Number of seed frames for Flowception (used only for the length check).
    latent_downsample : int
        Temporal down-sample factor of the VAE. For the identity VAE this is 1.
    length : int
        Virtual dataset length (streaming-style).
    padding_idx : float
        Value used for padding absent frames.
    """

    def __init__(
        self,
        num_frames: int = 30,
        min_length: int = 15,
        max_length: int = 30,
        height: int = 3,
        width: int = 3,
        num_start_frames: int = 3,
        latent_downsample: int = 1,
        length: int = 2_000_000,
        padding_idx: float = 3.0,
        **kwargs,
    ):
        super().__init__()
        self.num_frames = int(num_frames)
        self.min_length = int(min_length)
        self.max_length = int(max_length)
        self.height = int(height)
        self.width = int(width)
        self.num_start_frames = int(num_start_frames)
        self.latent_downsample = int(latent_downsample)
        self._length = int(length)
        self.padding_idx = float(padding_idx)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        T = self.num_frames
        videos, mask, lengths, strides = generate_projected_coloring_video_batch(
            batch_size=1,
            T_max=T,
            device="cpu",
            padding_idx=self.padding_idx,
            min_length=self.min_length,
            max_length=self.max_length,
            valid_stride_choices=(1, 2, 3),
        )

        # videos: [1, T, C, H, W]  in [0, 1]  →  normalise to [-1, 1]
        vid = videos[0]  # [T, C, H, W]
        vid = vid * 2.0 - 1.0  # [0,1] → [-1,1]

        frame_mask = mask[0]  # [T] bool
        video_length = int(lengths[0].item())

        # For identity VAE the latent length equals video length
        ld = self.latent_downsample
        latent_length = 1 + (video_length - 1) // ld if ld > 1 else video_length

        # img_tensor: [C, T, H, W]  (channel-first for the VAE / pipeline)
        img_tensor = vid.permute(1, 0, 2, 3).contiguous()  # [C, T, H, W]
        anchor_tensor = img_tensor[:, :1, :, :].contiguous()  # [C, 1, H, W]

        crop_coords = torch.zeros(8)

        return Datapoint(
            pixel_values=img_tensor,
            condition={
                "class_id": "",
                "caption_idx": torch.tensor(0),
                "crop_coords": crop_coords,
                "anchor_frame": anchor_tensor,
                "frame_mask": frame_mask,
                "video_length": torch.tensor(video_length),
                "latent_length": torch.tensor(latent_length),
                "stride": torch.tensor(1),
                "frame_indices": torch.arange(T),
            },
        )
