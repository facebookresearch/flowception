"""Denoiser output scaling functions.

Computes (c_skip, c_out, c_in, c_noise) from sigma, which control how
the denoiser output is combined with the input to produce the final prediction.
"""

import torch


class CondOTScaling:
    """Conditional Optimal Transport scaling — used by Flowception."""

    def __call__(self, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t = 1 / (sigma + 1.0)
        c_skip = t
        c_out = 1.0 - t
        c_in = t
        c_noise = t
        return c_skip, c_out, c_in, c_noise
