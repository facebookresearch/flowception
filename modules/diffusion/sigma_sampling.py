"""Sigma schedule discretization for the diffusion process."""

from abc import abstractmethod

import torch


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def rescale_sigma(t, s):
    """Rescale timestep t by sigma_scale factor s."""
    return t / (1 + (s - 1) * (1 - t))


class Discretization:
    """Base class for noise schedule discretization."""

    def __call__(self, n, do_append_zero=True, device="cpu", flip=False):
        sigmas = self.get_sigmas(n, device=device)
        sigmas = append_zero(sigmas) if do_append_zero else sigmas
        return sigmas if not flip else torch.flip(sigmas, (0,))

    @abstractmethod
    def get_sigmas(self, n, device):
        pass


class CondOTUniformDiscretization(Discretization):
    """Conditional Optimal Transport discretization with uniform spacing."""

    def __init__(self, sigma_scale=1.0):
        self.sigma_scale = sigma_scale

    def get_sigmas(self, n, device="cpu"):
        t = torch.linspace(1 / n, 1, n, device=device)
        t = rescale_sigma(t, self.sigma_scale)
        sigma = (1.0 - t) / (1e-7 + t)
        return sigma
