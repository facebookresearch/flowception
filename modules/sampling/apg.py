import torch
import torch.nn as nn


class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        # Handle size mismatch by padding/truncating along frame dimension
        if isinstance(self.running_average, torch.Tensor):
            old_len = self.running_average.shape[1]
            new_len = update_value.shape[1]
            if old_len != new_len:
                if new_len > old_len:
                    # Pad with zeros
                    padding = torch.zeros(
                        self.running_average.shape[0],
                        new_len - old_len,
                        *self.running_average.shape[2:],
                        device=self.running_average.device,
                        dtype=self.running_average.dtype,
                    )
                    self.running_average = torch.cat([self.running_average, padding], dim=1)
                else:
                    # Truncate
                    self.running_average = self.running_average[:, :new_len]

        new_average = self.momentum * self.running_average
        self.running_average = new_average + update_value

    def clear(self):
        self.running_average = 0


def project(
    v0: torch.Tensor,
    v1=torch.Tensor,
):
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def normalized_guidance(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale: float,
    momentum_buffer: MomentumBuffer = None,
    eta: float = 1.0,
    norm_threshold: float = 0.0,
):
    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    diff_parallel, diff_orthogonal = project(diff, pred_cond)

    # dp = diff_parallel.norm(2, dim=(-1,-2,-3)).cpu().numpy()
    # do = diff_orthogonal.norm(2, dim=(-1,-2,-3)).cpu().numpy()

    normalized_update = diff_orthogonal + eta * diff_parallel
    pred_guided = pred_cond + (guidance_scale - 1) * normalized_update
    # pred_guided = guidance_scale * pred_cond + (1-guidance_scale) * pred_uncond
    return pred_guided


class APGGuider(nn.Module):
    def __init__(self, eta, momentum, norm_threshold):
        super().__init__()
        self.eta = eta
        self.momentum = momentum
        self.norm_threshold = norm_threshold
        self.momentum_buffer = MomentumBuffer(momentum)

    def clean_buffer(self):
        self.momentum_buffer.running_average = 0.0

    def forward(self, x, y, guidance_scale):
        return normalized_guidance(
            pred_cond=x,
            pred_uncond=y,
            guidance_scale=guidance_scale,
            eta=self.eta,
            momentum_buffer=self.momentum_buffer,
            norm_threshold=self.norm_threshold,
        )
