import torch


def poisson_loss(rate, k, M, tau_0):
    # inv_ratio = torch.where(tau_0 < 1, 1 - tau_0, 0.0)
    loss = rate - k * torch.log(rate + 1e-8)
    # ratio = torch.where(tau_0 < 1, 1 / (1 - tau_0), 0.0)
    # loss = loss * ratio
    loss = torch.where(M, loss, 0.0)
    return loss.sum() / M.sum()