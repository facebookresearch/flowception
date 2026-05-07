"""Data classes for the training engine."""

from collections.abc import Mapping
from typing import Any, NamedTuple

import torch

Condition = Any


class Datapoint(NamedTuple):
    pixel_values: torch.Tensor
    condition: Condition


class LossTuple(NamedTuple):
    loss: torch.Tensor
    pred_eps: torch.Tensor
    vlb: Mapping[str, torch.Tensor]


class TrainTuple(NamedTuple):
    loss: float
    latents: torch.Tensor
    loss_dict: dict
    time_dict: dict
    vlb: Mapping[str, torch.Tensor]
    cond_t: torch.Tensor | None
