from __future__ import annotations

import torch
from torch import nn


def get_parameter_state(
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
    }


def set_parameter_state(
    model: nn.Module,
    state: dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(state[name])


def apply_direction(
    model: nn.Module,
    base_state: dict[str, torch.Tensor],
    direction: dict[str, torch.Tensor],
    alpha: float,
) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(
                base_state[name] + alpha * direction[name]
            )


def apply_two_directions(
    model: nn.Module,
    base_state: dict[str, torch.Tensor],
    direction_x: dict[str, torch.Tensor],
    direction_y: dict[str, torch.Tensor],
    alpha: float,
    beta: float,
) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(
                base_state[name]
                + alpha * direction_x[name]
                + beta * direction_y[name]
            )