from __future__ import annotations

import torch
from torch import nn


def create_random_direction(
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    """Create a random direction matching the model's parameter structure."""

    direction = {}

    for name, param in model.named_parameters():
        direction[name] = torch.randn_like(param)

    return direction

def filter_normalize_direction(
    model: nn.Module,
    direction: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:

    normalized = {}

    for name, param in model.named_parameters():
        d = direction[name]

        if param.ndim == 4:
            weight_norm = torch.linalg.vector_norm(
                param,
                dim=(1, 2, 3),
                keepdim=True,
            )

            direction_norm = torch.linalg.vector_norm(
                d,
                dim=(1, 2, 3),
                keepdim=True,
            )

            normalized[name] = (
                d / (direction_norm + 1e-12)
            ) * weight_norm

        elif param.ndim == 2:
            weight_norm = torch.linalg.vector_norm(
                param,
                dim=1,
                keepdim=True,
            )

            direction_norm = torch.linalg.vector_norm(
                d,
                dim=1,
                keepdim=True,
            )

            normalized[name] = (
                d / (direction_norm + 1e-12)
            ) * weight_norm

        else:
            normalized[name] = torch.zeros_like(d)

    return normalized