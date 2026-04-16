from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from .sources import generate_burgers_data, generate_darcy_data


class TensorOperatorDataset(Dataset):
    """Simple dataset wrapper over precomputed tensor pairs (x, y)."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if x.shape[0] != y.shape[0]:
            raise ValueError("Input and target tensors must have the same batch size")
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def _with_grid_1d(coefficients: torch.Tensor) -> torch.Tensor:
    n_x = coefficients.shape[-1]
    grid = torch.linspace(0, 1, n_x, dtype=coefficients.dtype, device=coefficients.device)
    grid = grid.unsqueeze(0).expand(coefficients.shape[0], n_x)
    return torch.stack([coefficients, grid], dim=1)


def _with_grid_2d(coefficients: torch.Tensor) -> torch.Tensor:
    n_samples, n_x, n_y = coefficients.shape
    grid_x, grid_y = torch.meshgrid(
        torch.linspace(0, 1, n_x, dtype=coefficients.dtype, device=coefficients.device),
        torch.linspace(0, 1, n_y, dtype=coefficients.dtype, device=coefficients.device),
        indexing="ij",
    )
    grid_x = grid_x.unsqueeze(0).expand(n_samples, n_x, n_y)
    grid_y = grid_y.unsqueeze(0).expand(n_samples, n_x, n_y)
    return torch.stack([coefficients, grid_x, grid_y], dim=1)


def build_burgers_dataset(
    *,
    n_samples: int,
    subsample: int = 1,
    **kwargs: Any,
) -> TensorOperatorDataset:
    data = generate_burgers_data(n_samples=n_samples, **kwargs)

    a = torch.tensor(data["a"], dtype=torch.float32)
    u = torch.tensor(data["u"], dtype=torch.float32)

    if subsample > 1:
        a = a[:, ::subsample]
        u = u[:, ::subsample]

    x = _with_grid_1d(a)
    y = u.unsqueeze(1)
    return TensorOperatorDataset(x, y)


def build_darcy_dataset(
    *,
    n_samples: int,
    subsample: int = 1,
    **kwargs: Any,
) -> TensorOperatorDataset:
    data = generate_darcy_data(n_samples=n_samples, **kwargs)

    a = torch.tensor(data["a"], dtype=torch.float32)
    u = torch.tensor(data["u"], dtype=torch.float32)

    if subsample > 1:
        a = a[:, ::subsample, ::subsample]
        u = u[:, ::subsample, ::subsample]

    x = _with_grid_2d(a)
    y = u.unsqueeze(1)
    return TensorOperatorDataset(x, y)
