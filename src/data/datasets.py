from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from .sources import (
    generate_burgers_data,
    generate_darcy_data,
    load_fd_bench_processed,
)


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


class FDBenchNextStepDataset(Dataset):
    """FD-Bench compatible next-step dataset returning channel-first tensors."""

    def __init__(
        self,
        data: torch.Tensor,
        grid: torch.Tensor,
        *,
        random_time: bool = True,
        append_grid: bool = True,
    ) -> None:
        if data.ndim < 4:
            raise ValueError(
                "FD-Bench tensor must be [batch, spatial..., time, channels]."
            )

        self.data = data
        self.grid = grid
        self.random_time = random_time
        self.append_grid = append_grid

        self.spatial_dim = data.ndim - 3
        self.base_input_channels = data.shape[-1]
        self.target_channels = data.shape[-1]
        self.input_channels = self.base_input_channels + (
            self.spatial_dim if append_grid else 0
        )

        if data.shape[-2] < 2:
            raise ValueError(
                "FD-Bench dataset needs at least two time steps for next-step forecasting."
            )

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        num_time_steps = self.data.shape[-2]
        if self.random_time:
            t0 = torch.randint(0, num_time_steps - 1, (1,)).item()
        else:
            t0 = 0

        x_t = self.data[idx, ..., t0, :]
        y_t1 = self.data[idx, ..., t0 + 1, :]

        # Convert [spatial..., channels] -> [channels, spatial...]
        permute_order = [x_t.ndim - 1, *range(x_t.ndim - 1)]
        x_t = x_t.permute(*permute_order)
        y_t1 = y_t1.permute(*permute_order)

        if self.append_grid:
            # Grid: [spatial..., dim] -> [dim, spatial...]
            grid = self.grid.permute(self.grid.ndim - 1, *range(self.grid.ndim - 1))
            x_t = torch.cat([x_t, grid], dim=0)

        return x_t.float(), y_t1.float()


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


def build_fd_bench_dataset(
    *,
    n_samples: int,
    subsample: int = 1,
    file_path: str,
    temporal_subsample: int = 1,
    batch_subsample: int = 1,
    include_nu_channel: bool = True,
    normalize: bool = False,
    shuffle: bool = True,
    seed: int = 42,
    random_time: bool = True,
    append_grid: bool = True,
    **kwargs: Any,
) -> FDBenchNextStepDataset:
    data, grid = load_fd_bench_processed(
        file_path=file_path,
        n_samples=n_samples,
        subsample=subsample,
        temporal_subsample=temporal_subsample,
        batch_subsample=batch_subsample,
        include_nu_channel=include_nu_channel,
        normalize=normalize,
        shuffle=shuffle,
        seed=seed,
    )

    return FDBenchNextStepDataset(
        data,
        grid,
        random_time=random_time,
        append_grid=append_grid,
    )
