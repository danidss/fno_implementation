from __future__ import annotations

from typing import Any
import warnings

import torch
from torch.utils.data import Dataset
import numpy as np

from .sources import (
    generate_burgers_data,
    generate_darcy_data,
    load_fd_bench_rows,
    resolve_fd_bench_config,
)


class TensorOperatorDataset(Dataset):
    """Simple dataset wrapper over precomputed tensor pairs (x, y)."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if x.shape[0] != y.shape[0]:
            raise ValueError("Input and target tensors must have the same batch size")
        if x.ndim < 2 or y.ndim < 2:
            raise ValueError(
                "Expected channel-first tensors with at least one spatial axis"
            )
        self.x = x
        self.y = y
        self.input_channels = int(x.shape[1])
        self.out_channels = int(y.shape[1])
        self.spatial_dim = int(x.ndim - 2)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class FDBenchDataset(Dataset):
    """FD-Bench next-step dataset backed by HF rows preprocessed via map."""

    def __init__(
        self,
        rows: Any,
        *,
        spatial_dim: int,
        target_channels: int,
        random_time: bool = True,
        append_grid: bool = True,
    ) -> None:
        self.rows = rows
        self.spatial_dim = int(spatial_dim)
        self.random_time = bool(random_time)
        self.append_grid = bool(append_grid)
        self.configured_target_channels = int(target_channels)
        self.target_channels = self.configured_target_channels

        if len(self.rows) == 0:
            raise ValueError("FD-Bench split produced no samples")

        first_row = self._get_row(0)
        first_sample = self._sample_from_row(first_row)
        if first_sample.shape[-2] < 2:
            raise ValueError(
                "FD-Bench dataset needs at least two time steps for next-step forecasting."
            )

        decoded_channels = int(first_sample.shape[-1])
        if decoded_channels != self.configured_target_channels:
            warnings.warn(
                "FD-Bench preset metadata channel count does not match decoded sample; "
                f"using decoded channels ({decoded_channels}) instead of configured "
                f"channels ({self.configured_target_channels})."
            )
            self.target_channels = decoded_channels

        self.out_channels = self.target_channels
        self.input_channels = self.target_channels + (
            self.spatial_dim if self.append_grid else 0
        )

        self.grid = _default_grid(spatial_shape=tuple(first_sample.shape[:-2]))

    def _sample_from_row(self, row: dict[str, Any]) -> np.ndarray:
        if "fd_sample" not in row:
            raise ValueError(
                "FD-Bench row is missing 'fd_sample'. "
                "Expected rows preprocessed by load_fd_bench_rows()."
            )

        sample = np.asarray(row["fd_sample"], dtype=np.float32)
        if sample.ndim < 3:
            raise ValueError("FD-Bench sample must be [spatial..., time, channels].")
        return sample

    def __len__(self) -> int:
        return len(self.rows)

    def _get_row(self, idx: int) -> dict[str, Any]:
        if hasattr(self.rows, "__getitem__"):
            return self.rows[idx]

        for i, row in enumerate(self.rows):
            if i == idx:
                return row
        raise IndexError(idx)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self._sample_from_row(self._get_row(idx))

        num_time_steps = sample.shape[-2]
        if self.random_time:
            t0 = torch.randint(0, num_time_steps - 1, (1,)).item()
        else:
            t0 = 0

        x_t = torch.as_tensor(sample[..., t0, :], dtype=torch.float32)
        y_t1 = torch.as_tensor(sample[..., t0 + 1, :], dtype=torch.float32)

        permute_order = [x_t.ndim - 1, *range(x_t.ndim - 1)]
        x_t = x_t.permute(*permute_order)
        y_t1 = y_t1.permute(*permute_order)

        if self.append_grid:
            grid = self.grid.permute(self.grid.ndim - 1, *range(self.grid.ndim - 1))
            x_t = torch.cat([x_t, grid], dim=0)

        return x_t, y_t1


def _default_grid(*, spatial_shape: tuple[int, ...]) -> torch.Tensor:
    if len(spatial_shape) == 0:
        raise ValueError("spatial_shape cannot be empty")

    coords = [torch.linspace(0.0, 1.0, n, dtype=torch.float32) for n in spatial_shape]
    mesh = torch.meshgrid(*coords, indexing="ij")
    return torch.stack(mesh, dim=-1)


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
    fd_dataset: str | None = None,
    split: str | None = None,
    temporal_subsample: int = 1,
    batch_subsample: int = 1,
    normalize: bool = False,
    shuffle: bool = True,
    seed: int = 42,
    hf_cache_dir: str | None = None,
    random_time: bool = True,
    append_grid: bool | None = None,
    **kwargs: Any,
) -> FDBenchDataset:
    if not fd_dataset:
        raise ValueError("FD-Bench builder requires an fd_dataset preset name")

    preset = resolve_fd_bench_config(fd_dataset=fd_dataset, split=split)
    if append_grid is None:
        append_grid = bool(preset.get("append_grid", False))

    rows, config = load_fd_bench_rows(
        fd_dataset=str(preset["fd_dataset"]),
        n_samples=n_samples,
        split=str(preset["split"]),
        subsample=subsample,
        temporal_subsample=temporal_subsample,
        batch_subsample=batch_subsample,
        normalize=normalize,
        shuffle=shuffle,
        seed=seed,
        hf_cache_dir=hf_cache_dir,
    )

    return FDBenchDataset(
        rows,
        spatial_dim=int(config["spatial_dim"]),
        target_channels=int(config["out_channels"]),
        random_time=random_time,
        append_grid=append_grid,
    )
