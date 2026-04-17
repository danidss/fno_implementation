from __future__ import annotations

import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


def _subsample_spatial(data: np.ndarray, subsample: int) -> np.ndarray:
    if subsample <= 1:
        return data
    if data.ndim < 3:
        raise ValueError("Expected at least [batch, time, spatial...] for FD-Bench data")

    slices = [slice(None), slice(None)]
    slices.extend([slice(None, None, subsample)] * (data.ndim - 2))
    return data[tuple(slices)]


def _to_spatial_time_channel(data: np.ndarray) -> np.ndarray:
    # Input data shape is [B, T, X1, ..., Xd]. Output is [B, X1, ..., Xd, T, C].
    spatial_dims = tuple(range(2, data.ndim))
    return np.transpose(data, (0, *spatial_dims, 1))[..., None]


def _build_grid(file_handle: h5py.File, subsample: int) -> torch.Tensor:
    coordinate_keys = [
        key for key in ("x-coordinate", "y-coordinate", "z-coordinate") if key in file_handle
    ]
    if not coordinate_keys:
        raise ValueError(
            "FD-Bench file must contain coordinate datasets (e.g., x-coordinate, y-coordinate)."
        )

    coords = [
        torch.tensor(np.array(file_handle[key], dtype=np.float32), dtype=torch.float32)
        for key in coordinate_keys
    ]
    mesh = torch.meshgrid(*coords, indexing="ij")
    grid = torch.stack(mesh, dim=-1)

    if subsample > 1:
        slices = [slice(None, None, subsample)] * len(coords)
        slices.append(slice(None))
        grid = grid[tuple(slices)]

    return grid


def _load_fd_bench_array(
    *,
    file_path: str,
    subsample: int,
    temporal_subsample: int,
    batch_subsample: int,
    include_nu_channel: bool,
) -> tuple[np.ndarray, torch.Tensor]:
    with h5py.File(file_path, "r") as f:
        if "density" in f:
            variable_keys = [key for key in ("density", "pressure", "Vx", "Vy", "Vz") if key in f]
            if not variable_keys:
                raise ValueError("No FD-Bench CNS variables found in file")

            fields = []
            for key in variable_keys:
                arr = np.array(f[key], dtype=np.float32)
                arr = arr[::batch_subsample, ::temporal_subsample]
                arr = _subsample_spatial(arr, subsample)
                fields.append(_to_spatial_time_channel(arr))

            data = np.concatenate(fields, axis=-1)
            grid = _build_grid(f, subsample)

        elif "tensor" in f:
            tensor = np.array(f["tensor"], dtype=np.float32)
            tensor = tensor[::batch_subsample, ::temporal_subsample]
            tensor = _subsample_spatial(tensor, subsample)
            data = _to_spatial_time_channel(tensor)
            grid = _build_grid(f, subsample)

            if include_nu_channel and "nu" in f:
                nu = np.array(f["nu"], dtype=np.float32)
                # Common PDEBench Darcy shape: [B, X, Y].
                if nu.ndim == tensor.ndim - 1:
                    nu = nu[::batch_subsample]
                    nu_slices = [slice(None)]
                    nu_slices.extend([slice(None, None, subsample)] * (nu.ndim - 1))
                    nu = nu[tuple(nu_slices)]
                    # Broadcast static coefficient over time.
                    nu = np.expand_dims(nu, axis=-1)
                    nu = np.repeat(nu, data.shape[-2], axis=-1)
                # Time-dependent viscosity case: [B, T, X, ...]
                elif nu.ndim == tensor.ndim:
                    nu = nu[::batch_subsample, ::temporal_subsample]
                    nu = _subsample_spatial(nu, subsample)
                    nu = np.transpose(nu, (0, *range(2, nu.ndim), 1))
                else:
                    raise ValueError(
                        f"Unsupported nu shape {nu.shape}; expected [B, X, ...] or [B, T, X, ...]."
                    )

                nu = nu[..., None]
                data = np.concatenate([nu, data], axis=-1)

        else:
            raise ValueError(
                "Unsupported FD-Bench file format. Expected either CNS keys (density/pressure/Vx...) or 'tensor'."
            )

    return data, grid


def load_fd_bench_processed(
    *,
    file_path: str,
    n_samples: int,
    subsample: int = 1,
    temporal_subsample: int = 1,
    batch_subsample: int = 1,
    include_nu_channel: bool = True,
    normalize: bool = False,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"FD-Bench file not found: {file_path}. Download data first or set dataset_kwargs.file_path correctly."
        )

    data, grid = _load_fd_bench_array(
        file_path=file_path,
        subsample=subsample,
        temporal_subsample=temporal_subsample,
        batch_subsample=batch_subsample,
        include_nu_channel=include_nu_channel,
    )

    if shuffle:
        rng = np.random.default_rng(seed)
        indices = np.arange(data.shape[0])
        rng.shuffle(indices)
        data = data[indices]

    if n_samples > 0:
        data = data[: min(n_samples, data.shape[0])]

    if normalize:
        mean = np.mean(data, axis=tuple(range(data.ndim - 1)), keepdims=True)
        std = np.std(data, axis=tuple(range(data.ndim - 1)), keepdims=True)
        std = np.where(std == 0, 1.0, std)
        data = (data - mean) / std

    return torch.tensor(data, dtype=torch.float32), grid


def generate_fd_bench_raw(
    *,
    n_samples: int,
    file_path: str,
    subsample: int = 1,
    temporal_subsample: int = 1,
    batch_subsample: int = 1,
    include_nu_channel: bool = True,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    data, _ = load_fd_bench_processed(
        file_path=file_path,
        n_samples=n_samples,
        subsample=subsample,
        temporal_subsample=temporal_subsample,
        batch_subsample=batch_subsample,
        include_nu_channel=include_nu_channel,
        normalize=False,
        shuffle=True,
        seed=seed,
    )

    if data.shape[-2] < 2:
        raise ValueError("FD-Bench data requires at least two time steps for next-step pairs")

    a = data[..., 0, :].numpy()
    u = data[..., 1, :].numpy()
    return {"a": a, "u": u}


def download_file(url: str, output_path: str) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, output)
    return str(output)


def download_fd_bench_data(
    *,
    url: str,
    output_dir: str,
    file_name: str | None = None,
) -> str:
    parsed = urllib.parse.urlparse(url)
    inferred_name = Path(parsed.path).name or "fd_bench_data.h5"
    destination = Path(output_dir) / (file_name or inferred_name)
    return download_file(url, str(destination))


def download_fd_bench_from_hf(
    *,
    dataset_id: str,
    output_dir: str,
    split: str = "train",
) -> str:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "HuggingFace datasets is required for HF download. Install with: pip install datasets"
        ) from exc

    ds = load_dataset(dataset_id, split=split)
    destination = Path(output_dir) / f"{dataset_id.replace('/', '_')}_{split}"
    destination.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(destination))
    return str(destination)
