from __future__ import annotations

from typing import Any

import numpy as np
import torch


FD_BENCH_COLLECTION_URL = "https://huggingface.co/collections/RuoyanLi1/fd-bench"

# Metadata is fixed by preset and used as registry source-of-truth.
FD_BENCH_DATASET_PRESETS: dict[str, dict[str, Any]] = {
    "advection0": {
        "registry_name": "fd_bench_advection0",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-Advection0",
        "family": "advection",
        "spatial_dim": 2,
        "input_channels": 1,
        "out_channels": 1,
        "append_grid": False,
        "split": "test",
    },
    "advection1": {
        "registry_name": "fd_bench_advection1",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-Advection1",
        "family": "advection",
        "spatial_dim": 2,
        "input_channels": 1,
        "out_channels": 1,
        "append_grid": False,
        "split": "test",
    },
    "advection2": {
        "registry_name": "fd_bench_advection2",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-Advection2",
        "family": "advection",
        "spatial_dim": 2,
        "input_channels": 1,
        "out_channels": 1,
        "append_grid": False,
        "split": "test",
    },
    "advection3": {
        "registry_name": "fd_bench_advection3",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-Advection3",
        "family": "advection",
        "spatial_dim": 2,
        "input_channels": 1,
        "out_channels": 1,
        "append_grid": False,
        "split": "test",
    },
    "advection4": {
        "registry_name": "fd_bench_advection4",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-Advection4",
        "family": "advection",
        "spatial_dim": 2,
        "input_channels": 1,
        "out_channels": 1,
        "append_grid": False,
        "split": "test",
    },
    "burgers0": {
        "registry_name": "fd_bench_burgers0",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-Burgers0",
        "family": "burgers",
        "spatial_dim": 2,
        "input_channels": 2,
        "out_channels": 2,
        "append_grid": False,
        "split": "test",
    },
    "burgers1": {
        "registry_name": "fd_bench_burgers1",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-Burgers1",
        "family": "burgers",
        "spatial_dim": 2,
        "input_channels": 2,
        "out_channels": 2,
        "append_grid": False,
        "split": "test",
    },
    "burgers2": {
        "registry_name": "fd_bench_burgers2",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-Burgers2",
        "family": "burgers",
        "spatial_dim": 2,
        "input_channels": 2,
        "out_channels": 2,
        "append_grid": False,
        "split": "test",
    },
    "burgers3": {
        "registry_name": "fd_bench_burgers3",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-Burgers3",
        "family": "burgers",
        "spatial_dim": 2,
        "input_channels": 2,
        "out_channels": 2,
        "append_grid": False,
        "split": "test",
    },
    "burgers4": {
        "registry_name": "fd_bench_burgers4",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-Burgers4",
        "family": "burgers",
        "spatial_dim": 2,
        "input_channels": 2,
        "out_channels": 2,
        "append_grid": False,
        "split": "test",
    },
    "ns0": {
        "registry_name": "fd_bench_ns0",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-NS0",
        "family": "navier_stokes",
        "spatial_dim": 2,
        "input_channels": 1,
        "out_channels": 1,
        "append_grid": False,
        "split": "test",
    },
    "ns1": {
        "registry_name": "fd_bench_ns1",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-NS1",
        "family": "navier_stokes",
        "spatial_dim": 2,
        "input_channels": 1,
        "out_channels": 1,
        "append_grid": False,
        "split": "test",
    },
    "ns2": {
        "registry_name": "fd_bench_ns2",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-NS2",
        "family": "navier_stokes",
        "spatial_dim": 2,
        "input_channels": 1,
        "out_channels": 1,
        "append_grid": False,
        "split": "test",
    },
    "ns3": {
        "registry_name": "fd_bench_ns3",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-NS3",
        "family": "navier_stokes",
        "spatial_dim": 2,
        "input_channels": 1,
        "out_channels": 1,
        "append_grid": False,
        "split": "test",
    },
    "ns4": {
        "registry_name": "fd_bench_ns4",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-NS4",
        "family": "navier_stokes",
        "spatial_dim": 2,
        "input_channels": 1,
        "out_channels": 1,
        "append_grid": False,
        "split": "test",
    },
    "ldc": {
        "registry_name": "fd_bench_ldc",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-LDC",
        "family": "lagrangian",
        "spatial_dim": 1,
        "input_channels": 2,
        "out_channels": 2,
        "append_grid": False,
        "split": "test",
    },
    "rpf": {
        "registry_name": "fd_bench_rpf",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-RPF",
        "family": "lagrangian",
        "spatial_dim": 1,
        "input_channels": 2,
        "out_channels": 2,
        "append_grid": False,
        "split": "test",
    },
    "tgv": {
        "registry_name": "fd_bench_tgv",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-TGV",
        "family": "lagrangian",
        "spatial_dim": 1,
        "input_channels": 2,
        "out_channels": 2,
        "append_grid": False,
        "split": "test",
    },
}


def _load_hf_dataset(*, dataset_id: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "HuggingFace datasets is required for FD-Bench ingestion. Install with: pip install datasets"
        ) from exc

    return load_dataset(dataset_id, split=split)


def list_fd_bench_datasets() -> list[str]:
    return sorted(FD_BENCH_DATASET_PRESETS.keys())


def list_fd_bench_registry_names() -> list[str]:
    return sorted(
        preset["registry_name"] for preset in FD_BENCH_DATASET_PRESETS.values()
    )


def get_fd_bench_dataset_preset(fd_dataset: str) -> dict[str, Any]:
    normalized = fd_dataset.strip().lower()
    if normalized in FD_BENCH_DATASET_PRESETS:
        return dict(FD_BENCH_DATASET_PRESETS[normalized])

    for preset_name, preset in FD_BENCH_DATASET_PRESETS.items():
        if preset["registry_name"] == normalized:
            resolved = dict(preset)
            resolved["fd_dataset"] = preset_name
            return resolved

    available = ", ".join(list_fd_bench_datasets())
    raise ValueError(
        f"Unknown fd_bench dataset '{fd_dataset}'. Available presets: {available}"
    )


def resolve_fd_bench_config(
    *,
    fd_dataset: str,
    split: str | None = None,
) -> dict[str, Any]:
    preset = get_fd_bench_dataset_preset(fd_dataset)
    preset_name = preset.get("fd_dataset", fd_dataset.strip().lower())
    preset["fd_dataset"] = preset_name
    preset["split"] = split or preset.get("split", "test")
    return preset


def setup_fd_bench_dataset(
    *,
    fd_dataset: str,
    split: str | None = None,
    check_access: bool = False,
) -> dict[str, Any]:
    config = resolve_fd_bench_config(fd_dataset=fd_dataset, split=split)
    if check_access:
        _ = _load_hf_dataset(
            dataset_id=config["hf_dataset_id"],
            split=config["split"],
        )
    return config


def setup_fd_bench_datasets(*, check_access: bool = False) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for name in list_fd_bench_datasets():
        configs.append(
            setup_fd_bench_dataset(fd_dataset=name, check_access=check_access)
        )
    return configs


def _default_grid(*, spatial_shape: tuple[int, ...]) -> torch.Tensor:
    if len(spatial_shape) == 0:
        raise ValueError("spatial_shape cannot be empty")

    coords = [torch.linspace(0.0, 1.0, n, dtype=torch.float32) for n in spatial_shape]
    mesh = torch.meshgrid(*coords, indexing="ij")
    return torch.stack(mesh, dim=-1)


def _subsample_spatial_time_channel(
    sample: np.ndarray,
    *,
    subsample: int,
    temporal_subsample: int,
) -> np.ndarray:
    if sample.ndim < 3:
        raise ValueError(
            "Decoded FD-Bench sample must be [spatial..., time, channels]."
        )

    result = sample
    if subsample > 1:
        slices = [slice(None, None, subsample)] * (result.ndim - 2)
        slices.extend([slice(None), slice(None)])
        result = result[tuple(slices)]

    if temporal_subsample > 1:
        result = result[..., ::temporal_subsample, :]

    return result


def _decode_array_to_spatial_time_channel(
    *,
    array: np.ndarray,
    spatial_dim: int,
    out_channels: int,
) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 0:
        raise ValueError("Scalar values are not supported in FD-Bench rows")

    if arr.ndim == spatial_dim:
        # [spatial...] -> single time step, scalar channel.
        return arr[..., None, None]

    if arr.ndim == spatial_dim + 2:
        # Expected common case: [time, spatial..., channels]
        if arr.shape[-1] == out_channels:
            spatial_axes = tuple(range(1, 1 + spatial_dim))
            return np.transpose(arr, (*spatial_axes, 0, arr.ndim - 1))

        # Scalar channel implicit: [time, spatial...]
        spatial_axes = tuple(range(1, 1 + spatial_dim))
        return np.transpose(arr, (*spatial_axes, 0))[..., None]

    if arr.ndim == spatial_dim + 1:
        # [spatial..., channels] -> single time step
        if arr.shape[-1] == out_channels:
            return np.expand_dims(arr, axis=-2)

        # [time, spatial...] -> scalar channel
        if spatial_dim == 1:
            return np.transpose(arr, (1, 0))[..., None]
        if spatial_dim == 2:
            return np.transpose(arr, (1, 2, 0))[..., None]

    if arr.ndim == spatial_dim + 3:
        # Alternate layout: [channels, time, spatial...]
        if arr.shape[0] == out_channels:
            spatial_axes = tuple(range(2, 2 + spatial_dim))
            return np.transpose(arr, (*spatial_axes, 1, 0))

    raise ValueError(
        f"Unsupported FD-Bench sample tensor shape {arr.shape} for spatial_dim={spatial_dim}"
    )


def _decode_row_to_spatial_time_channel(
    *,
    row: dict[str, Any],
    spatial_dim: int,
    out_channels: int,
) -> np.ndarray:
    if "data" in row:
        values = [row["data"]]
    else:
        values = [row[k] for k in sorted(row.keys())]

    decoded = [
        _decode_array_to_spatial_time_channel(
            array=np.asarray(value),
            spatial_dim=spatial_dim,
            out_channels=out_channels,
        )
        for value in values
    ]

    if len(decoded) == 1:
        return decoded[0]

    # If each column contributes one time slice, concatenate across time.
    if all(part.shape[-2] == 1 and part.shape[-1] == out_channels for part in decoded):
        return np.concatenate(decoded, axis=-2)

    # Otherwise treat columns as channels.
    return np.concatenate(decoded, axis=-1)


def load_fd_bench_processed(
    *,
    fd_dataset: str,
    n_samples: int,
    split: str | None = None,
    subsample: int = 1,
    temporal_subsample: int = 1,
    batch_subsample: int = 1,
    normalize: bool = False,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    config = resolve_fd_bench_config(fd_dataset=fd_dataset, split=split)

    dataset = _load_hf_dataset(
        dataset_id=config["hf_dataset_id"],
        split=config["split"],
    )

    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    total_rows = len(dataset)
    rows = dataset

    if batch_subsample > 1:
        kept_indices = list(range(0, total_rows, batch_subsample))
        rows = rows.select(kept_indices)

    if n_samples > 0:
        rows = rows.select(range(min(n_samples, len(rows))))

    samples: list[np.ndarray] = []
    for row in rows:
        sample = _decode_row_to_spatial_time_channel(
            row=row,
            spatial_dim=int(config["spatial_dim"]),
            out_channels=int(config["out_channels"]),
        )
        sample = _subsample_spatial_time_channel(
            sample,
            subsample=subsample,
            temporal_subsample=temporal_subsample,
        )
        samples.append(sample)

    if not samples:
        raise ValueError("FD-Bench split produced no samples")

    data = np.stack(samples, axis=0).astype(np.float32)

    if normalize:
        mean = np.mean(data, axis=tuple(range(data.ndim - 1)), keepdims=True)
        std = np.std(data, axis=tuple(range(data.ndim - 1)), keepdims=True)
        std = np.where(std == 0, 1.0, std)
        data = (data - mean) / std

    grid = _default_grid(spatial_shape=tuple(data.shape[1:-2]))
    return torch.tensor(data, dtype=torch.float32), grid


def generate_fd_bench_raw(*, n_samples: int, **kwargs: Any) -> dict[str, np.ndarray]:
    raise ValueError(
        "FD-Bench raw generation is disabled in HF-only mode. "
        "Use synthetic datasets for generate_data.py, or build FD-Bench via registry builders."
    )


def download_fd_bench_data(
    *, url: str, output_dir: str, file_name: str | None = None
) -> str:
    raise ValueError(
        "Direct FD-Bench downloads are disabled. "
        f"Use HuggingFace collection instead: {FD_BENCH_COLLECTION_URL}"
    )


def download_fd_bench_from_hf(
    *, dataset_id: str, output_dir: str, split: str = "test"
) -> str:
    raise ValueError(
        "Local export/download for FD-Bench is disabled. "
        "Load directly from HuggingFace at runtime via FD-Bench presets."
    )
