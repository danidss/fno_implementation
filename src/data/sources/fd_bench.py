from __future__ import annotations

import os
from typing import Any

import numpy as np


FD_BENCH_COLLECTION_URL = "https://huggingface.co/collections/RuoyanLi1/fd-bench"
DEFAULT_FD_BENCH_SPLIT = "test"

# Metadata is fixed by preset and used as registry source-of-truth.
FD_BENCH_DATASET_PRESETS: dict[str, dict[str, Any]] = {
    "advection0": {
        "registry_name": "fd_bench_advection0",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-Advection0",
        "description": "FD-Bench Advection dataset (index 0) on a 2D spatial domain.",
        "family": "advection",
        "spatial_dim": 2,
        "input_channels": 1,
        "out_channels": 1,
        "append_grid": False,
    },
    "advection1": {
        "registry_name": "fd_bench_advection1",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-Advection1",
        "description": "FD-Bench Advection dataset (index 1) on a 2D spatial domain.",
        "family": "advection",
        "spatial_dim": 2,
        "input_channels": 1,
        "out_channels": 1,
        "append_grid": False,
    },
    "advection2": {
        "registry_name": "fd_bench_advection2",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-Advection2",
        "description": "FD-Bench Advection dataset (index 2) on a 2D spatial domain.",
        "family": "advection",
        "spatial_dim": 2,
        "input_channels": 1,
        "out_channels": 1,
        "append_grid": False,
    },
    "advection3": {
        "registry_name": "fd_bench_advection3",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-Advection3",
        "description": "FD-Bench Advection dataset (index 3) on a 2D spatial domain.",
        "family": "advection",
        "spatial_dim": 2,
        "input_channels": 1,
        "out_channels": 1,
        "append_grid": False,
    },
    "advection4": {
        "registry_name": "fd_bench_advection4",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-Advection4",
        "description": "FD-Bench Advection dataset (index 4) on a 2D spatial domain.",
        "family": "advection",
        "spatial_dim": 2,
        "input_channels": 1,
        "out_channels": 1,
        "append_grid": False,
    },
    "burgers0": {
        "registry_name": "fd_bench_burgers0",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-Burgers0",
        "description": "FD-Bench Burgers dataset (index 0) on a 2D spatial domain.",
        "family": "burgers",
        "spatial_dim": 2,
        "input_channels": 2,
        "out_channels": 2,
        "append_grid": False,
    },
    "burgers1": {
        "registry_name": "fd_bench_burgers1",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-Burgers1",
        "description": "FD-Bench Burgers dataset (index 1) on a 2D spatial domain.",
        "family": "burgers",
        "spatial_dim": 2,
        "input_channels": 2,
        "out_channels": 2,
        "append_grid": False,
    },
    "burgers2": {
        "registry_name": "fd_bench_burgers2",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-Burgers2",
        "description": "FD-Bench Burgers dataset (index 2) on a 2D spatial domain.",
        "family": "burgers",
        "spatial_dim": 2,
        "input_channels": 2,
        "out_channels": 2,
        "append_grid": False,
    },
    "burgers3": {
        "registry_name": "fd_bench_burgers3",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-Burgers3",
        "description": "FD-Bench Burgers dataset (index 3) on a 2D spatial domain.",
        "family": "burgers",
        "spatial_dim": 2,
        "input_channels": 2,
        "out_channels": 2,
        "append_grid": False,
    },
    "burgers4": {
        "registry_name": "fd_bench_burgers4",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-Burgers4",
        "description": "FD-Bench Burgers dataset (index 4) on a 2D spatial domain.",
        "family": "burgers",
        "spatial_dim": 2,
        "input_channels": 2,
        "out_channels": 2,
        "append_grid": False,
    },
    "ns0": {
        "registry_name": "fd_bench_ns0",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-NS0",
        "description": "FD-Bench Navier-Stokes dataset (index 0) on a 2D spatial domain.",
        "family": "navier_stokes",
        "spatial_dim": 2,
        "input_channels": 1,
        "out_channels": 1,
        "append_grid": False,
    },
    "ns1": {
        "registry_name": "fd_bench_ns1",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-NS1",
        "description": "FD-Bench Navier-Stokes dataset (index 1) on a 2D spatial domain.",
        "family": "navier_stokes",
        "spatial_dim": 2,
        "input_channels": 1,
        "out_channels": 1,
        "append_grid": False,
    },
    "ns2": {
        "registry_name": "fd_bench_ns2",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-NS2",
        "description": "FD-Bench Navier-Stokes dataset (index 2) on a 2D spatial domain.",
        "family": "navier_stokes",
        "spatial_dim": 2,
        "input_channels": 1,
        "out_channels": 1,
        "append_grid": False,
    },
    "ns3": {
        "registry_name": "fd_bench_ns3",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-NS3",
        "description": "FD-Bench Navier-Stokes dataset (index 3) on a 2D spatial domain.",
        "family": "navier_stokes",
        "spatial_dim": 2,
        "input_channels": 1,
        "out_channels": 1,
        "append_grid": False,
    },
    "ns4": {
        "registry_name": "fd_bench_ns4",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-NS4",
        "description": "FD-Bench Navier-Stokes dataset (index 4) on a 2D spatial domain.",
        "family": "navier_stokes",
        "spatial_dim": 2,
        "input_channels": 1,
        "out_channels": 1,
        "append_grid": False,
    },
    "ldc": {
        "registry_name": "fd_bench_ldc",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-LDC",
        "description": "2D lid-driven cavity flow governed by compressible Navier-Stokes.",
        "family": "compressible_navier_stokes",
        "spatial_dim": 2,
        "input_channels": 2,
        "out_channels": 2,
        "append_grid": False,
    },
    "rpf": {
        "registry_name": "fd_bench_rpf",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-RPF",
        "description": "2D reverse Poiseuille flow governed by compressible Navier-Stokes.",
        "family": "compressible_navier_stokes",
        "spatial_dim": 2,
        "input_channels": 2,
        "out_channels": 2,
        "append_grid": False,
    },
    "tgv": {
        "registry_name": "fd_bench_tgv",
        "hf_dataset_id": "RuoyanLi1/FD-Bench-TGV",
        "description": "2D Taylor-Green vortex governed by compressible Navier-Stokes.",
        "family": "compressible_navier_stokes",
        "spatial_dim": 2,
        "input_channels": 2,
        "out_channels": 2,
        "append_grid": False,
    },
}


def _load_hf_dataset(
    *,
    dataset_id: str,
    split: str,
    cache_dir: str | None = None,
):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "HuggingFace datasets is required for FD-Bench ingestion. Install with: pip install datasets"
        ) from exc

    try:
        return load_dataset(dataset_id, split=split, cache_dir=cache_dir)
    except Exception as exc:
        raise ValueError(
            f"Unable to load FD-Bench dataset '{dataset_id}' split '{split}' from HuggingFace. "
            "Verify the split name and network access."
        ) from exc


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
    preset["split"] = split or DEFAULT_FD_BENCH_SPLIT
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


def _decode_row_for_hf_map(
    row: dict[str, Any],
    *,
    spatial_dim: int,
    out_channels: int,
    subsample: int,
    temporal_subsample: int,
) -> dict[str, np.ndarray]:
    sample = decode_fd_bench_row(
        row=row,
        spatial_dim=spatial_dim,
        out_channels=out_channels,
        subsample=subsample,
        temporal_subsample=temporal_subsample,
    )
    return {"fd_sample": sample.astype(np.float32, copy=False)}


def _compute_fd_sample_normalization_stats(rows: Any) -> tuple[np.ndarray, np.ndarray]:
    channel_sum: np.ndarray | None = None
    channel_sq_sum: np.ndarray | None = None
    total_count = 0

    for row in rows:
        sample = np.asarray(row["fd_sample"], dtype=np.float32)
        flat = sample.reshape(-1, sample.shape[-1])
        summed = flat.sum(axis=0)
        sq_summed = np.square(flat).sum(axis=0)

        if channel_sum is None:
            channel_sum = summed
            channel_sq_sum = sq_summed
        else:
            channel_sum += summed
            channel_sq_sum += sq_summed
        total_count += flat.shape[0]

    if channel_sum is None or channel_sq_sum is None or total_count == 0:
        raise ValueError("Unable to compute FD-Bench normalization statistics")

    mean = channel_sum / float(total_count)
    variance = channel_sq_sum / float(total_count) - np.square(mean)
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)
    std = np.where(std == 0, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _normalize_fd_sample_row(
    row: dict[str, Any], *, mean: np.ndarray, std: np.ndarray
) -> dict[str, np.ndarray]:
    sample = np.asarray(row["fd_sample"], dtype=np.float32)
    normalized = (sample - mean) / std
    return {"fd_sample": normalized.astype(np.float32, copy=False)}


def load_fd_bench_rows(
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
    hf_cache_dir: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a lazily-accessed HuggingFace subset without materializing all samples.

    Returns:
        rows: HuggingFace dataset object supporting __len__/__getitem__
        config: resolved FD-Bench preset metadata
    """

    resolved_cache_dir = hf_cache_dir or os.environ.get("FNO_HF_CACHE_DIR")
    config = resolve_fd_bench_config(fd_dataset=fd_dataset, split=split)
    rows = _load_hf_dataset(
        dataset_id=config["hf_dataset_id"],
        split=config["split"],
        cache_dir=resolved_cache_dir,
    )

    if shuffle:
        rows = rows.shuffle(seed=seed)

    total_rows = len(rows)
    if batch_subsample > 1:
        kept_indices = list(range(0, total_rows, batch_subsample))
        rows = rows.select(kept_indices)

    if n_samples > 0:
        rows = rows.select(range(min(n_samples, len(rows))))

    decode_kwargs = {
        "spatial_dim": int(config["spatial_dim"]),
        "out_channels": int(config["out_channels"]),
        "subsample": int(max(1, subsample)),
        "temporal_subsample": int(max(1, temporal_subsample)),
    }
    remove_columns = list(rows.column_names) if hasattr(rows, "column_names") else None
    rows = rows.map(
        lambda row: _decode_row_for_hf_map(row, **decode_kwargs),
        batched=False,
        remove_columns=remove_columns,
    )

    if normalize:
        mean, std = _compute_fd_sample_normalization_stats(rows)
        rows = rows.map(
            lambda row: _normalize_fd_sample_row(row, mean=mean, std=std),
            batched=False,
        )

    if hasattr(rows, "with_format"):
        rows = rows.with_format(
            type="numpy",
            columns=["fd_sample"],
            output_all_columns=False,
        )

    return rows, config


def decode_fd_bench_row(
    *,
    row: dict[str, Any],
    spatial_dim: int,
    out_channels: int,
    subsample: int = 1,
    temporal_subsample: int = 1,
) -> np.ndarray:
    """Decode one row into [spatial..., time, channels] with optional subsampling."""

    sample = _decode_row_to_spatial_time_channel(
        row=row,
        spatial_dim=spatial_dim,
        out_channels=out_channels,
    )
    return _subsample_spatial_time_channel(
        sample,
        subsample=subsample,
        temporal_subsample=temporal_subsample,
    )


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
