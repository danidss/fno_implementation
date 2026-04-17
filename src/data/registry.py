from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch.utils.data import Dataset


RawGenerator = Callable[..., dict[str, Any]]
DatasetBuilder = Callable[..., Dataset]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    input_channels: int
    out_channels: int
    dim: int


@dataclass(frozen=True)
class DatasetRegistration:
    spec: DatasetSpec
    build_dataset: DatasetBuilder | None = None
    generate_raw: RawGenerator | None = None


_REGISTRY: dict[str, DatasetRegistration] = {}


def register_dataset(
    *,
    name: str,
    input_channels: int,
    out_channels: int,
    dim: int,
    build_dataset: DatasetBuilder | None = None,
    generate_raw: RawGenerator | None = None,
    overwrite: bool = False,
) -> None:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Dataset name cannot be empty")

    if normalized in _REGISTRY and not overwrite:
        raise ValueError(
            f"Dataset '{normalized}' is already registered. Set overwrite=True to replace it."
        )

    _REGISTRY[normalized] = DatasetRegistration(
        spec=DatasetSpec(
            name=normalized,
            input_channels=input_channels,
            out_channels=out_channels,
            dim=dim,
        ),
        build_dataset=build_dataset,
        generate_raw=generate_raw,
    )


def get_registration(name: str) -> DatasetRegistration:
    normalized = name.strip().lower()
    if normalized not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown dataset '{normalized}'. Available datasets: {available or 'none'}"
        )
    return _REGISTRY[normalized]


def list_datasets(
    *,
    require_builder: bool = False,
    require_raw_generator: bool = False,
) -> list[str]:
    names: list[str] = []
    for name, registration in sorted(_REGISTRY.items()):
        if require_builder and registration.build_dataset is None:
            continue
        if require_raw_generator and registration.generate_raw is None:
            continue
        names.append(name)
    return names


def get_dataset_spec(name: str) -> DatasetSpec:
    return get_registration(name).spec


def build_dataset(name: str, *, n_samples: int, subsample: int = 1, **kwargs: Any) -> Dataset:
    registration = get_registration(name)
    if registration.build_dataset is None:
        raise ValueError(
            f"Dataset '{name}' does not define a PyTorch dataset builder."
        )
    return registration.build_dataset(
        n_samples=n_samples,
        subsample=subsample,
        **kwargs,
    )


def generate_raw_data(name: str, *, n_samples: int, **kwargs: Any) -> dict[str, Any]:
    registration = get_registration(name)
    if registration.generate_raw is None:
        raise ValueError(f"Dataset '{name}' does not define a raw data generator.")
    return registration.generate_raw(n_samples=n_samples, **kwargs)


def _to_dict(maybe_mapping: Any) -> dict[str, Any]:
    if maybe_mapping is None:
        return {}
    if isinstance(maybe_mapping, dict):
        return dict(maybe_mapping)
    if hasattr(maybe_mapping, "items"):
        return dict(maybe_mapping.items())
    raise TypeError("dataset_kwargs must be a mapping")


def build_train_test_split(
    name: str,
    *,
    n_samples: int,
    train_split: float,
    subsample: int = 1,
    dataset_kwargs: dict[str, Any] | None = None,
    split_seed: int = 42,
) -> tuple[Dataset, Dataset, DatasetSpec]:
    dataset = build_dataset(
        name,
        n_samples=n_samples,
        subsample=subsample,
        **(dataset_kwargs or {}),
    )

    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(split_seed),
    )

    spec = get_dataset_spec(name)
    return train_dataset, test_dataset, spec


def build_train_test_split_from_args(
    args: Any,
    *,
    n_samples: int,
    split_seed: int = 42,
) -> tuple[Dataset, Dataset, DatasetSpec]:
    dataset_name = args.dataset
    subsample = getattr(args, "subsample", 1)
    train_split = getattr(args, "train_split", 0.8)
    dataset_kwargs = _to_dict(getattr(args, "dataset_kwargs", None))

    return build_train_test_split(
        dataset_name,
        n_samples=n_samples,
        train_split=train_split,
        subsample=subsample,
        dataset_kwargs=dataset_kwargs,
        split_seed=split_seed,
    )
