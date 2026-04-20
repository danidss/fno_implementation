import warnings
from typing import Any, Callable


from .datasets import build_burgers_dataset, build_darcy_dataset, build_fd_bench_dataset
from .registry import register_dataset
from .sources import (
    get_fd_bench_dataset_preset,
    list_fd_bench_datasets,
    generate_burgers_data,
    generate_darcy_data,
    generate_navier_stokes_data,
)


_BOOTSTRAPPED = False


def _build_fd_bench_builder(fd_dataset: str) -> Callable[..., Any]:
    def _builder(*, n_samples: int, subsample: int = 1, **kwargs: Any):
        return build_fd_bench_dataset(
            n_samples=n_samples,
            subsample=subsample,
            fd_dataset=fd_dataset,
            **kwargs,
        )

    return _builder


def register_builtin_datasets() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        warnings.warn("Builtin datasets have already been registered.")
        return

    register_dataset(
        name="burgers",
        input_channels=2,
        out_channels=1,
        dim=1,
        build_dataset=build_burgers_dataset,
        generate_raw=generate_burgers_data,
        overwrite=False,
    )
    register_dataset(
        name="darcy",
        input_channels=3,
        out_channels=1,
        dim=2,
        build_dataset=build_darcy_dataset,
        generate_raw=generate_darcy_data,
        overwrite=False,
    )
    register_dataset(
        name="navier_stokes",
        input_channels=3,
        out_channels=1,
        dim=2,
        build_dataset=None,
        generate_raw=generate_navier_stokes_data,
        overwrite=False,
    )
    for fd_name in list_fd_bench_datasets():
        preset = get_fd_bench_dataset_preset(fd_name)
        register_dataset(
            name=str(preset["registry_name"]),
            input_channels=int(preset["input_channels"]),
            out_channels=int(preset["out_channels"]),
            dim=int(preset["spatial_dim"]),
            build_dataset=_build_fd_bench_builder(fd_name),
            generate_raw=None,
            overwrite=False,
        )

    _BOOTSTRAPPED = True
