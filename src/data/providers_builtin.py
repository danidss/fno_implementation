import warnings


from .datasets import build_burgers_dataset, build_darcy_dataset, build_fd_bench_dataset
from .registry import register_dataset
from .sources import (
    generate_fd_bench_raw,
    generate_burgers_data,
    generate_darcy_data,
    generate_navier_stokes_data,
)


_BOOTSTRAPPED = False


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
    register_dataset(
        name="fd_bench",
        input_channels=4,
        out_channels=2,
        dim=2,
        build_dataset=build_fd_bench_dataset,
        generate_raw=generate_fd_bench_raw,
        overwrite=False,
    )

    _BOOTSTRAPPED = True
