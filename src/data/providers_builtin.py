from .datasets import build_burgers_dataset, build_darcy_dataset
from .registry import register_dataset
from .sources import (
    generate_burgers_data,
    generate_darcy_data,
    generate_navier_stokes_data,
)


_BOOTSTRAPPED = False


def register_builtin_datasets() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    register_dataset(
        name="burgers",
        input_channels=2,
        spatial_dim=1,
        build_dataset=build_burgers_dataset,
        generate_raw=generate_burgers_data,
        overwrite=False,
    )
    register_dataset(
        name="darcy",
        input_channels=3,
        spatial_dim=2,
        build_dataset=build_darcy_dataset,
        generate_raw=generate_darcy_data,
        overwrite=False,
    )
    register_dataset(
        name="navier_stokes",
        input_channels=None,
        spatial_dim=2,
        build_dataset=None,
        generate_raw=generate_navier_stokes_data,
        overwrite=False,
    )

    _BOOTSTRAPPED = True
