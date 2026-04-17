from .data_visualization import (
	visualize_burgers_comparison,
	visualize_burgers_samples,
	visualize_burgers_statistics,
	visualize_darcy_samples,
	visualize_darcy_statistics,
	visualize_navier_stokes_difference,
	visualize_navier_stokes_samples,
	visualize_navier_stokes_statistics,
)
from .datasets import build_burgers_dataset, build_darcy_dataset, build_fd_bench_dataset
from .providers_builtin import register_builtin_datasets
from .registry import (
    build_train_test_split,
    build_train_test_split_from_args,
    build_dataset,
    generate_raw_data,
    get_dataset_spec,
    get_registration,
    list_datasets,
    register_dataset,
)


register_builtin_datasets()


__all__ = [
    "register_builtin_datasets",
    "register_dataset",
    "list_datasets",
    "get_registration",
    "get_dataset_spec",
    "build_dataset",
    "build_train_test_split",
    "build_train_test_split_from_args",
    "generate_raw_data",
    "build_burgers_dataset",
    "build_darcy_dataset",
    "build_fd_bench_dataset",
    "visualize_burgers_samples",
    "visualize_burgers_comparison",
    "visualize_burgers_statistics",
    "visualize_darcy_samples",
    "visualize_darcy_statistics",
    "visualize_navier_stokes_samples",
    "visualize_navier_stokes_difference",
    "visualize_navier_stokes_statistics",
]
