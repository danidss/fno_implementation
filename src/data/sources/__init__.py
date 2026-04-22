from .cache import cache_incremental_hdf5
from .fd_bench import (
    FD_BENCH_COLLECTION_URL,
    FD_BENCH_DATASET_PRESETS,
    download_fd_bench_data,
    download_fd_bench_from_hf,
    generate_fd_bench_raw,
    get_fd_bench_dataset_preset,
    list_fd_bench_datasets,
    list_fd_bench_registry_names,
    load_fd_bench_probe_sample,
    load_fd_bench_rows,
    decode_fd_bench_row,
    resolve_fd_bench_config,
    setup_fd_bench_dataset,
    setup_fd_bench_datasets,
)
from .synthetic import (
    generate_burgers_data,
    generate_darcy_data,
    generate_navier_stokes_data,
)


__all__ = [
    "cache_incremental_hdf5",
    "FD_BENCH_COLLECTION_URL",
    "FD_BENCH_DATASET_PRESETS",
    "download_fd_bench_data",
    "download_fd_bench_from_hf",
    "load_fd_bench_probe_sample",
    "load_fd_bench_rows",
    "decode_fd_bench_row",
    "generate_fd_bench_raw",
    "list_fd_bench_datasets",
    "list_fd_bench_registry_names",
    "get_fd_bench_dataset_preset",
    "resolve_fd_bench_config",
    "setup_fd_bench_dataset",
    "setup_fd_bench_datasets",
    "generate_burgers_data",
    "generate_darcy_data",
    "generate_navier_stokes_data",
]
