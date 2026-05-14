import argparse

from src.data.providers_builtin import register_builtin_datasets
from src.data.registry import generate_raw_data, list_datasets
from src.data.data_visualization import (
    visualize_burgers_comparison,
    visualize_burgers_statistics,
    visualize_darcy_samples,
    visualize_darcy_statistics,
    visualize_navier_stokes_difference,
    visualize_navier_stokes_samples,
    visualize_navier_stokes_statistics,
)


def _get_visualization_registry():
    return {
        "darcy": {
            "title": "Darcy Flow",
            "sample_plotters": [visualize_darcy_samples],
            "stats_plotters": [visualize_darcy_statistics],
            "plots_arg": "n_plots",
        },
        "burgers": {
            "title": "Burgers 1-D",
            "sample_plotters": [
                visualize_burgers_comparison,
            ],
            "stats_plotters": [visualize_burgers_statistics],
            "plots_arg": "n_plots",
        },
        "navier_stokes": {
            "title": "Navier-Stokes",
            "sample_plotters": [
                visualize_navier_stokes_samples,
                visualize_navier_stokes_difference,
            ],
            "stats_plotters": [visualize_navier_stokes_statistics],
            "plots_arg": "n_plots_navier",
        },
    }


def get_parser() -> argparse.ArgumentParser:
    register_builtin_datasets()
    visualizable = sorted(
        set(_get_visualization_registry().keys())
        & set(list_datasets(require_raw_generator=True))
    )

    parser = argparse.ArgumentParser(description="Visualize generated PDE datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[name for name in ["darcy", "burgers"] if name in visualizable]
        or visualizable,
        choices=visualizable,
        help="Datasets to visualize",
    )
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--n_plots", type=int, default=5)
    parser.add_argument("--n_plots_navier", type=int, default=3)
    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    registry = _get_visualization_registry()

    for dataset_name in args.datasets:
        config = registry[dataset_name]
        print(f"Loading {config['title']} data …")
        data = generate_raw_data(dataset_name, n_samples=args.n_samples)

        n_plots = getattr(args, config["plots_arg"])
        for plotter in config["sample_plotters"]:
            plotter(data, n_samples=n_plots)

        for plotter in config["stats_plotters"]:
            plotter(data)


if __name__ == "__main__":
    main()
