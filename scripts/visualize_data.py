import argparse

from src.data.data_visualization import (
    visualize_burgers_comparison,
    visualize_burgers_samples,
    visualize_burgers_statistics,
    visualize_darcy_samples,
    visualize_darcy_statistics,
    visualize_navier_stokes_difference,
    visualize_navier_stokes_samples,
    visualize_navier_stokes_statistics,
)
from src.data.generate_data import (
    generate_burgers_data,
    generate_darcy_data,
    generate_navier_stokes_data,
)


def _get_visualization_registry():
    return {
        "darcy": {
            "title": "Darcy Flow",
            "generator": generate_darcy_data,
            "sample_plotters": [visualize_darcy_samples],
            "stats_plotters": [visualize_darcy_statistics],
            "plots_arg": "n_plots",
        },
        "burgers": {
            "title": "Burgers 1-D",
            "generator": generate_burgers_data,
            "sample_plotters": [visualize_burgers_samples, visualize_burgers_comparison],
            "stats_plotters": [visualize_burgers_statistics],
            "plots_arg": "n_plots",
        },
        "navier_stokes": {
            "title": "Navier-Stokes",
            "generator": generate_navier_stokes_data,
            "sample_plotters": [
                visualize_navier_stokes_samples,
                visualize_navier_stokes_difference,
            ],
            "stats_plotters": [visualize_navier_stokes_statistics],
            "plots_arg": "n_plots_navier",
        },
    }


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize generated PDE datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["darcy", "burgers"],
        choices=["burgers", "darcy", "navier_stokes"],
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
        data = config["generator"](n_samples=args.n_samples)

        n_plots = getattr(args, config["plots_arg"])
        for plotter in config["sample_plotters"]:
            plotter(data, n_samples=n_plots)

        for plotter in config["stats_plotters"]:
            plotter(data)


if __name__ == "__main__":
    main()
