import argparse

from src.data.generate_data import (
    generate_burgers_data,
    generate_darcy_data,
    generate_navier_stokes_data,
)
from src.utils import set_seed


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate PDE datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["burgers", "darcy", "navier_stokes"],
        choices=["burgers", "darcy", "navier_stokes"],
        help="Datasets to generate",
    )
    parser.add_argument("--burgers_samples", type=int, default=1000)
    parser.add_argument("--darcy_samples", type=int, default=1000)
    parser.add_argument("--navier_stokes_samples", type=int, default=100)
    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    set_seed()

    if "burgers" in args.datasets:
        print("Generating Burgers' data...")
        generate_burgers_data(n_samples=args.burgers_samples)

    if "darcy" in args.datasets:
        print("Generating Darcy Flow data...")
        generate_darcy_data(n_samples=args.darcy_samples)

    if "navier_stokes" in args.datasets:
        print("Generating Navier-Stokes data...")
        generate_navier_stokes_data(n_samples=args.navier_stokes_samples)


if __name__ == "__main__":
    main()
