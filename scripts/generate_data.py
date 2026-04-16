import argparse

from src.data.providers_builtin import register_builtin_datasets
from src.data.registry import generate_raw_data, list_datasets
from src.utils import set_seed


def get_parser() -> argparse.ArgumentParser:
    register_builtin_datasets()
    available = list_datasets(require_raw_generator=True)

    parser = argparse.ArgumentParser(description="Generate PDE datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            name for name in ["burgers", "darcy", "navier_stokes"] if name in available
        ]
        or available,
        choices=available,
        help="Datasets to generate",
    )
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--burgers_samples", type=int, default=1000)
    parser.add_argument("--darcy_samples", type=int, default=1000)
    parser.add_argument("--navier_stokes_samples", type=int, default=100)
    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    set_seed()
    register_builtin_datasets()

    samples_per_dataset = {
        "burgers": args.burgers_samples,
        "darcy": args.darcy_samples,
        "navier_stokes": args.navier_stokes_samples,
    }

    for dataset_name in args.datasets:
        n_samples = samples_per_dataset.get(dataset_name, args.n_samples)
        print(f"Generating {dataset_name} data...")
        generate_raw_data(dataset_name, n_samples=n_samples)


if __name__ == "__main__":
    main()
