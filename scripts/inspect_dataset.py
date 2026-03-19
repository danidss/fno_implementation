import argparse
from types import SimpleNamespace

from src.data.pytorch_data import get_dataset


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect generated training dataset")
    parser.add_argument("--dataset", type=str, required=True, choices=["burgers", "darcy"])
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--train_split", type=float, default=0.8)
    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    dataset_args = SimpleNamespace(
        dataset=args.dataset,
        subsample=args.subsample,
        train_split=args.train_split,
    )
    train_dataset, test_dataset, in_channels, dim = get_dataset(
        dataset_args, args.n_samples
    )

    print(f"Example input: {train_dataset[0][0]}")
    print(f"Example input shape: {train_dataset[0][0].shape}")
    print(f"Example target: {train_dataset[0][1]}")
    print(f"Example target shape: {train_dataset[0][1].shape}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print(f"Input channels: {in_channels}")
    print(f"Dimension: {dim}")


if __name__ == "__main__":
    main()
