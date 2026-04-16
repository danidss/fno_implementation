import argparse
from types import SimpleNamespace

from src.data.providers_builtin import register_builtin_datasets
from src.data.registry import build_train_test_split_from_args, list_datasets


def get_parser() -> argparse.ArgumentParser:
    register_builtin_datasets()
    available = list_datasets(require_builder=True)

    parser = argparse.ArgumentParser(description="Inspect generated training dataset")
    parser.add_argument("--dataset", type=str, required=True, choices=available)
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
    train_dataset, test_dataset, spec = build_train_test_split_from_args(
        dataset_args,
        n_samples=args.n_samples,
    )
    if spec.input_channels is None or spec.spatial_dim is None:
        raise ValueError(
            f"Dataset '{args.dataset}' is not trainable because input_channels/spatial_dim are undefined."
        )

    print(f"Example input: {train_dataset[0][0]}")
    print(f"Example input shape: {train_dataset[0][0].shape}")
    print(f"Example target: {train_dataset[0][1]}")
    print(f"Example target shape: {train_dataset[0][1].shape}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print(f"Input channels: {spec.input_channels}")
    print(f"Dimension: {spec.spatial_dim}")


if __name__ == "__main__":
    main()
