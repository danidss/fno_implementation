import argparse

from src.data.sources import list_fd_bench_datasets, setup_fd_bench_dataset


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect/setup HuggingFace FD-Bench presets (download is disabled)."
    )
    parser.add_argument(
        "--dataset",
        choices=list_fd_bench_datasets(),
        default=None,
        help="FD-Bench preset name to inspect or check.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available FD-Bench presets.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional split override when checking a preset.",
    )
    parser.add_argument(
        "--check_access",
        action="store_true",
        help="Try loading the HuggingFace split to validate runtime access.",
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()

    if args.list:
        for name in list_fd_bench_datasets():
            print(name)
        return

    if not args.dataset:
        raise ValueError("Use --list or provide --dataset <name>")

    cfg = setup_fd_bench_dataset(
        fd_dataset=args.dataset,
        split=args.split,
        check_access=args.check_access,
    )
    print(f"Preset: {cfg['fd_dataset']}")
    print(f"Registry name: {cfg['registry_name']}")
    print(f"HF dataset: {cfg['hf_dataset_id']}")
    print(f"Split: {cfg['split']}")
    print(f"Spatial dim: {cfg['spatial_dim']}")
    print(f"Input channels: {cfg['input_channels']}")
    print(f"Output channels: {cfg['out_channels']}")
    print(f"Access check: {'ok' if args.check_access else 'skipped'}")


if __name__ == "__main__":
    main()
