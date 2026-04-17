import argparse

from src.data.sources import download_fd_bench_data, download_fd_bench_from_hf


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download FD-Bench data")
    parser.add_argument(
        "--mode",
        choices=["url", "hf"],
        default="url",
        help="Download source mode: direct URL or HuggingFace datasets.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Direct URL to an FD-Bench-compatible .h5/.hdf5 file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_data/fd_bench",
        help="Directory where downloaded files are stored.",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default=None,
        help="Optional output file name for URL mode.",
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default=None,
        help="HuggingFace dataset ID for mode=hf.",
    )
    parser.add_argument(
        "--hf_split",
        type=str,
        default="train",
        help="HuggingFace split to export for mode=hf.",
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()

    if args.mode == "url":
        if not args.url:
            raise ValueError("--url is required when --mode url")
        output = download_fd_bench_data(
            url=args.url,
            output_dir=args.output_dir,
            file_name=args.file_name,
        )
        print(f"Downloaded FD-Bench data to: {output}")
        return

    if not args.hf_dataset:
        raise ValueError("--hf_dataset is required when --mode hf")
    output = download_fd_bench_from_hf(
        dataset_id=args.hf_dataset,
        output_dir=args.output_dir,
        split=args.hf_split,
    )
    print(f"Exported HuggingFace dataset to: {output}")


if __name__ == "__main__":
    main()
