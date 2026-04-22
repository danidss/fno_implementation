import argparse
from collections import Counter
import json
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.data.providers_builtin import register_builtin_datasets
from src.data.registry import build_train_test_split_from_args, list_datasets
from src.data.sources import (
    decode_fd_bench_row,
    get_fd_bench_dataset_preset,
    list_fd_bench_datasets,
    load_fd_bench_probe_sample,
)


def get_parser() -> argparse.ArgumentParser:
    register_builtin_datasets()
    available = list_datasets(require_builder=True)

    parser = argparse.ArgumentParser(
        description="Inspect datasets and FD-Bench metadata"
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--dataset",
        type=str,
        choices=available,
        help="Inspect a single registry dataset through build_train_test_split.",
    )
    mode_group.add_argument(
        "--fd_bench_overview",
        action="store_true",
        help="Inspect one sample from each FD-Bench preset and compare metadata vs decoded shape.",
    )
    mode_group.add_argument(
        "--fd_bench_deep_inspect",
        action="store_true",
        help=(
            "Inspect raw FD-Bench row structure and decoding behavior for one or all presets."
        ),
    )

    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--temporal_subsample", type=int, default=1)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--fd_bench_split", type=str, default=None)
    parser.add_argument(
        "--fd_bench_preset",
        type=str,
        default=None,
        help="Inspect one FD-Bench preset in deep mode (default: inspect all presets).",
    )
    parser.add_argument(
        "--fd_bench_probe_rows",
        type=int,
        default=3,
        help="Number of streaming rows to inspect per preset in deep mode.",
    )
    parser.add_argument(
        "--fd_bench_show_columns",
        type=int,
        default=20,
        help="Maximum number of array-like columns printed per inspected row.",
    )
    parser.add_argument(
        "--fd_bench_decode_candidates",
        type=str,
        default="1,2,25,51",
        help=(
            "Comma-separated out_channels candidates to test in deep mode, "
            "for example: '1,2,25,51'."
        ),
    )
    parser.add_argument(
        "--dataset_kwargs_json",
        type=str,
        default="{}",
        help="JSON mapping passed to registry dataset builder.",
    )
    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    if args.fd_bench_overview:
        inspect_fd_bench_overview(args)
        return

    if args.fd_bench_deep_inspect:
        inspect_fd_bench_deep(args)
        return

    dataset_kwargs = json.loads(args.dataset_kwargs_json)
    inspect_single_dataset(args, dataset_kwargs)


def inspect_single_dataset(
    args: argparse.Namespace, dataset_kwargs: dict[str, Any]
) -> None:
    dataset_args = SimpleNamespace(
        dataset=args.dataset,
        subsample=args.subsample,
        train_split=args.train_split,
        dataset_kwargs=dataset_kwargs,
    )
    train_dataset, test_dataset, spec = build_train_test_split_from_args(
        dataset_args,
        n_samples=args.n_samples,
    )

    sample_x, sample_y = train_dataset[0]

    print("=" * 72)
    print("Dataset Inspection")
    print("=" * 72)
    print(f"Dataset           : {args.dataset}")
    print(f"Train/Test sizes  : {len(train_dataset)} / {len(test_dataset)}")
    print(f"Spec dim          : {spec.dim}")
    print(f"Spec in/out chans : {spec.input_channels} / {spec.out_channels}")
    print(f"Sample x shape    : {tuple(sample_x.shape)}")
    print(f"Sample y shape    : {tuple(sample_y.shape)}")
    print("=" * 72)


def inspect_fd_bench_overview(args: argparse.Namespace) -> None:
    rows_out: list[list[str]] = []

    for fd_name in list_fd_bench_datasets():
        preset = get_fd_bench_dataset_preset(fd_name)
        try:
            sample, config = load_fd_bench_probe_sample(
                fd_dataset=fd_name,
                split=args.fd_bench_split,
                subsample=args.subsample,
                temporal_subsample=args.temporal_subsample,
                batch_subsample=1,
                shuffle=False,
                seed=args.seed,
                hf_cache_dir=args.hf_cache_dir,
            )

            sample = np.asarray(sample)
            decoded_dim = max(0, sample.ndim - 2)
            decoded_channels = int(sample.shape[-1]) if sample.ndim >= 1 else -1
            status = (
                "OK"
                if (
                    int(config["spatial_dim"]) == decoded_dim
                    and int(config["out_channels"]) == decoded_channels
                )
                else "MISMATCH"
            )
            rows_out.append(
                [
                    fd_name,
                    str(config["registry_name"]),
                    str(config["family"]),
                    str(config.get("description", "")),
                    str(config["split"]),
                    str(config["spatial_dim"]),
                    str(config["out_channels"]),
                    str(tuple(sample.shape)),
                    str(decoded_dim),
                    str(decoded_channels),
                    status,
                ]
            )
        except Exception as exc:
            rows_out.append(
                [
                    fd_name,
                    str(preset["registry_name"]),
                    str(preset["family"]),
                    str(preset.get("description", "")),
                    str(args.fd_bench_split or "test"),
                    str(preset["spatial_dim"]),
                    str(preset["out_channels"]),
                    "-",
                    "-",
                    "-",
                    f"ERROR: {exc}",
                ]
            )

    headers = [
        "preset",
        "registry",
        "family",
        "description",
        "split",
        "meta_dim",
        "meta_c",
        "decoded_shape",
        "dec_dim",
        "dec_c",
        "status",
    ]

    print("=" * 144)
    print("FD-Bench Overview (one sample per preset)")
    print("=" * 144)
    print(_format_table(headers, rows_out))
    print("=" * 144)
    print(
        "Legend: meta_* values come from preset metadata; dec_* values are decoded from one loaded sample."
    )


def _parse_decode_candidates(raw: str) -> list[int]:
    candidates: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            candidates.append(max(1, int(token)))
        except ValueError as exc:
            raise ValueError(
                f"Invalid --fd_bench_decode_candidates token '{token}'; expected integers."
            ) from exc
    if not candidates:
        raise ValueError("At least one decode candidate must be provided.")
    return sorted(set(candidates))


def _array_like_columns(row: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    columns: list[tuple[str, np.ndarray]] = []
    for key, value in row.items():
        if key == "fd_sample":
            continue
        arr = np.asarray(value)
        if arr.ndim == 0:
            continue
        columns.append((key, arr))
    return columns


def _summarize_array(arr: np.ndarray) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "shape": tuple(int(x) for x in arr.shape),
        "ndim": int(arr.ndim),
        "dtype": str(arr.dtype),
    }
    if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
        summary["min"] = float(np.nanmin(arr))
        summary["max"] = float(np.nanmax(arr))
        summary["mean"] = float(np.nanmean(arr))
    return summary


def inspect_fd_bench_deep(args: argparse.Namespace) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "HuggingFace datasets is required for FD-Bench deep inspection. "
            "Install with: pip install datasets"
        ) from exc

    candidates = _parse_decode_candidates(args.fd_bench_decode_candidates)
    probe_rows = max(1, int(args.fd_bench_probe_rows))
    show_columns = max(1, int(args.fd_bench_show_columns))
    fd_names = (
        [args.fd_bench_preset] if args.fd_bench_preset else list_fd_bench_datasets()
    )

    print("=" * 160)
    print("FD-Bench Deep Inspection")
    print("=" * 160)
    print(
        "Purpose: expose raw row layout, per-column tensor shapes, and decode outcomes across candidate channel assumptions."
    )
    print(
        f"Settings: split={args.fd_bench_split or 'test'}, probe_rows={probe_rows}, "
        f"decode_candidates={candidates}, show_columns={show_columns}"
    )
    print("=" * 160)

    for fd_name in fd_names:
        preset = get_fd_bench_dataset_preset(fd_name)
        split = args.fd_bench_split or "test"

        print("\n" + "-" * 160)
        print(f"Preset: {fd_name}")
        print(f"Registry: {preset['registry_name']}")
        print(f"HF dataset id: {preset['hf_dataset_id']}")
        print(f"Family: {preset['family']}")
        print(f"Description: {preset.get('description', '')}")
        print(
            "Metadata: "
            f"spatial_dim={preset['spatial_dim']}, "
            f"input_channels={preset['input_channels']}, "
            f"out_channels={preset['out_channels']}, "
            f"append_grid={preset['append_grid']}"
        )

        try:
            rows = load_dataset(
                str(preset["hf_dataset_id"]),
                split=split,
                streaming=True,
                cache_dir=args.hf_cache_dir,
            )
        except Exception as exc:
            print(f"ERROR: unable to stream dataset: {exc}")
            continue

        key_count_hist: Counter[int] = Counter()
        array_col_count_hist: Counter[int] = Counter()
        array_shape_hist: Counter[str] = Counter()
        decode_hist: Counter[str] = Counter()
        rows_checked = 0

        for row in rows:
            rows_checked += 1
            key_count_hist[len(row)] += 1
            columns = _array_like_columns(row)
            array_col_count_hist[len(columns)] += 1

            print(f"\nRow #{rows_checked}")
            print(f"  keys={len(row)} array_like_columns={len(columns)}")
            if not columns:
                print("  WARNING: no array-like columns found")

            for key, arr in columns[:show_columns]:
                summary = _summarize_array(arr)
                array_shape_hist[str(summary["shape"])] += 1
                if {"min", "max", "mean"}.issubset(summary):
                    print(
                        f"  - {key}: shape={summary['shape']} dtype={summary['dtype']} "
                        f"min={summary['min']:.4g} max={summary['max']:.4g} mean={summary['mean']:.4g}"
                    )
                else:
                    print(
                        f"  - {key}: shape={summary['shape']} dtype={summary['dtype']}"
                    )

            for candidate_out_channels in candidates:
                try:
                    decoded = decode_fd_bench_row(
                        row=row,
                        spatial_dim=int(preset["spatial_dim"]),
                        out_channels=int(candidate_out_channels),
                        subsample=max(1, int(args.subsample)),
                        temporal_subsample=max(1, int(args.temporal_subsample)),
                    )
                    decoded = np.asarray(decoded)
                    decode_dim = max(0, decoded.ndim - 2)
                    decode_channels = (
                        int(decoded.shape[-1]) if decoded.ndim >= 1 else -1
                    )
                    outcome = (
                        f"out={candidate_out_channels} -> shape={tuple(decoded.shape)} "
                        f"dec_dim={decode_dim} dec_c={decode_channels}"
                    )
                except Exception as exc:
                    outcome = (
                        f"out={candidate_out_channels} -> ERROR: "
                        f"{type(exc).__name__}: {exc}"
                    )
                decode_hist[outcome] += 1
                print(f"  * {outcome}")

            if rows_checked >= probe_rows:
                break

        print("\nSummary")
        print(f"  rows_checked={rows_checked}")
        print(f"  key_count_hist={dict(key_count_hist)}")
        print(f"  array_col_count_hist={dict(array_col_count_hist)}")
        if array_shape_hist:
            print("  top_array_shapes=")
            for shape, count in array_shape_hist.most_common(10):
                print(f"    {shape}: {count}")
        if decode_hist:
            print("  decode_outcomes=")
            for outcome, count in decode_hist.items():
                print(f"    {outcome} | count={count}")

    print("\n" + "=" * 160)
    print("Deep inspection complete")
    print("=" * 160)


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    head = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"

    lines = [sep, head, sep]
    for row in rows:
        line = (
            "| "
            + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers)))
            + " |"
        )
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
