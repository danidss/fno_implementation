import argparse

from src.eval.eval_framework import evaluate_from_checkpoints


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate FNO Implementations")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=200,
        help="Number of generated samples used for evaluation",
    )
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument(
        "--checkpoint_a",
        type=str,
        required=False,
        help="Path to first model checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint_a",
        type=str,
        help="Backward-compatible alias for --checkpoint_a",
    )
    parser.add_argument(
        "--checkpoint_b",
        type=str,
        default=None,
        help="Optional path to second model checkpoint",
    )
    parser.add_argument(
        "--original_checkpoint",
        dest="checkpoint_b",
        type=str,
        help="Backward-compatible alias for --checkpoint_b",
    )
    parser.add_argument(
        "--name_a",
        type=str,
        default=None,
        help="Optional display name for first model",
    )
    parser.add_argument(
        "--name_b",
        type=str,
        default=None,
        help="Optional display name for second model",
    )
    parser.add_argument(
        "--n_plots",
        type=int,
        default=3,
        help="Backward-compatible alias for --top_k",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Number of best/worst samples to plot per model",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="rel_l2",
        choices=["rel_l2", "mse", "mae"],
        help="Metric used for ranking and error distributions",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Directory to save evaluation plots",
    )
    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    if not args.checkpoint_a:
        parser.error("one of --checkpoint_a/--checkpoint is required")

    top_k = args.top_k if args.top_k is not None else args.n_plots

    evaluate_from_checkpoints(
        checkpoint_a=args.checkpoint_a,
        checkpoint_b=args.checkpoint_b,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        train_split=args.train_split,
        top_k=top_k,
        metric=args.metric,
        output_dir=args.output_dir,
        name_a=args.name_a,
        name_b=args.name_b,
    )


if __name__ == "__main__":
    main()
