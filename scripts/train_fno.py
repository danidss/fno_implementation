import argparse

import torch
from torch.utils.data import DataLoader

from src.FNO.train import train_model
from src.data.pytorch_data import get_dataset
from src.utils import set_seed, init_model


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Fourier Neural Operator")
    parser.add_argument("--n_samples", type=int, default=1200)
    parser.add_argument("--train_split", type=float, default=1000 / 1200)
    parser.add_argument(
        "--dataset", type=str, default="burgers", choices=["burgers", "darcy"]
    )
    parser.add_argument(
        "--implementation",
        type=str,
        default="ours",
        choices=["ours", "original"],
    )
    parser.add_argument("--model_path", type=str, default="models/fno_burgers.pth")
    parser.add_argument("--modes", type=int, default=16, help="Number of Fourier modes")
    parser.add_argument(
        "--width",
        type=int,
        default=64,
        help="Width (channels) of the FNO hidden layers",
    )
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument(
        "--subsample", type=int, default=8, help="Spatial sub-sampling factor"
    )
    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset, test_dataset, in_channels, dim = get_dataset(args, args.n_samples)
    print(
        f"Loaded {args.dataset} dataset. Train size: {len(train_dataset)}, Test size: {len(test_dataset)}"
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    hp = {
        "dim": dim,
        "modes": args.modes,
        "width": args.width,
        "layers": args.layers,
        "in_channels": in_channels,
        "out_channels": 1,
        "dataset": args.dataset,
        "subsample": getattr(args, "subsample", 1),
        "implementation": args.implementation,
    }

    model = init_model(hp, device)

    train_model(args, model, train_loader, test_loader, device, dim, in_channels)


if __name__ == "__main__":
    main()
