import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.FNO.model import FNO, LpLoss
from src.data.pytorch_data import get_dataset


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Fourier Neural Operator")
    parser.add_argument("--n_samples", type=int, default=1200)
    parser.add_argument("--train_split", type=float, default=1000 / 1200)
    parser.add_argument(
        "--dataset", type=str, default="burgers", choices=["burgers", "darcy"]
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


def train_model(
    args: argparse.Namespace,
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> None:
    # Standard configuration from the paper
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

    # 4. Training Loop
    best_val_loss = float("inf")

    # Use relative L2-loss over mean square error
    criterion = LpLoss(size_average=False)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_l2 = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)

            # Loss and Backward
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_l2 += loss.item()

            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                }
            )

        # Validation Evaluation
        model.eval()
        val_l2 = 0.0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                out = model(x)
                loss = criterion(out, y)
                val_l2 += loss.item()

        # Aggregate epoch metrics
        train_l2 /= len(train_loader)
        val_l2 /= len(test_loader)

        print(f"Epoch {epoch} | Train Loss: {train_l2:.6f} | Val Loss: {val_l2:.6f}")

        scheduler.step()

        if val_l2 < best_val_loss:
            best_val_loss = val_l2
            checkpoint_path = args.model_path
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved new best model to {checkpoint_path}")


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset, test_dataset, in_channels, dim = get_dataset(args, args.n_samples)
    print(
        f"Loaded {args.dataset} dataset. Train size: {len(train_dataset)}, Test size: {len(test_dataset)}"
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    layer_shapes = (args.width,) * args.layers

    model = FNO(
        dim=dim,
        modes=args.modes,
        layer_shapes=layer_shapes,
        in_channels=in_channels,
        out_channels=1,
    ).to(device)

    train_model(args, model, train_loader, test_loader, device)


if __name__ == "__main__":
    main()
