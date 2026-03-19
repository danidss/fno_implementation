import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.FNO.model import LpLoss
from src.utils import check_and_save_checkpoint


def train_model(
    args,
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    dim: int,
    in_channels: int,
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

        hp = {
            "dim": dim,
            "modes": args.modes,
            "width": args.width,
            "layers": args.layers,
            "in_channels": in_channels,
            "out_channels": 1,
            "dataset": args.dataset,
            "subsample": getattr(args, "subsample", 1),
            "implementation": getattr(args, "implementation", "ours"),
        }
        best_val_loss = check_and_save_checkpoint(
            val_l2=val_l2,
            best_val_loss=best_val_loss,
            model=model,
            checkpoint_path=args.model_path,
            hp=hp,
        )
