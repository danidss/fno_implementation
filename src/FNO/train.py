import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.FNO.model import LpLoss
from src.utils import check_and_save_checkpoint


def train_model(
    cfg,
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    dim: int,
    in_channels: int,
) -> None:
    # Standard configuration from the paper
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.training.step_size,
        gamma=cfg.training.gamma,
    )

    # 4. Training Loop
    best_val_loss = float("inf")

    # Use relative L2-loss over mean square error
    criterion = LpLoss(size_average=False)

    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        train_l2 = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.training.epochs} [Train]")
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
        current_lr = scheduler.get_last_lr()[0]

        wandb.log(
            {
                "epoch": epoch,
                "train/l2": train_l2,
                "val/l2": val_l2,
                "train/lr": current_lr,
            }
        )

        print(f"Epoch {epoch} | Train Loss: {train_l2:.6f} | Val Loss: {val_l2:.6f}")

        scheduler.step()

        hp = {
            "dim": dim,
            "modes": cfg.model.modes,
            "width": cfg.model.width,
            "layers": cfg.model.layers,
            "in_channels": in_channels,
            "out_channels": 1,
            "dataset": cfg.data.dataset,
            "subsample": cfg.data.subsample,
            "implementation": cfg.model.implementation,
        }
        best_val_loss = check_and_save_checkpoint(
            val_l2=val_l2,
            best_val_loss=best_val_loss,
            model=model,
            checkpoint_path=cfg.model.model_path,
            hp=hp,
        )
