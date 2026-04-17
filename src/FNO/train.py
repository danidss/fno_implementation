import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.FNO.model import LpLoss
from src.utils import check_and_save_checkpoint
from src.eval.evaluate import evaluate_model


def train_model(
    cfg,
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    in_channels: int,
    out_channels: int,
    dim: int,
    checkpoint_path: str,
) -> None:
    """
    Trains the FNO model using the provided configuration and data loaders.

    Args:
        cfg: Omegaconf config object.
        model: FNO or wrapper model.
        train_loader: Training data generator.
        test_loader: Validation data generator.
        device: CPU or CUDA device.
        in_channels: Feature count of input.
        out_channels: Feature count of output.
        dim: Dimensionality of spatial domain.
        checkpoint_path: Storage path for best model.
    """
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

        val_results = evaluate_model(model, test_loader, device)

        wandb.log(
            {
                "train_loss": train_l2 / len(train_loader.dataset),
                "val_rel_l2": val_results["rel_l2"],
                "val_mse": val_results["mse"],
                "val_mae": val_results["mae"],
            },
            step=epoch,
        )

        print(
            f"Epoch {epoch} | Train Loss: {train_l2:.6f} | Val Loss: {val_results['rel_l2']:.6f}"
        )

        scheduler.step()

        hp = {
            "modes": cfg.model.modes,
            "width": cfg.model.width,
            "layers": cfg.model.layers,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "dim": dim,
            "dataset": cfg.data.dataset,
            "subsample": cfg.data.subsample,
            "implementation": cfg.model.implementation,
        }
        best_val_loss = check_and_save_checkpoint(
            val_l2=val_results["rel_l2"],
            best_val_loss=best_val_loss,
            model=model,
            checkpoint_path=checkpoint_path,
            hp=hp,
        )
