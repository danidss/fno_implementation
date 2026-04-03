import os

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.FNO.train import train_model
from src.data.pytorch_data import get_dataset
from src.utils import set_seed, init_model


@hydra.main(version_base=None, config_path="../configs", config_name="train_fno")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    if not cfg.wandb.run_name:
        raise ValueError("wandb.run_name must be set to save checkpoints and artifacts")

    run_config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(run_config, dict):
        raise TypeError("Resolved Hydra config must be a dictionary")

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        mode=cfg.wandb.mode,
        config=run_config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset, test_dataset, in_channels, dim = get_dataset(
        cfg.data,
        cfg.data.n_samples,
    )
    print(
        f"Loaded {cfg.data.dataset} dataset. Train size: {len(train_dataset)}, Test size: {len(test_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    hp = {
        "dim": dim,
        "modes": cfg.model.modes,
        "width": cfg.model.width,
        "layers": cfg.model.layers,
        "in_channels": in_channels,
        "out_channels": 1,
        "norm_class": None,
        "dataset": cfg.data.dataset,
        "subsample": cfg.data.subsample,
        "implementation": cfg.model.implementation,
    }

    checkpoint_path = os.path.join(
        cfg.model.models_folder,
        f"{cfg.wandb.run_name}.pth",
    )

    model = init_model(hp, device)

    try:
        train_model(
            cfg,
            model,
            train_loader,
            test_loader,
            device,
            dim,
            in_channels,
            checkpoint_path=checkpoint_path,
            artifact_name=cfg.wandb.run_name,
        )
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
