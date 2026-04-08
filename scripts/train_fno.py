import os

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.FNO.train import train_model
from src.FNO.model import FNO
from src.FNO.original_model import OriginalFNO
from src.data.pytorch_data import get_dataset
from src.eval.evaluate import generate_prediction_plots
from src.utils import set_seed, load_model_from_checkpoint


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

    # Initialize model
    modes = tuple(hp["modes"])
    layer_shapes = (hp["width"],) * hp["layers"]
    implementation = hp.get("implementation", "ours")

    if implementation == "ours":
        model = FNO(
            modes=modes,
            in_channels=hp["in_channels"],
            out_channels=1,
            layer_shapes=layer_shapes,
            norm_class=hp.get("norm_class", None),
        ).to(device)
    elif implementation == "original":
        model = OriginalFNO(
            modes=modes,
            layer_shapes=layer_shapes,
            in_channels=hp["in_channels"],
            out_channels=1,
        ).to(device)
    else:
        raise ValueError(f"Unknown implementation '{implementation}'")

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
        )
    finally:
        if os.path.exists(checkpoint_path) and cfg.wandb.mode not in {
            "disabled",
            "offline",
        }:
            artifact = wandb.Artifact(name=cfg.wandb.run_name, type="model")
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact, aliases=["best", "latest"])

            try:
                _, best_model, _ = load_model_from_checkpoint(
                    checkpoint_path,
                    device,
                    model_name="best_model",
                )
                plot_paths = generate_prediction_plots(
                    model=best_model,
                    test_dataset=test_dataset,
                    dataset_name=cfg.data.dataset,
                    dim=dim,
                    device=device,
                    n_plots=3,
                    model_name="best_model",
                )

                plot_images = [
                    wandb.Image(path, caption=f"Best model {os.path.basename(path)}")
                    for path in plot_paths
                ]

                if plot_images:
                    wandb.log({"eval/prediction_plots": plot_images})
            except Exception as exc:
                print(f"Warning: failed to generate/upload evaluation plots: {exc}")
        wandb.finish()


if __name__ == "__main__":
    main()
