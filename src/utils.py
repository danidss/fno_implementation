import os
import torch
import random
import numpy as np
import wandb
from collections import OrderedDict


def build_activation(nonlinearity: str) -> torch.nn.Module:
    activation_registry = {
        "relu": torch.nn.ReLU,
        "gelu": torch.nn.GELU,
        "silu": torch.nn.SiLU,
        "tanh": torch.nn.Tanh,
        "elu": torch.nn.ELU,
        "leaky_relu": torch.nn.LeakyReLU,
    }
    key = nonlinearity.lower()
    if key not in activation_registry:
        options = ", ".join(sorted(activation_registry))
        raise ValueError(
            f"Unsupported nonlinearity '{nonlinearity}'. Available: {options}."
        )
    return activation_registry[key]()


def build_normalization(
    norm_class: str | None,
    num_channels: int,
    dim: int,
) -> torch.nn.Module | None:
    if norm_class is None or norm_class.lower() == "none":
        return None

    norm_key = norm_class.lower()
    if norm_key == "batch":
        if dim == 1:
            return torch.nn.BatchNorm1d(num_channels)
        if dim == 2:
            return torch.nn.BatchNorm2d(num_channels)
        if dim == 3:
            return torch.nn.BatchNorm3d(num_channels)
        raise ValueError(
            "Batch normalization only supports 1D/2D/3D inputs in this implementation. "
            f"Received dim={dim}."
        )
    elif norm_key == "instance":
        if dim == 1:
            return torch.nn.InstanceNorm1d(num_channels)
        if dim == 2:
            return torch.nn.InstanceNorm2d(num_channels)
        if dim == 3:
            return torch.nn.InstanceNorm3d(num_channels)
        raise ValueError(
            "Instance normalization only supports 1D/2D/3D inputs in this implementation. "
            f"Received dim={dim}."
        )
    elif norm_key == "layer":
        # Channel-wise LayerNorm equivalent for channel-first tensors.
        return torch.nn.GroupNorm(1, num_channels)

    raise ValueError(
        f"Unsupported norm_class '{norm_class}'. Available: none, batch, instance, layer."
    )


def set_seed(seed: int = 42) -> None:
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_and_save_checkpoint(
    val_l2: float,
    best_val_loss: float,
    model: torch.nn.Module,
    checkpoint_path: str,
    hp: dict,
) -> float:
    """Saves the model state dict and hyperparameters if it exceeds the best loss."""
    if val_l2 < best_val_loss:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "hyperparameters": hp,
            },
            checkpoint_path,
        )
        print(f"Saved new best model to {checkpoint_path}")
        return val_l2
    return best_val_loss


def load_checkpoint_hyperparameters(
    checkpoint_path: str, device: torch.device
) -> tuple[dict, dict]:
    """Loads checkpoint and extracts hyperparameters. Falls back to defaults if missing."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint_data = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint_data, dict) and "hyperparameters" in checkpoint_data:
        hp = checkpoint_data["hyperparameters"]
        print("Loaded hyperparameters from checkpoint.")
        return hp, checkpoint_data
    else:
        raise ValueError(
            "Checkpoint does not contain 'hyperparameters'. Cannot proceed with loading."
        )


def _clean_state_dict(state_dict: dict) -> dict:
    if isinstance(state_dict, dict) and "_metadata" in state_dict:
        return OrderedDict(
            (key, value) for key, value in state_dict.items() if key != "_metadata"
        )
    return state_dict


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    model_name: str | None,
) -> tuple[dict, torch.nn.Module, str]:
    hp, checkpoint_data = load_checkpoint_hyperparameters(checkpoint_path, device)
    implementation = hp.get("implementation", "ours")
    modes = tuple(hp["modes"])
    layer_shapes = (hp["width"],) * hp["layers"]

    if implementation == "ours":
        from src.FNO.model import FNO

        model = FNO(
            modes=modes,
            in_channels=hp["in_channels"],
            out_channels=1,
            layer_shapes=layer_shapes,
            norm_class=hp.get("norm_class", "none"),
            nonlinearity=hp.get("nonlinearity", "relu"),
            lift_hidden_dims=tuple(hp.get("lift_hidden_dims", ())),
            projection_hidden_dims=tuple(hp.get("projection_hidden_dims", ())),
            skip_hidden_dims=tuple(hp.get("skip_hidden_dims", ())),
            pointwise_dropout=hp.get("pointwise_dropout", 0.0),
        ).to(device)
    elif implementation == "original":
        from src.FNO.original_model import OriginalFNO

        model = OriginalFNO(
            modes=modes,
            layer_shapes=layer_shapes,
            in_channels=hp["in_channels"],
            out_channels=1,
            norm_class=hp.get("norm_class", "none"),
            nonlinearity=hp.get("nonlinearity", "relu"),
            lift_hidden_dims=tuple(hp.get("lift_hidden_dims", ())),
            projection_hidden_dims=tuple(hp.get("projection_hidden_dims", ())),
        ).to(device)
    else:
        raise ValueError(f"Unknown implementation '{implementation}'")

    state_dict = checkpoint_data.get("model_state_dict", checkpoint_data)
    model.load_state_dict(_clean_state_dict(state_dict))
    name = model_name or model_display_name(hp.get("implementation", "ours"))
    return hp, model, name


def load_model_from_wandb_artifact(
    artifact_path: str,
    device: torch.device,
    model_name: str | None = None,
    download_dir: str = ".wandb_artifacts",
) -> tuple[dict, torch.nn.Module, str]:
    """Downloads a model artifact from W&B and loads it as a local checkpoint."""
    api = wandb.Api()
    artifact = api.artifact(artifact_path, type="model")
    artifact_dir = artifact.download(root=download_dir)

    checkpoint_candidates = [
        os.path.join(artifact_dir, filename)
        for filename in os.listdir(artifact_dir)
        if filename.endswith((".pth", ".pt"))
    ]
    if not checkpoint_candidates:
        raise FileNotFoundError(
            f"No checkpoint file found in artifact '{artifact_path}' at '{artifact_dir}'"
        )

    checkpoint_path = sorted(checkpoint_candidates)[0]
    return load_model_from_checkpoint(checkpoint_path, device, model_name)


def model_display_name(implementation: str) -> str:
    names = {
        "ours": "Our FNO Implementation",
        "original": "NeuralOperator FNO",
    }
    return names.get(implementation, implementation)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
