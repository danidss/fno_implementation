import os
import torch
import random
import numpy as np
from collections import OrderedDict

from src.FNO.model import FNO, SPECTRAL_CONV
from src.FNO.original_model import OriginalFNO


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
    model = init_model(hp, device)
    state_dict = checkpoint_data.get("model_state_dict", checkpoint_data)
    model.load_state_dict(_clean_state_dict(state_dict))
    name = model_name or model_display_name(hp.get("implementation", "ours"))
    return hp, model, name


def init_our_fno(hp: dict, device: torch.device) -> FNO:
    """Initializes Our FNO model based on hyperparameters."""
    dim = hp["dim"]
    layer_shapes = (hp["width"],) * hp["layers"]
    model = FNO(
        dim=dim,
        modes=hp["modes"],
        layer_shapes=layer_shapes,
        spectral_conv_class=SPECTRAL_CONV[dim],
        norm_class=hp.get("norm_class", None),
        in_channels=hp["in_channels"],
        out_channels=1,
    ).to(device)
    return model


def init_original_fno(hp: dict, device: torch.device) -> torch.nn.Module:
    layer_shapes = (hp["width"],) * hp["layers"]
    model = OriginalFNO(
        dim=hp["dim"],
        modes=hp["modes"],
        layer_shapes=layer_shapes,
        in_channels=hp["in_channels"],
        out_channels=1,
    ).to(device)
    return model


def init_model(hp: dict, device: torch.device) -> torch.nn.Module:
    implementation = hp.get("implementation", "ours")
    if implementation == "ours":
        return init_our_fno(hp, device)
    if implementation == "original":
        return init_original_fno(hp, device)
    raise ValueError(f"Unknown implementation '{implementation}'")


def model_display_name(implementation: str) -> str:
    names = {
        "ours": "Our FNO Implementation",
        "original": "NeuralOperator FNO",
    }
    return names.get(implementation, implementation)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
