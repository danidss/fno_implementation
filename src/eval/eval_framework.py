from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader

from src.data.pytorch_data import get_dataset
from src.utils import set_seed, load_model_from_checkpoint

from src.eval.evaluate import evaluate_models


def _validate_compatible_checkpoints(hp_a: dict, hp_b: dict) -> None:
    keys = ("dataset", "subsample", "dim")
    for key in keys:
        if hp_a.get(key) != hp_b.get(key):
            raise ValueError(
                f"Checkpoints are incompatible for '{key}': {hp_a.get(key)} != {hp_b.get(key)}"
            )


def evaluate_from_checkpoints(
    checkpoint_a: str,
    checkpoint_b: str | None = None,
    n_samples: int = 200,
    batch_size: int = 20,
    train_split: float = 0.8,
    n_plots: int = 3,
    name_a: str | None = None,
    name_b: str | None = None,
) -> None:
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    hp_a, model_a, resolved_name_a = load_model_from_checkpoint(
        checkpoint_a, device, name_a
    )

    args = SimpleNamespace(
        dataset=hp_a["dataset"],
        subsample=hp_a.get("subsample", 1),
        train_split=train_split,
    )

    _, test_dataset, _, dim = get_dataset(args, n_samples=n_samples)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(
        f"Dataset: {args.dataset.capitalize()} | Evaluated on {len(test_dataset)} test samples\n"
    )
    print(f"Loaded checkpoint A: {resolved_name_a}\n")

    models = [(resolved_name_a, model_a)]

    if checkpoint_b:
        hp_b, model_b, resolved_name_b = load_model_from_checkpoint(
            checkpoint_b, device, name_b
        )
        _validate_compatible_checkpoints(hp_a, hp_b)
        models.append((resolved_name_b, model_b))
        print(f"Loaded checkpoint B: {resolved_name_b}\n")

    evaluate_models(
        models=models,
        test_loader=test_loader,
        test_dataset=test_dataset,
        dataset_name=args.dataset,
        dim=dim,
        device=device,
        n_plots=n_plots,
    )
