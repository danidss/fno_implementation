import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from src.FNO.model import LpLoss


def evaluate_model(model: torch.nn.Module, test_loader, device: torch.device):
    """
    Evaluates the model on test data to record error metrics and efficiency.

    Args:
        model: FNO or benchmark model.
        test_loader: DataLoader for testing.
        device: CPU or CUDA device.

    Returns:
        dict: Performance and error metrics.
    """
    model.eval()
    criterion_l2 = LpLoss(size_average=True)

    total_l2 = 0.0
    total_mse = 0.0
    total_mae = 0.0

    import time
    start_time = time.perf_counter()

    sample_elements = None
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            sample_elements = y[0].numel()

            out = model(x)

            total_l2 += criterion_l2(out, y).item() * x.size(0)
            total_mse += F.mse_loss(out, y, reduction="sum").item()
            total_mae += F.l1_loss(out, y, reduction="sum").item()

    end_time = time.perf_counter()

    n_samples = len(test_loader.dataset)
    if sample_elements is None:
        raise ValueError("Test loader is empty")

    avg_l2 = total_l2 / n_samples
    avg_mse = total_mse / (n_samples * sample_elements)
    avg_mae = total_mae / (n_samples * sample_elements)

    avg_inference_time = (end_time - start_time) / n_samples

    from src.utils import count_parameters
    params = count_parameters(model)

    return {
        "params": params,
        "inference_time_ms": avg_inference_time * 1000,
        "rel_l2": avg_l2,
        "mse": avg_mse,
        "mae": avg_mae,
    }


def print_results_table(model_name: str, results: dict):
    print(f"--- {model_name} ---")
    print(f"Parameters   : {results['params']:,}")
    print(f"Inference    : {results['inference_time_ms']:.4f} ms/sample")
    print(f"Relative L2  : {results['rel_l2']:.6f}")
    print(f"MSE          : {results['mse']:.6f}")
    print(f"MAE          : {results['mae']:.6f}")
    print("-" * 30)


def plot_prediction(
    dataset_name, dim, sample_x, sample_y, pred, model_name="Model", sample_idx=0
):
    os.makedirs("plots", exist_ok=True)
    if dim == 1:
        plt.figure(figsize=(10, 5))
        x_grid = sample_x[1, :].cpu().numpy()
        y_true = sample_y[0, :].cpu().numpy()
        y_pred = pred[0, :].cpu().numpy()

        plt.plot(
            x_grid,
            y_true,
            label="Ground Truth",
            linestyle="-",
            color="black",
            alpha=0.7,
        )
        plt.plot(x_grid, y_pred, label=model_name, linestyle="--", color="blue")

        plt.title(
            f"{dataset_name.capitalize()} - 1D Predictions - {model_name} (Sample {sample_idx})"
        )
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        filename = model_name.replace(" ", "_").replace("'", "").lower()
        output_path = f"plots/eval_{dataset_name}_1d_{filename}_sample{sample_idx}.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path
    elif dim == 2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        y_true = sample_y[0].cpu().numpy()
        y_pred = pred[0].cpu().numpy()

        im0 = axes[0].imshow(y_true, origin="lower", cmap="viridis")
        axes[0].set_title("Ground Truth")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(y_pred, origin="lower", cmap="viridis")
        axes[1].set_title(model_name)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(np.abs(y_true - y_pred), origin="lower", cmap="plasma")
        axes[2].set_title(f"|Truth - {model_name}|")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.suptitle(
            f"{dataset_name.capitalize()} - 2D Predictions - {model_name} (Sample {sample_idx})"
        )
        plt.tight_layout()
        filename = model_name.replace(" ", "_").replace("'", "").lower()
        output_path = f"plots/eval_{dataset_name}_2d_{filename}_sample{sample_idx}.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path
    return None


def plot_models_comparison(
    dataset_name,
    dim,
    sample_x,
    sample_y,
    pred1,
    pred2,
    name1="Model 1",
    name2="Model 2",
    sample_idx=0,
):
    os.makedirs("plots", exist_ok=True)
    if dim == 1:
        plt.figure(figsize=(10, 5))
        x_grid = sample_x[1, :].cpu().numpy()
        y_true = sample_y[0, :].cpu().numpy()
        y_pred1 = pred1[0, :].cpu().numpy()
        y_pred2 = pred2[0, :].cpu().numpy()

        plt.plot(
            x_grid,
            y_true,
            label="Ground Truth",
            linestyle="-",
            color="black",
            alpha=0.7,
        )
        plt.plot(x_grid, y_pred1, label=name1, linestyle="--", color="blue")
        plt.plot(x_grid, y_pred2, label=name2, linestyle=":", color="red")

        plt.title(
            f"{dataset_name.capitalize()} - 1D Comparison: {name1} vs {name2} (Sample {sample_idx})"
        )
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        filename1 = name1.replace(" ", "_").replace("'", "").lower()
        filename2 = name2.replace(" ", "_").replace("'", "").lower()
        output_path = f"plots/eval_{dataset_name}_1d_compare_{filename1}_{filename2}_sample{sample_idx}.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path
    elif dim == 2:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        y_true = sample_y[0].cpu().numpy()
        y_pred1 = pred1[0].cpu().numpy()
        y_pred2 = pred2[0].cpu().numpy()

        im0 = axes[0].imshow(y_true, origin="lower", cmap="viridis")
        axes[0].set_title("Ground Truth")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(y_pred1, origin="lower", cmap="viridis")
        axes[1].set_title(name1)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(y_pred2, origin="lower", cmap="viridis")
        axes[2].set_title(name2)
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        im3 = axes[3].imshow(np.abs(y_pred1 - y_pred2), origin="lower", cmap="plasma")
        axes[3].set_title(f"|{name1} - {name2}|")
        fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

        plt.suptitle(
            f"{dataset_name.capitalize()} - 2D Comparison: {name1} vs {name2} (Sample {sample_idx})"
        )
        plt.tight_layout()
        filename1 = name1.replace(" ", "_").replace("'", "").lower()
        filename2 = name2.replace(" ", "_").replace("'", "").lower()
        output_path = f"plots/eval_{dataset_name}_2d_compare_{filename1}_{filename2}_sample{sample_idx}.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path
    return None


def generate_prediction_plots(
    model: torch.nn.Module,
    test_dataset,
    dataset_name: str,
    dim: int,
    device: torch.device,
    n_plots: int = 3,
    model_name: str = "Model",
) -> list[str]:
    """Generates prediction plots and returns the list of saved image paths."""
    model.eval()
    saved_paths = []

    n_plots_actual = min(n_plots, len(test_dataset))
    for idx in range(n_plots_actual):
        sample_x, sample_y = test_dataset[idx]
        sample_x_batch = sample_x.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(sample_x_batch).squeeze(0).cpu()

        path = plot_prediction(
            dataset_name,
            dim,
            sample_x.cpu(),
            sample_y.cpu(),
            pred,
            model_name=model_name,
            sample_idx=idx,
        )
        if path and os.path.exists(path):
            saved_paths.append(path)

    return saved_paths


def evaluate_models(
    models: list[tuple[str, torch.nn.Module]],
    test_loader,
    test_dataset,
    dataset_name: str,
    dim: int,
    device: torch.device,
    n_plots: int = 1,
):
    for name, model in models:
        results = evaluate_model(model, test_loader, device)
        print_results_table(name, results)

    print(f"Generating plots for {dataset_name.capitalize()}...")
    for _, model in models:
        model.eval()

    n_plots_actual = min(n_plots, len(test_dataset))
    for i in range(n_plots_actual):
        sample_x, sample_y = test_dataset[i]
        sample_x_batch = sample_x.unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = [
                (name, model(sample_x_batch).squeeze(0)) for name, model in models
            ]

        for name, pred in predictions:
            plot_prediction(
                dataset_name,
                dim,
                sample_x,
                sample_y,
                pred,
                model_name=name,
                sample_idx=i,
            )

        if len(predictions) >= 2:
            plot_models_comparison(
                dataset_name,
                dim,
                sample_x,
                sample_y,
                predictions[0][1],
                predictions[1][1],
                name1=predictions[0][0],
                name2=predictions[1][0],
                sample_idx=i,
            )

    print("Plots saved in the 'plots' directory.")


def evaluate_and_compare_models(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    name1: str,
    name2: str,
    test_loader,
    test_dataset,
    dataset_name: str,
    dim: int,
    device: torch.device,
    n_plots: int = 1,
):
    models = [(name1, model1)]
    if model2 is not None:
        models.append((name2, model2))
    evaluate_models(
        models=models,
        test_loader=test_loader,
        test_dataset=test_dataset,
        dataset_name=dataset_name,
        dim=dim,
        device=device,
        n_plots=n_plots,
    )
