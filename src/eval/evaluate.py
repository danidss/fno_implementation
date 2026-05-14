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


def _sanitize_name(name: str) -> str:
    return name.replace(" ", "_").replace("'", "").lower()


def _ensure_output_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)


def _resolve_x_grid(sample_x: torch.Tensor, sample_y: torch.Tensor) -> np.ndarray:
    if sample_x.ndim >= 2 and sample_x.shape[0] >= 2:
        return sample_x[1, :].cpu().numpy()
    y = sample_y.squeeze().cpu().numpy()
    return np.linspace(0, 1, len(y))


def _sample_error(y_true: torch.Tensor, y_pred: torch.Tensor, metric: str) -> float:
    if metric == "rel_l2":
        denom = torch.norm(y_true) + 1e-12
        return (torch.norm(y_pred - y_true) / denom).item()
    if metric == "mse":
        return F.mse_loss(y_pred, y_true, reduction="mean").item()
    if metric == "mae":
        return F.l1_loss(y_pred, y_true, reduction="mean").item()
    raise ValueError(f"Unknown metric '{metric}'")


def compute_per_sample_errors(
    models: list[tuple[str, torch.nn.Module]],
    test_dataset,
    device: torch.device,
    metric: str,
) -> dict[str, np.ndarray]:
    errors = {name: [] for name, _ in models}
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            sample_x, sample_y = test_dataset[idx]
            sample_x = sample_x.unsqueeze(0).to(device)
            sample_y = sample_y.to(device)

            for name, model in models:
                pred = model(sample_x).squeeze(0)
                errors[name].append(_sample_error(sample_y, pred, metric))

    return {name: np.array(vals) for name, vals in errors.items()}


def plot_error_distributions(
    errors: dict[str, np.ndarray],
    dataset_name: str,
    metric: str,
    output_dir: str,
) -> str:
    _ensure_output_dir(output_dir)
    labels = list(errors.keys())
    data = [errors[label] for label in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
    means = [np.mean(values) for values in data]
    ax.scatter(range(1, len(means) + 1), means, color="black", s=30, zorder=3)

    ax.set_title(f"{dataset_name.capitalize()} - Error Distribution ({metric})")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.3)

    output_path = os.path.join(
        output_dir, f"eval_{dataset_name}_error_distribution_{metric}.png"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_burgers_sample(
    ax: plt.Axes,
    sample_x: torch.Tensor,
    sample_y: torch.Tensor,
    pred: torch.Tensor,
    title: str,
    show_error: bool = True,
) -> None:
    x_grid = _resolve_x_grid(sample_x, sample_y)
    y_true = sample_y[0, :].cpu().numpy()
    y_pred = pred[0, :].cpu().numpy()

    ax.plot(x_grid, y_true, color="black", linewidth=1.6, label="Truth")
    ax.plot(
        x_grid, y_pred, color="tab:blue", linestyle="--", linewidth=1.4, label="Pred"
    )
    if show_error:
        ax.plot(
            x_grid,
            y_pred - y_true,
            color="tab:red",
            alpha=0.6,
            linewidth=1.0,
            label="Error",
        )
        ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.grid(True, alpha=0.3)


def _plot_darcy_sample(
    sample_y: torch.Tensor,
    pred: torch.Tensor,
    title: str,
    output_path: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    y_true = sample_y[0].cpu().numpy()
    y_pred = pred[0].cpu().numpy()

    im0 = axes[0].imshow(y_true, origin="lower", cmap="viridis")
    axes[0].set_title("Ground Truth")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(y_pred, origin="lower", cmap="viridis")
    axes[1].set_title("Prediction")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(np.abs(y_true - y_pred), origin="lower", cmap="plasma")
    axes[2].set_title("|Error|")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_best_worst_samples(
    model: torch.nn.Module,
    model_name: str,
    test_dataset,
    dataset_name: str,
    dim: int,
    device: torch.device,
    errors: np.ndarray,
    top_k: int,
    output_dir: str,
) -> list[str]:
    _ensure_output_dir(output_dir)
    paths: list[str] = []
    top_k = max(1, min(top_k, len(errors)))

    sorted_indices = np.argsort(errors)
    best_indices = sorted_indices[:top_k]
    worst_indices = sorted_indices[-top_k:][::-1]

    with torch.no_grad():
        for label, indices in ("best", best_indices), ("worst", worst_indices):
            for rank, idx in enumerate(indices, start=1):
                sample_x, sample_y = test_dataset[int(idx)]
                pred = model(sample_x.unsqueeze(0).to(device)).squeeze(0).cpu()

                filename = _sanitize_name(model_name)
                output_path = os.path.join(
                    output_dir,
                    f"eval_{dataset_name}_{dim}d_{filename}_{label}_{rank}_sample{idx}.png",
                )

                if dim == 1:
                    fig, ax = plt.subplots(figsize=(9, 4.5))
                    _plot_burgers_sample(
                        ax,
                        sample_x,
                        sample_y,
                        pred,
                        title=f"{model_name} - {label.capitalize()} {rank} (Sample {idx})",
                    )
                    ax.legend(loc="best", fontsize=8)
                    fig.tight_layout()
                    fig.savefig(output_path, dpi=150)
                    plt.close(fig)
                elif dim == 2:
                    _plot_darcy_sample(
                        sample_y,
                        pred,
                        title=f"{model_name} - {label.capitalize()} {rank} (Sample {idx})",
                        output_path=output_path,
                    )

                if os.path.exists(output_path):
                    paths.append(output_path)

    return paths


def compute_error_stats_1d(
    model: torch.nn.Module,
    test_dataset,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sum_error = None
    sumsq_error = None
    x_grid = None

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            sample_x, sample_y = test_dataset[idx]
            pred = model(sample_x.unsqueeze(0).to(device)).squeeze(0).cpu()
            error = (pred - sample_y).abs().squeeze()

            if sum_error is None:
                sum_error = error.clone()
                sumsq_error = error.pow(2)
                x_grid = _resolve_x_grid(sample_x, sample_y)
            else:
                sum_error += error
                sumsq_error += error.pow(2)

    if sum_error is None or sumsq_error is None or x_grid is None:
        raise ValueError("Test dataset is empty")

    mean_error = (sum_error / len(test_dataset)).numpy()
    var = (sumsq_error / len(test_dataset)).numpy() - mean_error**2
    std_error = np.sqrt(np.maximum(var, 0.0))
    return x_grid, mean_error, std_error


def compute_mean_abs_error_field(
    model: torch.nn.Module,
    test_dataset,
    device: torch.device,
) -> np.ndarray:
    sum_error = None

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            sample_x, sample_y = test_dataset[idx]
            pred = model(sample_x.unsqueeze(0).to(device)).squeeze(0).cpu()
            error = (pred - sample_y).abs().squeeze()

            if sum_error is None:
                sum_error = error.clone()
            else:
                sum_error += error

    if sum_error is None:
        raise ValueError("Test dataset is empty")

    mean_error = sum_error / len(test_dataset)
    return mean_error.numpy()


def plot_mean_error_profile(
    model: torch.nn.Module,
    model_name: str,
    test_dataset,
    dataset_name: str,
    device: torch.device,
    output_dir: str,
) -> str:
    _ensure_output_dir(output_dir)
    x_grid, mean_error, std_error = compute_error_stats_1d(model, test_dataset, device)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x_grid, mean_error, color="tab:blue", linewidth=1.6, label="Mean |Error|")
    ax.fill_between(
        x_grid,
        mean_error - std_error,
        mean_error + std_error,
        color="tab:blue",
        alpha=0.2,
        label="Std",
    )
    ax.set_title(f"{dataset_name.capitalize()} - Error Profile ({model_name})")
    ax.set_xlabel("x")
    ax.set_ylabel("|Error|")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    output_path = os.path.join(
        output_dir,
        f"eval_{dataset_name}_1d_error_profile_{_sanitize_name(model_name)}.png",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_mean_error_heatmap(
    model: torch.nn.Module,
    model_name: str,
    test_dataset,
    dataset_name: str,
    device: torch.device,
    output_dir: str,
) -> str:
    _ensure_output_dir(output_dir)
    mean_error = compute_mean_abs_error_field(model, test_dataset, device)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mean_error, origin="lower", cmap="plasma")
    ax.set_title(f"{dataset_name.capitalize()} - Mean |Error| ({model_name})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    output_path = os.path.join(
        output_dir,
        f"eval_{dataset_name}_2d_mean_error_{_sanitize_name(model_name)}.png",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_creative_burgers_waterfall(
    model: torch.nn.Module,
    model_name: str,
    test_dataset,
    dataset_name: str,
    device: torch.device,
    output_dir: str,
    n_samples: int = 8,
) -> str:
    _ensure_output_dir(output_dir)
    total_samples = len(test_dataset)
    if total_samples == 0:
        raise ValueError("Test dataset is empty")

    sample_indices = np.linspace(
        0, total_samples - 1, num=min(n_samples, total_samples)
    )
    sample_indices = [int(idx) for idx in sample_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    offset = 0.0

    with torch.no_grad():
        for idx in sample_indices:
            sample_x, sample_y = test_dataset[idx]
            pred = model(sample_x.unsqueeze(0).to(device)).squeeze(0).cpu()

            x_grid = _resolve_x_grid(sample_x, sample_y)
            y_true = sample_y[0, :].cpu().numpy()
            y_pred = pred[0, :].cpu().numpy()

            scale = np.max(np.abs(y_true))
            if scale > 0:
                offset += scale * 0.2

            ax.plot(x_grid, y_true + offset, color="black", linewidth=1.1, alpha=0.7)
            ax.plot(x_grid, y_pred + offset, color="tab:blue", linewidth=1.0, alpha=0.7)

    ax.set_title(f"{dataset_name.capitalize()} - Waterfall Overlay ({model_name})")
    ax.set_xlabel("x")
    ax.set_ylabel("Offset u(x)")
    ax.grid(True, alpha=0.2)

    output_path = os.path.join(
        output_dir,
        f"eval_{dataset_name}_1d_waterfall_{_sanitize_name(model_name)}.png",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_creative_darcy_contour_error(
    model: torch.nn.Module,
    model_name: str,
    test_dataset,
    dataset_name: str,
    device: torch.device,
    errors: np.ndarray,
    output_dir: str,
) -> str:
    _ensure_output_dir(output_dir)
    if len(errors) == 0:
        raise ValueError("No errors available for plotting")

    median_idx = int(np.argsort(errors)[len(errors) // 2])
    sample_x, sample_y = test_dataset[median_idx]

    with torch.no_grad():
        pred = model(sample_x.unsqueeze(0).to(device)).squeeze(0).cpu()

    y_true = sample_y[0].cpu().numpy()
    y_pred = pred[0].cpu().numpy()
    error = np.abs(y_true - y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(error, origin="lower", cmap="plasma")
    contours = ax.contour(y_true, colors="white", linewidths=0.6, alpha=0.6)
    ax.clabel(contours, inline=True, fontsize=7, fmt="%0.2f")
    ax.set_title(f"{dataset_name.capitalize()} - Error + Truth Contours ({model_name})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    output_path = os.path.join(
        output_dir,
        f"eval_{dataset_name}_2d_contour_error_{_sanitize_name(model_name)}.png",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def evaluate_models(
    models: list[tuple[str, torch.nn.Module]],
    test_loader,
    test_dataset,
    dataset_name: str,
    dim: int,
    device: torch.device,
    top_k: int = 2,
    metric: str = "rel_l2",
    output_dir: str = "plots",
):
    for name, model in models:
        results = evaluate_model(model, test_loader, device)
        print_results_table(name, results)

    print(f"Generating high-impact plots for {dataset_name.capitalize()}...")
    for _, model in models:
        model.eval()

    errors = compute_per_sample_errors(models, test_dataset, device, metric)
    plot_error_distributions(errors, dataset_name, metric, output_dir)

    for name, model in models:
        plot_best_worst_samples(
            model,
            name,
            test_dataset,
            dataset_name,
            dim,
            device,
            errors[name],
            top_k,
            output_dir,
        )

        if dim == 1:
            plot_mean_error_profile(
                model,
                name,
                test_dataset,
                dataset_name,
                device,
                output_dir,
            )
            plot_creative_burgers_waterfall(
                model,
                name,
                test_dataset,
                dataset_name,
                device,
                output_dir,
            )
        elif dim == 2:
            plot_mean_error_heatmap(
                model,
                name,
                test_dataset,
                dataset_name,
                device,
                output_dir,
            )
            plot_creative_darcy_contour_error(
                model,
                name,
                test_dataset,
                dataset_name,
                device,
                errors[name],
                output_dir,
            )

    print("High-impact plots saved in the 'plots' directory.")


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
    top_k: int = 2,
    metric: str = "rel_l2",
    output_dir: str = "plots",
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
        top_k=top_k,
        metric=metric,
        output_dir=output_dir,
    )


def generate_high_impact_plots(
    models: list[tuple[str, torch.nn.Module]],
    test_dataset,
    dataset_name: str,
    dim: int,
    device: torch.device,
    top_k: int = 2,
    metric: str = "rel_l2",
    output_dir: str = "plots",
) -> list[str]:
    """Generates the high-impact plot suite and returns saved paths."""
    for _, model in models:
        model.eval()

    paths: list[str] = []
    errors = compute_per_sample_errors(models, test_dataset, device, metric)

    paths.append(plot_error_distributions(errors, dataset_name, metric, output_dir))

    for name, model in models:
        paths.extend(
            plot_best_worst_samples(
                model,
                name,
                test_dataset,
                dataset_name,
                dim,
                device,
                errors[name],
                top_k,
                output_dir,
            )
        )

        if dim == 1:
            paths.append(
                plot_mean_error_profile(
                    model,
                    name,
                    test_dataset,
                    dataset_name,
                    device,
                    output_dir,
                )
            )
            paths.append(
                plot_creative_burgers_waterfall(
                    model,
                    name,
                    test_dataset,
                    dataset_name,
                    device,
                    output_dir,
                )
            )
        elif dim == 2:
            paths.append(
                plot_mean_error_heatmap(
                    model,
                    name,
                    test_dataset,
                    dataset_name,
                    device,
                    output_dir,
                )
            )
            paths.append(
                plot_creative_darcy_contour_error(
                    model,
                    name,
                    test_dataset,
                    dataset_name,
                    device,
                    errors[name],
                    output_dir,
                )
            )

    return [path for path in paths if path and os.path.exists(path)]
