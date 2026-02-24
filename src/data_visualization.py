"""
Visualization utilities for PDE datasets:
  - Darcy Flow (2D)
  - Burgers 1D (input → output mapping)
  - Navier-Stokes (2D vorticity)

Each dataset returns batches with keys "x" (input) and "y" (target/solution).
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from src.generate_data import (
    generate_darcy_data,
    generate_burgers_data,
    generate_navier_stokes_data,
)


def visualize_navier_stokes_samples(data, n_samples=3, cmap="RdBu_r"):
    """
    Visualise samples from a Navier-Stokes dataset.

    Each sample shows:
      - Left:  input vorticity  ω(x, y) at initial time
      - Right: target vorticity ω(x, y) at final time
    """
    a, u = data["a"], data["u"]
    n_samples = min(n_samples, a.shape[0])

    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_samples):
        # a is [B, H, W], u is [B, H, W, T]
        w_in = a[i].squeeze()
        w_out = u[i, ..., -1].squeeze()  # Select final time step

        vmax_in = max(abs(w_in.min()), abs(w_in.max()))
        vmax_out = max(abs(w_out.min()), abs(w_out.max()))

        im0 = axes[i, 0].imshow(
            w_in,
            cmap=cmap,
            origin="lower",
            norm=mcolors.TwoSlopeNorm(vmin=-vmax_in, vcenter=0, vmax=vmax_in),
        )
        axes[i, 0].set_title(f"Sample {i} — Input vorticity $\\omega_{{in}}$")
        axes[i, 0].set_xlabel("x")
        axes[i, 0].set_ylabel("y")
        fig.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)

        im1 = axes[i, 1].imshow(
            w_out,
            cmap=cmap,
            origin="lower",
            norm=mcolors.TwoSlopeNorm(vmin=-vmax_out, vcenter=0, vmax=vmax_out),
        )
        axes[i, 1].set_title(f"Sample {i} — Target vorticity $\\omega_{{out}}$")
        axes[i, 1].set_xlabel("x")
        axes[i, 1].set_ylabel("y")
        fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

    fig.suptitle("Navier-Stokes — Input & Target Vorticity", fontsize=14, y=1.01)
    fig.tight_layout()
    plt.show()


def visualize_navier_stokes_difference(data, n_samples=3, cmap="coolwarm"):
    """
    Show the difference ω_out − ω_in for each sample.
    """
    a, u = data["a"], data["u"]
    n_samples = min(n_samples, a.shape[0])

    fig, axes = plt.subplots(1, n_samples, figsize=(5 * n_samples, 4))
    if n_samples == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        w_in = a[i].squeeze()
        w_out = u[i, ..., -1].squeeze()

        diff = w_out - w_in
        vmax = max(abs(diff.min()), abs(diff.max()))

        im = ax.imshow(
            diff,
            cmap=cmap,
            origin="lower",
            norm=mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
        )
        ax.set_title(f"Sample {i} — $\\Delta\\omega$")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Navier-Stokes — Vorticity Difference ($\\omega_{{out}} - \\omega_{{in}}$)",
        fontsize=14,
    )
    fig.tight_layout()
    plt.show()


def visualize_navier_stokes_statistics(data, cmap="inferno"):
    """
    Per-pixel mean and std of input and target (final) vorticity fields.
    """
    a, u = data["a"], data["u"]

    # Extract final timestep for u, preserve batch dimension
    w_in = a.squeeze()
    w_out = u[..., -1].squeeze()

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    titles = [
        ("Mean $\\omega_{{in}}$", w_in.mean(axis=0)),
        ("Std  $\\omega_{{in}}$", w_in.std(axis=0)),
        ("Mean $\\omega_{{out}}$", w_out.mean(axis=0)),
        ("Std  $\\omega_{{out}}$", w_out.std(axis=0)),
    ]

    for ax, (title, field) in zip(axes.flat, titles):
        im = ax.imshow(field, cmap=cmap, origin="lower")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Navier-Stokes — Batch Statistics", fontsize=14)
    fig.tight_layout()
    plt.show()


def visualize_burgers_samples(data, n_samples=3, cmap="RdBu_r"):
    """
    Visualise samples from a Burgers-1D dataset.

    Each sample shows:
      - Left:  input function  a(x)  (initial condition drawn from GRF)
      - Right: output function u(x)  (solution at t = t_end)
    """
    a, u = data["a"], data["u"]
    n_samples = min(n_samples, a.shape[0])

    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3.5 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_samples):
        ic = a[i].squeeze()  # [S]
        sol = u[i].squeeze()  # [S]
        x = np.linspace(0, 1, len(ic))

        axes[i, 0].plot(x, ic, color="steelblue", linewidth=1.5)
        axes[i, 0].set_title(f"Sample {i} — Input $a(x)$")
        axes[i, 0].set_xlabel("$x$")
        axes[i, 0].set_ylabel("$a$")
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].plot(x, sol, color="firebrick", linewidth=1.5)
        axes[i, 1].set_title(f"Sample {i} — Output $u(x)$")
        axes[i, 1].set_xlabel("$x$")
        axes[i, 1].set_ylabel("$u$")
        axes[i, 1].grid(True, alpha=0.3)

    fig.suptitle("Burgers 1-D — Input & Output Functions", fontsize=14, y=1.01)
    fig.tight_layout()
    plt.show()


def visualize_burgers_comparison(data, n_samples=3):
    """
    Overlay input (initial condition) and output (final solution) for several
    samples so we can see how the operator maps a -> u.
    """
    a, u = data["a"], data["u"]
    n_samples = min(n_samples, a.shape[0])

    fig, axes = plt.subplots(1, n_samples, figsize=(5 * n_samples, 4), sharey=True)
    if n_samples == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ic = a[i].squeeze()
        sol = u[i].squeeze()
        x = np.linspace(0, 1, len(ic))

        ax.plot(x, ic, color="steelblue", linewidth=1.4, label="Input $a(x)$")
        ax.plot(x, sol, color="firebrick", linewidth=1.4, label="Output $u(x)$")
        ax.set_title(f"Sample {i}")
        ax.set_xlabel("$x$")
        if i == 0:
            ax.set_ylabel("$u$")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Burgers 1-D — Input / Output Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    plt.show()


def visualize_burgers_statistics(data):
    """
    Show per-spatial-point mean and ±1 std band for the input and output
    fields across the whole batch.
    """
    a, u = data["a"], data["u"]
    x = np.linspace(0, 1, a.shape[1])

    a_mean, a_std = a.mean(axis=0), a.std(axis=0)
    u_mean, u_std = u.mean(axis=0), u.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(x, a_mean, color="steelblue", linewidth=1.5, label="Mean")
    axes[0].fill_between(
        x,
        a_mean - a_std,
        a_mean + a_std,
        color="steelblue",
        alpha=0.25,
        label="$\\pm 1\\sigma$",
    )
    axes[0].set_title("Input $a(x)$ — Batch Statistics")
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$a$")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, u_mean, color="firebrick", linewidth=1.5, label="Mean")
    axes[1].fill_between(
        x,
        u_mean - u_std,
        u_mean + u_std,
        color="firebrick",
        alpha=0.25,
        label="$\\pm 1\\sigma$",
    )
    axes[1].set_title("Output $u(x)$ — Batch Statistics")
    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel("$u$")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Burgers 1-D — Batch Statistics", fontsize=14)
    fig.tight_layout()
    plt.show()


def visualize_darcy_samples(data, n_samples=3, cmap="viridis"):
    """
    Visualise random samples from a Darcy Flow dataset.

    Each sample shows:
      - Left:  coefficient / permeability field  a(x, y)
      - Right: pressure / solution field           u(x, y)
    """
    a, u = data["a"], data["u"]
    n_samples = min(n_samples, a.shape[0])

    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_samples):
        a_field = a[i].squeeze()
        u_field = u[i].squeeze()

        im0 = axes[i, 0].imshow(a_field, cmap=cmap, origin="lower")
        axes[i, 0].set_title(f"Sample {i} — Coefficient $a(x,y)$")
        axes[i, 0].set_xlabel("x")
        axes[i, 0].set_ylabel("y")
        fig.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)

        im1 = axes[i, 1].imshow(u_field, cmap=cmap, origin="lower")
        axes[i, 1].set_title(f"Sample {i} — Solution $u(x,y)$")
        axes[i, 1].set_xlabel("x")
        axes[i, 1].set_ylabel("y")
        fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

    fig.suptitle("Darcy Flow — Coefficient & Solution Fields", fontsize=14, y=1.01)
    fig.tight_layout()
    plt.show()


def visualize_darcy_statistics(data, cmap="inferno"):
    """
    Show per-pixel mean and std of the coefficient and solution fields.
    """
    a, u = data["a"], data["u"]

    a_field = a.squeeze()
    u_field = u.squeeze()

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    titles = [
        ("Mean $a(x,y)$", a_field.mean(axis=0)),
        ("Std  $a(x,y)$", a_field.std(axis=0)),
        ("Mean $u(x,y)$", u_field.mean(axis=0)),
        ("Std  $u(x,y)$", u_field.std(axis=0)),
    ]

    for ax, (title, field) in zip(axes.flat, titles):
        im = ax.imshow(field, cmap=cmap, origin="lower")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Darcy Flow — Batch Statistics", fontsize=14)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main — generate data and visualise
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---- Darcy Flow -------------------------------------------------------
    print("Loading Darcy Flow data …")
    darcy = generate_darcy_data(
        n_samples=100,
    )
    visualize_darcy_samples(darcy, n_samples=5)
    visualize_darcy_statistics(darcy)

    # ---- Burgers 1D -------------------------------------------------------
    print("Loading Burgers 1-D data …")
    burgers = generate_burgers_data(
        n_samples=100,
    )
    visualize_burgers_samples(burgers, n_samples=5)
    visualize_burgers_comparison(burgers, n_samples=5)
    visualize_burgers_statistics(burgers)

    # # ---- Navier-Stokes ----------------------------------------------------
    # print("Loading Navier-Stokes data …")
    # ns = generate_navier_stokes_data(
    #     n_samples=100,
    # )
    # visualize_navier_stokes_samples(ns, n_samples=3)
    # visualize_navier_stokes_difference(ns, n_samples=3)
    # visualize_navier_stokes_statistics(ns)
