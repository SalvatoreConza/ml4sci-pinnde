"""
plot.py — Visualization utilities for PINN experiments.

Author: Conza Salvatore Angelo

Provides functions to:
  - Compare PINN predictions against exact analytical solutions
  - Plot training loss curves (log scale)
  - Plot absolute error over the time domain
  - Print summary metric tables (MAE, Max Error, L2 Relative Error)

Not-so-trivial observation / premise:
    Since what I am plotting is basically a function that oscillates more or less
    towards zero and is very smooth, I can imagine that the model will get less
    precise at the points where the derivative changes its sign. This was the case
    for all experiments: the graphs plotting absolute error over time show a bigger
    absolute error at the points where the function changes the sign of its derivative.
    This also explains why a damped harmonic oscillator with bigger ξ gets predicted
    better from my PINN: there are fewer points where the function changes its
    derivative so it's easier to predict. In literature this is called "spectral bias"
    or "failure modes". A way of solving this is to sample fresh random points at each
    step — this helps to "fill" better the domain, though it adds some gradient noise.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from model import analytical_solution


def plot_all_predictions(results, test_xis=[0.1, 0.25, 0.4], save_path=None):
    """Plot PINN predictions vs. exact solution for multiple experiments.

    Testing with ξ = 0.1 (low damping), ξ = 0.25 (medium damping), ξ = 0.4 (high damping).
    """
    z_plot = torch.linspace(0, 20, 200).view(-1, 1)
    plt.figure(figsize=(12, 5))
    for i, xi_val in enumerate(test_xis):
        xi_plot = torch.full_like(z_plot, xi_val)
        # Exact solution
        x_exact = analytical_solution(z_plot.numpy(), xi_val)
        plt.subplot(1, 3, i + 1)
        plt.plot(z_plot, x_exact, 'k--', linewidth=2, label="Exact analytical", alpha=0.6)
        for name, data in results.items():
            # PINN prediction
            with torch.no_grad():
                pred = data["model"](z_plot, xi_plot).numpy()
            style = ':' if "ReLU" in name else '-'
            plt.plot(z_plot, pred, linestyle=style, linewidth=1.5, label=name)
        plt.title(f"Solution for $\\xi={xi_val}$")
        plt.xlabel("z")
        plt.ylabel("x")
        plt.grid(True, alpha=0.3)
        if i == 0:
            plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_all_losses(results, save_path=None):
    """Plot training loss curves (log scale) for multiple experiments.

    The error drops drastically at first and then very slowly, so log scale is best.
    """
    plt.figure(figsize=(8, 5))
    for name, data in results.items():
        plt.plot(data["hist"], label=name)
    plt.yscale('log')
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_all_errors(results, test_xis=[0.1, 0.25, 0.4], save_path=None):
    """Plot absolute error |exact − PINN| over the time domain for multiple experiments."""
    z_plot = torch.linspace(0, 20, 200).view(-1, 1)
    plt.figure(figsize=(12, 5))
    for i, xi_val in enumerate(test_xis):
        xi_plot = torch.full_like(z_plot, xi_val)
        x_exact = analytical_solution(z_plot.numpy(), xi_val)
        plt.subplot(1, 3, i + 1)
        for name, data in results.items():
            with torch.no_grad():
                pred = data["model"](z_plot, xi_plot).numpy()
            # Calculate absolute error
            error = np.abs(x_exact - pred)
            plt.plot(z_plot, error, label=name)
        plt.yscale('log')
        plt.title(f"Absolute Error ($\\xi={xi_val}$)")
        plt.xlabel("z")
        plt.ylabel("Absolute Error")
        plt.grid(True, which="both", ls='-', alpha=0.5)
        plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def summary_tables(results, test_xis=[0.1, 0.25, 0.4]):
    """Print summary metrics tables: MAE, Max Error, L2 Relative Error.

    Evaluated at 500 equally-spaced points over z ∈ [0, 20].
    """
    z_test = torch.linspace(0, 20, 500).view(-1, 1)
    for xi_val in test_xis:
        xi_test = torch.full_like(z_test, xi_val)
        exact = analytical_solution(z_test.numpy(), xi_val)
        rows = []
        for name, data in results.items():
            with torch.no_grad():
                pred = data["model"](z_test, xi_test).numpy()
            err = np.abs(exact - pred)
            rows.append({
                "Model": name,
                "MAE": np.mean(err),
                "Max Error": np.max(err),
                "L2 Rel Error": np.sqrt(np.sum((exact - pred) ** 2) / np.sum(exact ** 2)),
                "Final Loss": data["hist"][-1],
                "Steps": len(data["hist"])
            })
        df = pd.DataFrame(rows)
        print(f"\n{'=' * 70}")
        print(f"  ξ = {xi_val}")
        print(f"{'=' * 70}")
        print(df.to_string(index=False, float_format="%.6f"))
    print()
