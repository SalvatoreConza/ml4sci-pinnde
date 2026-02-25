#!/usr/bin/env python3
"""
run_experiments.py — Main entry point for running PINN experiments.

Author: Conza Salvatore Angelo

Usage:
    python run_experiments.py                  # run all experiment sets
    python run_experiments.py --set baseline   # run only baseline experiments
    python run_experiments.py --set hybrid     # run only hybrid Adam+L-BFGS experiments
    python run_experiments.py --set literature  # run literature architectures (Adam + hybrid)
    python run_experiments.py --save-plots     # save plots to images/ directory

All experiments use torch.manual_seed(42) and np.random.seed(42) for reproducibility.
"""

import argparse
import os

import torch
import numpy as np

from train import train_adam, train_hybrid
from plot import plot_all_predictions, plot_all_losses, plot_all_errors, summary_tables
from experiments import (
    BASELINE_EXPERIMENTS,
    BATCH_SIZE_EXPERIMENTS,
    PATIENCE_EXPERIMENTS,
    HYBRID_EXPERIMENTS,
    LITERATURE_ADAM_EXPERIMENTS,
    LITERATURE_HYBRID_EXPERIMENTS,
)


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_adam_experiments(experiments, label=""):
    """Run a set of Adam-only experiments and return results dict."""
    results = {}
    for exp in experiments:
        model, hist = train_adam(
            model_name=exp["name"],
            activation=exp["act"],
            lambda_ic=exp["lambda"],
            hidden_dim=exp["dim"],
            layers=exp["layers"],
            steps=exp["number of steps"],
            batch_size=exp["bs"],
            lr=exp["lr"],
            patience=exp["patience"],
        )
        results[exp["name"]] = {"model": model, "hist": hist}
    return results


def run_hybrid_experiments(experiments, label=""):
    """Run a set of hybrid Adam+L-BFGS experiments and return results dict."""
    results = {}
    for exp in experiments:
        model, hist = train_hybrid(
            model_name=exp["name"],
            activation=exp["act"],
            lambda_ic=exp["lambda"],
            hidden_dim=exp["dim"],
            layers=exp["layers"],
            adam_steps=exp["adam_steps"],
            adam_lr=exp["adam_lr"],
            batch_size=exp["bs"],
            lbfgs_max_iter=exp["lbfgs_max_iter"],
            lbfgs_batch_size=exp["lbfgs_bs"],
        )
        results[exp["name"]] = {"model": model, "hist": hist}
    return results


def visualize(results, set_name, save_plots=False, img_dir="images"):
    """Generate all plots and summary tables for a result set."""
    prefix = os.path.join(img_dir, set_name.replace(" ", "_").lower())
    plot_all_predictions(
        results,
        save_path=f"{prefix}_predictions.png" if save_plots else None,
    )
    plot_all_losses(
        results,
        save_path=f"{prefix}_loss.png" if save_plots else None,
    )
    plot_all_errors(
        results,
        save_path=f"{prefix}_errors.png" if save_plots else None,
    )
    summary_tables(results)


def main():
    parser = argparse.ArgumentParser(
        description="Run PINN experiments for the parametric damped harmonic oscillator."
    )
    parser.add_argument(
        "--set",
        choices=["baseline", "batch_size", "patience", "hybrid", "literature", "all"],
        default="all",
        help="Which experiment set to run (default: all).",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots as PNG files in the images/ directory.",
    )
    args = parser.parse_args()

    if args.save_plots:
        os.makedirs("images", exist_ok=True)

    sets_to_run = (
        ["baseline", "batch_size", "patience", "hybrid", "literature"]
        if args.set == "all"
        else [args.set]
    )

    for s in sets_to_run:
        set_seed(42)

        if s == "baseline":
            # ── Set 1: Baseline Architectures (patience=2000) ──
            # The deeper network stops earlier than the others, even though it doesn't
            # exhibit a plateau like the wider network. This motivated Sets 2 and 3.
            print("\n" + "=" * 70)
            print("  SET 1: Baseline Architectures")
            print("=" * 70)
            results = run_adam_experiments(BASELINE_EXPERIMENTS)
            visualize(results, "set1_baseline", args.save_plots)

        elif s == "batch_size":
            # ── Set 2: Increased Batch Size ──
            # We hoped larger batches would trigger patience later, for more steps.
            print("\n" + "=" * 70)
            print("  SET 2: Increased Batch Size")
            print("=" * 70)
            results = run_adam_experiments(BATCH_SIZE_EXPERIMENTS)
            visualize(results, "set2_batch_size", args.save_plots)

        elif s == "patience":
            # ── Set 3: Extended Patience ──
            # Dramatic improvements, especially for Standard Tanh and Deeper Network.
            print("\n" + "=" * 70)
            print("  SET 3: Extended Patience")
            print("=" * 70)
            results = run_adam_experiments(PATIENCE_EXPERIMENTS)
            visualize(results, "set3_patience", args.save_plots)

        elif s == "hybrid":
            # ── Hybrid Adam + L-BFGS ──
            # ~10× speedup compared to Adam alone.
            print("\n" + "=" * 70)
            print("  HYBRID: Adam + L-BFGS Optimization")
            print("=" * 70)
            results = run_hybrid_experiments(HYBRID_EXPERIMENTS)
            visualize(results, "hybrid", args.save_plots)

        elif s == "literature":
            # ── Literature Architectures ──
            # Adam-only
            print("\n" + "=" * 70)
            print("  LITERATURE: Published Architectures (Adam-only)")
            print("=" * 70)
            set_seed(42)
            results_adam = run_adam_experiments(LITERATURE_ADAM_EXPERIMENTS)
            visualize(results_adam, "literature_adam", args.save_plots)

            # Hybrid Adam + L-BFGS
            print("\n" + "=" * 70)
            print("  LITERATURE: Published Architectures (Adam + L-BFGS)")
            print("=" * 70)
            set_seed(42)
            results_hybrid = run_hybrid_experiments(LITERATURE_HYBRID_EXPERIMENTS)
            visualize(results_hybrid, "literature_hybrid", args.save_plots)


if __name__ == "__main__":
    main()
