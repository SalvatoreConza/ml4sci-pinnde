"""
train.py — Training loops for the PINN damped harmonic oscillator.

Author: Conza Salvatore Angelo

Two training strategies are provided:

1. Adam-only:
    Standard Adam optimizer with early stopping based on a windowed average loss.
    Fresh random collocation points are sampled at each step — this mitigates
    spectral bias by better filling the domain over time, but adds gradient noise.
    > — S. Cuomo et al., Ch. 2.3

2. Hybrid Adam + L-BFGS:
    He et al (2020) propose a two-step training approach in which the loss function
    is minimized first by the Adam algorithm with a predefined stop condition, then
    by the L-BFGS-B optimizer. According to the paper, for cases with a small amount
    of training data and/or residual points, L-BFGS-B performs better with a faster
    rate of convergence and reduced computing cost.
    > — S. Cuomo et al., Ch. 2.3, p. 24

    In practice, the hybrid approach achieves comparable accuracy in ~8K steps
    versus ~78K steps for Adam alone — a roughly 10× speedup.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from model import BasePINN, physics_loss


def train_adam(model_name, activation, lambda_ic=1.0, hidden_dim=50, layers=3,
               steps=100000, batch_size=1000, lr=1e-3, patience=2000, min_delta=1e-6):
    """Train a PINN using Adam optimizer with early stopping.

    Parameters
    ----------
    model_name : str
        Human-readable name for logging.
    activation : nn.Module class
        Activation function class (e.g., nn.Tanh).
    lambda_ic : float
        Weight for initial condition loss terms. The default (1.0) treats physics
        and IC losses equally. Higher values (e.g. 10.0) enforce ICs more strongly.
        Observation: with λ=10 the IC loss drops too fast in the first 1000 steps,
        suggesting either a larger λ or different loss balancing methods are needed.
    hidden_dim : int
        Number of neurons per hidden layer.
    layers : int
        Number of hidden layers.
    steps : int
        Maximum number of training steps.
    batch_size : int
        Number of random collocation points per step.
    lr : float
        Learning rate for Adam.
    patience : int
        Early stopping patience (on windowed average loss).
    min_delta : float
        Minimum improvement threshold for early stopping.

    Returns
    -------
    model : BasePINN
        Trained model.
    loss_history : list[float]
        Per-step loss values.
    """
    print(f"--- Starting Experiment: {model_name} (loss lambda_ICs={lambda_ic}) ---")

    # Initialize the model
    model = BasePINN(activation=activation, hidden_dim=hidden_dim, hidden_layers=layers)

    # Chosen optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initial conditions targets (specified in the test on the ML4SC website)
    x0_target = 0.7
    v0_target = 1.2

    # Windows for the average loss calculation
    window = 100
    loss_window = []

    loss_history = []
    best_loss = float('inf')
    patience_counter = 0

    for step in range(steps):
        optimizer.zero_grad()

        # Sample random points for z and xi in given domains at each step.
        # This helps us to cover the domain well over time (mitigates spectral bias).
        z_col = torch.rand(batch_size, 1) * 20.0
        xi_col = torch.rand(batch_size, 1) * (0.4 - 0.1) + 0.1
        # At each of these random points we calculate the loss/physics violation
        loss_physics = physics_loss(model, z_col, xi_col)

        # Initial condition loss (boundary loss)
        # Sample random xi for the boundary condition
        xi_bc = torch.rand(batch_size // 4, 1) * (0.4 - 0.1) + 0.1  # we pick a random xi again; no matter what xi, the oscillator always starts in the same spot
        z_bc = torch.zeros_like(xi_bc)  # z is always 0 here
        z_bc.requires_grad = True  # we still need to track z to compute dx/dz

        # Guess of the position at t=0:
        x_bc = model(z_bc, xi_bc)

        # Then compute the velocity at t=0:
        dx_bc = torch.autograd.grad(
            x_bc, z_bc,
            torch.ones_like(x_bc),
            create_graph=True
        )[0]

        # Then calculate the penalty for position and velocity:
        loss_ic_val = torch.mean((x_bc - x0_target) ** 2) + torch.mean((dx_bc - v0_target) ** 2)

        # Total loss: without the penalty for position and velocity,
        # x=0 and v=0 for every time step would minimize the loss (trivial solution),
        # but that cannot be since ICs are different from zero.
        loss = loss_physics + (lambda_ic * loss_ic_val)

        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        loss_history.append(current_loss)

        # Early stopping: if the average loss doesn't improve at least by min_delta
        # for a number of consecutive steps equal to patience, it stops early.
        loss_window.append(current_loss)
        if len(loss_window) > window:
            loss_window.pop(0)
        avg_loss = sum(loss_window) / len(loss_window)

        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at step {step}")
            print(f"Best loss: {best_loss:.6f}, Current loss: {current_loss:.6f}")
            break

        if step % 1000 == 0:
            print(f"Step {step}: Loss = {current_loss:.6f}, "
                  f"Avg Loss (last {window}) = {avg_loss:.6f} "
                  f"(Best: {best_loss:.6f}, Patience: {patience_counter}/{patience})")

    return model, loss_history


def train_hybrid(model_name, activation, lambda_ic=1.0, hidden_dim=50, layers=3,
                 adam_steps=5000, adam_lr=1e-3, batch_size=1000,
                 lbfgs_max_iter=5000, lbfgs_batch_size=2000,
                 patience=2000, min_delta=1e-6):
    """Train a PINN using hybrid Adam + L-BFGS optimization.

    Phase 1 (Adam): Random collocation points, early stopping.
    Phase 2 (L-BFGS): Fixed collocation points for stable second-order optimization.

    The L-BFGS sample is NOT in a loop — otherwise L-BFGS would get confused
    by the changing distribution of points when we change steps.

    Returns
    -------
    model : BasePINN
        Trained model.
    loss_history : list[float]
        Per-step loss values (Adam + L-BFGS combined).
    """
    print(f"\n{'=' * 70}")
    print(f"  Hybrid Adam + L-BFGS Experiment: {model_name}")
    print(f"{'=' * 70}")

    model = BasePINN(activation=activation, hidden_dim=hidden_dim, hidden_layers=layers)

    # Initial conditions targets (specified in the test on the ML4SC website)
    x0_target = 0.7
    v0_target = 1.2

    # Windows for the average loss calculation
    window = 100
    loss_window = []

    loss_history = []
    best_loss = float('inf')
    patience_counter = 0

    # ── Phase 1: Adam ──
    optimizer_adam = optim.Adam(model.parameters(), lr=adam_lr)

    print("Start training (with Adam)...")

    for step in range(adam_steps):
        optimizer_adam.zero_grad()

        # Sample random points for z and xi in given domains
        z_col = torch.rand(batch_size, 1) * 20.0
        xi_col = torch.rand(batch_size, 1) * (0.4 - 0.1) + 0.1
        # At each of these random points we calculate the loss/physics violation
        loss_physics = physics_loss(model, z_col, xi_col)

        # Initial condition loss (boundary loss)
        xi_bc = torch.rand(batch_size // 4, 1) * (0.4 - 0.1) + 0.1  # random xi; oscillator always starts in the same spot
        z_bc = torch.zeros_like(xi_bc)  # z is always 0 here
        z_bc.requires_grad = True  # we still need to track z to compute dx/dz

        # Guess of the position at t=0:
        x_bc = model(z_bc, xi_bc)

        # Then compute the velocity at t=0:
        dx_bc = torch.autograd.grad(
            x_bc, z_bc,
            torch.ones_like(x_bc),
            create_graph=True
        )[0]

        # Then calculate the penalty for position and velocity:
        loss_ic_val = torch.mean((x_bc - x0_target) ** 2) + torch.mean((dx_bc - v0_target) ** 2)

        # Total loss
        loss = loss_physics + (lambda_ic * loss_ic_val)

        loss.backward()
        optimizer_adam.step()

        current_loss = loss.item()
        loss_history.append(current_loss)

        # Early stopping
        loss_window.append(current_loss)
        if len(loss_window) > window:
            loss_window.pop(0)
        avg_loss = sum(loss_window) / len(loss_window)

        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at step {step}")
            print(f"Best avg loss: {best_loss:.6f}, Current avg loss: {avg_loss:.6f}")
            break

        if step % 1000 == 0:
            print(f"Step {step}: Loss = {current_loss:.6f}, "
                  f"Avg Loss (last {window}) = {avg_loss:.6f} "
                  f"(Best: {best_loss:.6f}, Patience: {patience_counter}/{patience})")

    # ── Phase 2: L-BFGS fine-tuning ──
    # After optimizing with Adam, do a fine tuning with L-BFGS on FIXED points.
    print("Continue training (with L-BFGS fine-tuning)...")

    optimizer_lbfgs = optim.LBFGS(
        model.parameters(),
        lr=1.0,
        history_size=50,
        max_iter=lbfgs_max_iter,
        line_search_fn="strong_wolfe"
    )

    # Fixed sample — NOT in a loop, otherwise L-BFGS would get confused by
    # the changing distribution of points when we change steps.
    z_col_fixed = torch.rand(lbfgs_batch_size, 1) * 20.0
    xi_col_fixed = torch.rand(lbfgs_batch_size, 1) * (0.4 - 0.1) + 0.1

    xi_bc_fixed = torch.rand(lbfgs_batch_size // 4, 1) * (0.4 - 0.1) + 0.1
    z_bc_fixed = torch.zeros_like(xi_bc_fixed)
    z_bc_fixed.requires_grad = True

    def closure():
        optimizer_lbfgs.zero_grad()

        # At each of these random points (NOW FIXED) we calculate the loss/physics violation
        l_physics = physics_loss(model, z_col_fixed, xi_col_fixed)

        # Guess of the position at t=0:
        x_bc = model(z_bc_fixed, xi_bc_fixed)
        # Then compute velocity at t=0:
        dx_bc = torch.autograd.grad(x_bc, z_bc_fixed, torch.ones_like(x_bc), create_graph=True)[0]

        # Then calculate the penalty for position and velocity:
        loss_ic_x = torch.mean((x_bc - x0_target) ** 2)
        loss_ic_v = torch.mean((dx_bc - v0_target) ** 2)

        # Total loss
        total_loss = l_physics + loss_ic_x + loss_ic_v

        total_loss.backward()

        # Append to history for plotting
        loss_history.append(total_loss.item())
        return total_loss

    optimizer_lbfgs.step(closure)
    print("Training complete.")
    return model, loss_history
