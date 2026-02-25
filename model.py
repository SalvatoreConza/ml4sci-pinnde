"""
model.py — PINN architecture, physics-informed loss, and analytical solution.

Author: Conza Salvatore Angelo

Architecture:
    Feedforward fully connected network (Input: 2, Hidden: configurable, Output: 1).
    From a literature review this seemed a good start:
    > Since Raissi et al (2019) original vanilla PINN, the majority of solutions
    > have used feed-forward neural networks.
    > — S. Cuomo et al., Scientific Machine Learning through PINNs, Ch. 2.1, p. 11

Activation:
    Tanh — infinitely differentiable; most widely used in PINNs.
    > Most authors tend to use the infinitely differentiable hyperbolic tangent
    > activation function α(x) = tanh(x).
    > — S. Cuomo et al., Ch. 2.1.1, p. 13
    NOTE: ReLU completely fails for second-order ODEs because d²(ReLU)/dz² = 0
    everywhere. This causes the physics loss term d²x/dz² to vanish, preventing
    the model from learning the acceleration.

Input Normalization:
    Both z and ξ are mapped to [−1, 1] to prevent Tanh saturation in extreme
    input ranges.
"""

import torch
import torch.nn as nn
import numpy as np


class BasePINN(nn.Module):
    """Configurable feedforward PINN for the damped harmonic oscillator.

    Parameters
    ----------
    activation : nn.Module class
        Activation function (default: nn.Tanh). ReLU is unsuitable — see module docstring.
    hidden_dim : int
        Number of neurons per hidden layer.
    hidden_layers : int
        Number of hidden layers.

    Architecture reference:
        Tartakovsky et al (2020) empirically determined three hidden layers and
        50 units per layer with hyperbolic tangent activation.
        — S. Cuomo et al., Ch. 2.1.1, pp. 11–12
    """

    def __init__(self, activation=nn.Tanh, hidden_dim=50, hidden_layers=3):
        super().__init__()
        layers = []  # I initialize the number of layers as an empty list, I will add more layers with .append

        # Input layer: linear layer + activation function
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(activation())

        # A configurable number of hidden layers, each a linear layer + activation
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())

        # Output layer (hidden -> 1)
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, z, xi):
        # Normalization step: map z ∈ [0, 20] and ξ ∈ [0.1, 0.4] to [−1, 1]
        z_norm = 2.0 * (z - 0.0) / (20.0 - 0.0) - 1.0
        xi_norm = 2.0 * (xi - 0.1) / (0.4 - 0.1) - 1.0
        # I feed the normalized values into the layers
        inputs = torch.cat([z_norm, xi_norm], dim=1)
        return self.net(inputs)


def physics_loss(model, z, xi):
    """Compute the physics-informed residual loss for the damped harmonic oscillator.

    The ODE is: d²x/dz² + 2ξ dx/dz + x = 0

    Loss function:
        Physics Residual + Initial Conditions for velocity and position.
        This method of simply adding those losses is standard soft-BC approach
        with tunable λ weighting.
        — S. Cuomo et al., Ch. 2.3
    """
    # Enable gradient tracking for z
    z.requires_grad = True

    # Predict x
    x = model(z, xi)

    # Compute first derivative (dx/dz)
    dx_dz = torch.autograd.grad(
        x, z,
        torch.ones_like(x),
        create_graph=True
    )[0]

    # Compute second derivative (d²x/dz²)
    d2x_dz2 = torch.autograd.grad(
        dx_dz, z,
        torch.ones_like(dx_dz),
        create_graph=True
    )[0]

    # This is just the left hand side of the ODE; since it equals 0, the residual
    # tells me how much off the prediction x is from satisfying the equation.
    residual = d2x_dz2 + 2 * xi * dx_dz + x

    # The mean of one positive and one negative value of the same magnitude is 0,
    # so it's required to square to account for them as penalty.
    return torch.mean(residual ** 2)


def analytical_solution(z, xi, x0=0.7, v0=1.2):
    """Compute the exact analytical solution for the underdamped harmonic oscillator.

    From Wikipedia (Harmonic oscillator):
        x(z) = exp(-ξ ω_n z) * (A cos(ω_d z) + B sin(ω_d z))

    Parameters
    ----------
    z : array-like
        Time domain values.
    xi : float
        Damping ratio (must satisfy ξ < 1 for underdamped case).
    x0, v0 : float
        Initial conditions: x(0) = x0, dx/dz(0) = v0.
    """
    omega_n = 1.0  # Natural frequency — this is just the coeff of x in the ODE so it's 1
    omega_d = np.sqrt(omega_n ** 2 - xi ** 2)  # Damped natural frequency (when omega_n=1)

    # Integration constants A and B found from ICs:
    # x(0) = A = x0
    # x'(0) = -ξ ω_n A + ω_d B = v0
    A = x0
    B = (v0 + xi * omega_n * x0) / omega_d

    return np.exp(-xi * omega_n * z) * (A * np.cos(omega_d * z) + B * np.sin(omega_d * z))
