"""
experiments.py — Experiment configurations for the PINN damped harmonic oscillator.

Author: Conza Salvatore Angelo

This file defines all experiment groups as dictionaries that can be passed
to the training functions.  Each group explores different architecture /
hyperparameter choices guided by the PINN survey of Cuomo et al. (2022).

Experiment groups
-----------------
SET 1 — Baseline Architectures (patience=2000):
    Four architectures compared under identical training conditions.
    Key finding: Standard Tanh (3×50) outperformed all variants at this stage.
    The deeper network stopped early due to patience limits.

SET 2 — Increased Batch Size (batch_size=5000):
    Hoping larger batches delay patience triggering and allow more steps.
    Observation: the loss became a lot more wobbly — sign that the batch size
    increase wasn't enough; should probably implement dropout, normalization,
    or decrease the learning rate because it's too aggressive.

SET 3 — Extended Patience (patience=5000):
    Allowing longer training yielded dramatic improvements.
    The Deeper Network achieved the best MAE of 0.007 with sufficient patience,
    confirming that depth helps model the oscillatory structure.
    From step 40000 the first model's loss barely decreased, but the deeper model
    was still decreasing — as expected since more activation functions can simulate
    more complex functions.
    The wider model stopped decreasing after 10000–20000 steps because the
    Universal Approximation Theorem states that a wider NN can approximate the
    function but only if it's infinitely wide.

HYBRID — Adam + L-BFGS (from He et al., 2020):
    Two-phase: 5000 Adam steps → L-BFGS fine-tuning on fixed collocation points.
    Achieves comparable accuracy in ~8K steps vs ~78K for Adam alone (~10× speedup).
    What we did before this experiment is not wasted: we can adapt the architectures
    found in the previous experiments to this mixed Adam + L-BFGS optimization.

LITERATURE — Architectures from the PINN literature:
    Kharazmi et al (2021): 4 layers × 20 neurons (hp-VPINN)
    Zhu et al (2021): 5 layers × 250 neurons (fully connected)
    He et al (2020): 3–5 layers × 32 neurons (examines effect of NN size on accuracy)
    > — S. Cuomo et al., Ch. 2.1, pp. 12

    Other possible models include CNNs, RNNs, or Bayesian NNs:
    > — S. Cuomo et al., Ch. 2.1.2, 2.1.3, 2.1.4, pp. 14–18
"""

import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────
# SET 1: Baseline Architectures
# ─────────────────────────────────────────────────────────────────────
BASELINE_EXPERIMENTS = [
    {"name": "Standard Tanh",    "act": nn.Tanh, "lambda": 1.0,  "dim": 50,  "layers": 3, "number of steps": 100000, "bs": 1000, "lr": 1e-3, "patience": 2000},
    {"name": "Deeper Network",   "act": nn.Tanh, "lambda": 1.0,  "dim": 50,  "layers": 6, "number of steps": 100000, "bs": 1000, "lr": 1e-3, "patience": 2000},
    {"name": "Wider Network",    "act": nn.Tanh, "lambda": 1.0,  "dim": 200, "layers": 1, "number of steps": 100000, "bs": 1000, "lr": 1e-3, "patience": 2000},
    {"name": "Weighted loss ICs", "act": nn.Tanh, "lambda": 10.0, "dim": 50,  "layers": 3, "number of steps": 100000, "bs": 1000, "lr": 1e-3, "patience": 2000},
]

# ─────────────────────────────────────────────────────────────────────
# SET 2: Increased Batch Size
# ─────────────────────────────────────────────────────────────────────
BATCH_SIZE_EXPERIMENTS = [
    {"name": "Standard Tanh",    "act": nn.Tanh, "lambda": 1.0,  "dim": 50,  "layers": 3, "number of steps": 100000, "bs": 5000, "lr": 1e-3, "patience": 2000},
    {"name": "Deeper Network",   "act": nn.Tanh, "lambda": 1.0,  "dim": 50,  "layers": 6, "number of steps": 100000, "bs": 5000, "lr": 1e-3, "patience": 2000},
    {"name": "Wider Network",    "act": nn.Tanh, "lambda": 1.0,  "dim": 200, "layers": 1, "number of steps": 100000, "bs": 5000, "lr": 1e-3, "patience": 2000},
    {"name": "Weighted loss ICs", "act": nn.Tanh, "lambda": 10.0, "dim": 50,  "layers": 3, "number of steps": 100000, "bs": 5000, "lr": 1e-3, "patience": 2000},
]

# ─────────────────────────────────────────────────────────────────────
# SET 3: Extended Patience
# ─────────────────────────────────────────────────────────────────────
PATIENCE_EXPERIMENTS = [
    {"name": "Standard Tanh",    "act": nn.Tanh, "lambda": 1.0,  "dim": 50,  "layers": 3, "number of steps": 100000, "bs": 1000, "lr": 1e-3, "patience": 5000},
    {"name": "Deeper Network",   "act": nn.Tanh, "lambda": 1.0,  "dim": 50,  "layers": 6, "number of steps": 100000, "bs": 1000, "lr": 1e-3, "patience": 5000},
    {"name": "Wider Network",    "act": nn.Tanh, "lambda": 1.0,  "dim": 200, "layers": 1, "number of steps": 100000, "bs": 1000, "lr": 1e-3, "patience": 5000},
    {"name": "Weighted loss ICs", "act": nn.Tanh, "lambda": 10.0, "dim": 50,  "layers": 3, "number of steps": 100000, "bs": 1000, "lr": 1e-3, "patience": 5000},
]

# ─────────────────────────────────────────────────────────────────────
# HYBRID: Adam + L-BFGS (baseline architectures)
# ─────────────────────────────────────────────────────────────────────
HYBRID_EXPERIMENTS = [
    {"name": "Standard Tanh (Adam+L-BFGS)",    "act": nn.Tanh, "lambda": 1.0,  "dim": 50,  "layers": 3, "adam_steps": 5000, "adam_lr": 1e-3, "bs": 1000, "lbfgs_max_iter": 5000, "lbfgs_bs": 2000},
    {"name": "Deeper Network (Adam+L-BFGS)",   "act": nn.Tanh, "lambda": 1.0,  "dim": 50,  "layers": 6, "adam_steps": 5000, "adam_lr": 1e-3, "bs": 1000, "lbfgs_max_iter": 5000, "lbfgs_bs": 2000},
    {"name": "Wider Network (Adam+L-BFGS)",    "act": nn.Tanh, "lambda": 1.0,  "dim": 200, "layers": 1, "adam_steps": 5000, "adam_lr": 1e-3, "bs": 1000, "lbfgs_max_iter": 5000, "lbfgs_bs": 2000},
    {"name": "Weighted loss ICs (Adam+L-BFGS)", "act": nn.Tanh, "lambda": 10.0, "dim": 50,  "layers": 3, "adam_steps": 5000, "adam_lr": 1e-3, "bs": 1000, "lbfgs_max_iter": 5000, "lbfgs_bs": 2000},
]

# ─────────────────────────────────────────────────────────────────────
# LITERATURE: Architectures from published PINN papers (Adam-only)
# ─────────────────────────────────────────────────────────────────────
LITERATURE_ADAM_EXPERIMENTS = [
    {"name": "Kharazmi Deep Network",  "act": nn.Tanh, "lambda": 1.0, "dim": 20,  "layers": 4, "number of steps": 100000, "bs": 1000, "lr": 1e-3, "patience": 5000},
    {"name": "Zhu Deeper Network",     "act": nn.Tanh, "lambda": 1.0, "dim": 250, "layers": 5, "number of steps": 100000, "bs": 1000, "lr": 1e-3, "patience": 5000},
    {"name": "He 3 layer, 32 neurons", "act": nn.Tanh, "lambda": 1.0, "dim": 32,  "layers": 3, "number of steps": 100000, "bs": 1000, "lr": 1e-3, "patience": 5000},
    {"name": "He 4 layer, 32 neurons", "act": nn.Tanh, "lambda": 1.0, "dim": 32,  "layers": 4, "number of steps": 100000, "bs": 1000, "lr": 1e-3, "patience": 5000},
    {"name": "He 5 layer, 32 neurons", "act": nn.Tanh, "lambda": 1.0, "dim": 32,  "layers": 5, "number of steps": 100000, "bs": 1000, "lr": 1e-3, "patience": 5000},
]

# ─────────────────────────────────────────────────────────────────────
# LITERATURE: Same architectures with hybrid Adam + L-BFGS
# ─────────────────────────────────────────────────────────────────────
LITERATURE_HYBRID_EXPERIMENTS = [
    {"name": "Kharazmi Deep Network",  "act": nn.Tanh, "lambda": 1.0, "dim": 20,  "layers": 4, "adam_steps": 5000, "adam_lr": 1e-3, "bs": 1000, "lbfgs_max_iter": 5000, "lbfgs_bs": 2000},
    {"name": "Zhu Deeper Network",     "act": nn.Tanh, "lambda": 1.0, "dim": 250, "layers": 5, "adam_steps": 5000, "adam_lr": 1e-3, "bs": 1000, "lbfgs_max_iter": 5000, "lbfgs_bs": 2000},
    {"name": "He 3 layer, 32 neurons", "act": nn.Tanh, "lambda": 1.0, "dim": 32,  "layers": 3, "adam_steps": 5000, "adam_lr": 1e-3, "bs": 1000, "lbfgs_max_iter": 5000, "lbfgs_bs": 2000},
    {"name": "He 4 layer, 32 neurons", "act": nn.Tanh, "lambda": 1.0, "dim": 32,  "layers": 4, "adam_steps": 5000, "adam_lr": 1e-3, "bs": 1000, "lbfgs_max_iter": 5000, "lbfgs_bs": 2000},
    {"name": "He 5 layer, 32 neurons", "act": nn.Tanh, "lambda": 1.0, "dim": 32,  "layers": 5, "adam_steps": 5000, "adam_lr": 1e-3, "bs": 1000, "lbfgs_max_iter": 5000, "lbfgs_bs": 2000},
]
