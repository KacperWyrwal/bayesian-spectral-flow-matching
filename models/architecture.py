"""This file contains different model architectures
The structure of the file and the fomder is temporary and used for dummy tests rights now 
we can split into other files later
"""

import torch
import Flows.config as config
from models.FNO import FNO1dVelocityField, FNO2dVelocityField
from models.manifold_fno import (
    GraphLaplacianFNOVelocityField,
    GraphSobolevFNOVelocityField,
    LaplacianFNOVelocityField,
    SobolevFNOVelocityField,
)


def light_fno2d_config(n_theta, n_phi, **overrides):
    """Small FNO2d preset for faster FM experiments on sphere grids."""
    n_grid = n_theta * n_phi
    defaults = {
        "input_dim": n_grid,
        "output_dim": n_grid,
        "intermediate_dim": 256,
        "model_type": "fno2d",
        "grid_shape": (n_theta, n_phi),
        "fno_modes": 36,
        "fno_width": 24,
        "fno_layers": 4,
        "fno_proj_dim": 64,
        "fno_use_norm": False,
    }
    defaults.update(overrides)
    return config.ModelConfig(**defaults)


def light_manifold_fno_config(
    n_theta, n_phi, laplacian, model_type="laplacian_fno", **overrides
):
    """Small manifold FNO preset for FM experiments on sphere theta/phi grids."""
    n_grid = n_theta * n_phi
    defaults = {
        "input_dim": n_grid,
        "output_dim": n_grid,
        "intermediate_dim": 256,
        "model_type": model_type,
        "grid_shape": (n_theta, n_phi),
        "laplacian": laplacian,
        "spectral_modes": 64,
        "fno_width": 24,
        "fno_layers": 4,
        "fno_proj_dim": 64,
        "fno_use_norm": False,
        "sobolev_alpha": -1.0,
    }
    defaults.update(overrides)
    return config.ModelConfig(**defaults)


def light_graph_fno_config(
    state_dim, laplacian, model_type="graph_laplacian_fno", **overrides
):
    """Small graph FNO preset for FM experiments on molecular bond graphs."""
    defaults = {
        "input_dim": state_dim,
        "output_dim": state_dim,
        "intermediate_dim": 256,
        "model_type": model_type,
        "laplacian": laplacian,
        "spectral_modes": min(66, state_dim),
        "fno_modes": min(66, state_dim),
        "fno_width": 24,
        "fno_layers": 4,
        "fno_proj_dim": 64,
        "fno_use_norm": False,
        "sobolev_alpha": -1.0,
    }
    defaults.update(overrides)
    return config.ModelConfig(**defaults)


# Reference size for AD3 (state_dim=66) model comparisons (~137k params).
AD3_BALANCED_TARGET_PARAMS = 137_441


def balanced_ad3_model_config(
    state_dim, laplacian, model_type="mlp", sobolev_alpha=-1.0, **overrides
):
    """
    Per-backbone hyperparameters tuned for comparable capacity on AD3 (N=66).

    Target ~137k trainable parameters (matching graph_laplacian_fno at
    width=32, layers=2, modes=22). Counts at state_dim=66:
      mlp                  ~138k  (intermediate_dim=141)
      fno1d                ~138k  (width=32, layers=2, modes=64)
      graph_laplacian_fno  ~137k  (width=32, layers=2, modes=22)
      graph_sobolev_fno    ~137k  (same as graph_laplacian_fno)
    """
    spectral_modes = min(66, state_dim)
    presets = {
        "mlp": {
            "intermediate_dim": 141,
        },
        "fno1d": {
            "intermediate_dim": 256,
            "fno_modes": 64,
            "fno_width": 32,
            "fno_layers": 2,
        },
        "graph_laplacian_fno": {
            "intermediate_dim": 256,
            "laplacian": laplacian,
            "spectral_modes": spectral_modes,
            "fno_modes": 22,
            "fno_width": 32,
            "fno_layers": 2,
        },
        "graph_sobolev_fno": {
            "intermediate_dim": 256,
            "laplacian": laplacian,
            "spectral_modes": spectral_modes,
            "fno_modes": 22,
            "fno_width": 32,
            "fno_layers": 2,
            "sobolev_alpha": sobolev_alpha,
        },
    }
    if model_type not in presets:
        raise ValueError(
            f"balanced_ad3_model_config does not support model_type='{model_type}'. "
            f"Expected one of: {sorted(presets)}."
        )
    defaults = {
        "input_dim": state_dim,
        "output_dim": state_dim,
        "model_type": model_type,
        **presets[model_type],
    }
    defaults.update(overrides)
    return config.ModelConfig(**defaults)


def count_model_parameters(model_config):
    """Return total trainable parameter count for a ModelConfig."""
    model = build_model(model_config)
    return sum(p.numel() for p in model.parameters())


def print_ad3_balanced_param_counts(state_dim, laplacian, sobolev_alpha=-1.0):
    """Print parameter counts for all AD3 comparison backbones."""
    model_types = ("mlp", "fno1d", "graph_laplacian_fno", "graph_sobolev_fno")
    print(f"AD3 balanced model sizes (state_dim={state_dim}, target ~{AD3_BALANCED_TARGET_PARAMS:,}):")
    for model_type in model_types:
        cfg = balanced_ad3_model_config(
            state_dim, laplacian, model_type=model_type, sobolev_alpha=sobolev_alpha
        )
        n_params = count_model_parameters(cfg)
        print(f"  ({model_type}): {n_params:,} parameters")


# S2 sphere benchmarks use the same ~137k target as AD3 for cross-domain comparability.
S2_BALANCED_TARGET_PARAMS = AD3_BALANCED_TARGET_PARAMS


def balanced_s2_model_config(
    n_theta,
    n_phi,
    laplacian,
    model_type="mlp",
    sobolev_alpha=-1.0,
    **overrides,
):
    """
    Per-backbone hyperparameters tuned for comparable capacity on S2 grids.

    Target ~137k trainable parameters (same budget as AD3 balanced configs).
    Counts at n_theta=20, n_phi=40 (n_grid=800):
      mlp            ~137k  (intermediate_dim=40)
      fno2d          ~134k  (width=32, layers=2, modes=64)
      laplacian_fno  ~134k  (width=32, layers=2, modes=64)
      sobolev_fno    ~134k  (same as laplacian_fno)
    """
    n_grid = n_theta * n_phi
    spectral_modes = min(64, n_grid)
    presets = {
        "mlp": {
            "intermediate_dim": 40,
        },
        "fno2d": {
            "intermediate_dim": 256,
            "grid_shape": (n_theta, n_phi),
            "fno_modes": 64,
            "fno_width": 32,
            "fno_layers": 2,
        },
        "laplacian_fno": {
            "intermediate_dim": 256,
            "grid_shape": (n_theta, n_phi),
            "laplacian": laplacian,
            "spectral_modes": spectral_modes,
            "fno_modes": spectral_modes,
            "fno_width": 32,
            "fno_layers": 2,
        },
        "sobolev_fno": {
            "intermediate_dim": 256,
            "grid_shape": (n_theta, n_phi),
            "laplacian": laplacian,
            "spectral_modes": spectral_modes,
            "fno_modes": spectral_modes,
            "fno_width": 32,
            "fno_layers": 2,
            "sobolev_alpha": sobolev_alpha,
        },
    }
    if model_type not in presets:
        raise ValueError(
            f"balanced_s2_model_config does not support model_type='{model_type}'. "
            f"Expected one of: {sorted(presets)}."
        )
    defaults = {
        "input_dim": n_grid,
        "output_dim": n_grid,
        "model_type": model_type,
        **presets[model_type],
    }
    defaults.update(overrides)
    return config.ModelConfig(**defaults)


def print_s2_balanced_param_counts(
    n_theta, n_phi, laplacian, sobolev_alpha=-1.0
):
    """Print parameter counts for all S2 comparison backbones."""
    n_grid = n_theta * n_phi
    model_types = ("mlp", "fno2d", "laplacian_fno", "sobolev_fno")
    print(
        f"S2 balanced model sizes (n_grid={n_grid}, target ~{S2_BALANCED_TARGET_PARAMS:,}):"
    )
    for model_type in model_types:
        cfg = balanced_s2_model_config(
            n_theta,
            n_phi,
            laplacian,
            model_type=model_type,
            sobolev_alpha=sobolev_alpha,
        )
        n_params = count_model_parameters(cfg)
        print(f"  ({model_type}): {n_params:,} parameters")


def build_model(model_config):
    """Factory for FM velocity-field backbones."""
    model_type = getattr(model_config, "model_type", "mlp")
    if model_type == "mlp":
        return dummy_mlp(model_config)
    if model_type == "fno1d":
        return FNO1dVelocityField(model_config)
    if model_type == "fno2d":
        return FNO2dVelocityField(model_config)
    if model_type == "laplacian_fno":
        return LaplacianFNOVelocityField(model_config)
    if model_type == "sobolev_fno":
        return SobolevFNOVelocityField(model_config)
    if model_type == "graph_laplacian_fno":
        return GraphLaplacianFNOVelocityField(model_config)
    if model_type == "graph_sobolev_fno":
        return GraphSobolevFNOVelocityField(model_config)
    raise ValueError(
        f"Unknown model_type '{model_type}'. "
        f"Expected one of: mlp, fno1d, fno2d, laplacian_fno, sobolev_fno, "
        f"graph_laplacian_fno, graph_sobolev_fno."
    )


class dummy_mlp(torch.nn.Module):
    def __init__(self, model_config):
        super(dummy_mlp, self).__init__()
        self.model_config = model_config
        # do not mutate the passed config object; compute internal input dim
        in_dim = model_config.input_dim + (1 if model_config.enable_time else 0)
        self.linear1 = torch.nn.Linear(in_dim, 2*model_config.intermediate_dim)
        self.linear2 = torch.nn.Linear(2*model_config.intermediate_dim, model_config.intermediate_dim)
        self.linear3 = torch.nn.Linear(model_config.intermediate_dim, model_config.intermediate_dim)
        self.linear4 = torch.nn.Linear(model_config.intermediate_dim, 2*model_config.intermediate_dim)
        self.linear5 = torch.nn.Linear(2*model_config.intermediate_dim, model_config.output_dim)

    def forward(self, x, t=None):
        if t is not None and self.model_config.enable_time:
            x = torch.cat([x, t], dim=1)
        x = self.linear1(x)
        x = torch.selu(x)
        x = self.linear2(x)
        x = torch.selu(x)
        x = self.linear3(x)
        x = torch.selu(x)
        x = self.linear4(x)
        x = torch.selu(x)
        x = self.linear5(x)
        return x