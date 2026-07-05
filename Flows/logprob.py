"""CNF log-density via augmented ODE (instantaneous change of variables)."""

from __future__ import annotations

from typing import Callable, Optional

import torch

LogProbFn = Callable[[torch.Tensor], torch.Tensor]


def log_p0_gaussian(x0: torch.Tensor) -> torch.Tensor:
    """Log density of N(0, I) on R^d."""
    d = x0.shape[-1]
    log_norm = -0.5 * d * torch.log(
        torch.tensor(2.0 * torch.pi, device=x0.device, dtype=x0.dtype)
    )
    return -0.5 * (x0 ** 2).sum(dim=-1) + log_norm


def exact_trace(model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Exact Jacobian trace tr(dv/dx) for batch of states.

    x: (B, d), t: (B, 1)
    """
    batch_size, dim = x.shape
    x = x.detach().requires_grad_(True)
    velocity = model(x, t)
    trace = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
    for i in range(dim):
        grad_i = torch.autograd.grad(
            velocity[:, i].sum(),
            x,
            create_graph=False,
            retain_graph=True,
        )[0]
        trace = trace + grad_i[:, i]
    return trace


def euler_integrate_with_logprob(
    model,
    x0: torch.Tensor,
    config,
    log_p0: torch.Tensor,
    num_steps: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fixed-step Euler integration with log-density tracking.

    d log p / dt = -tr(dv/dx)
    """
    num_steps = num_steps if num_steps is not None else config.num_steps
    device = x0.device
    dtype = x0.dtype
    x = x0
    log_p = log_p0

    t_grid = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
    for step in range(num_steps):
        t_val = t_grid[step]
        dt = t_grid[step + 1] - t_val
        t_batch = t_val.expand(x.shape[0], 1)

        with torch.enable_grad():
            trace = exact_trace(model, x, t_batch)

        with torch.no_grad():
            velocity = model(x, t_batch)
            x = x + dt * velocity
            log_p = log_p - dt * trace.detach()

    return x, log_p


def infer_with_logprob(
    model,
    x0: torch.Tensor,
    config,
    source_log_prob_fn: LogProbFn,
    model_type: str = "mlp",
    num_steps: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample x1 from x0 and compute log p_theta(x1) under the CNF induced by v_theta.

    Phase 1: exact-trace likelihood is implemented for model_type='mlp' only.
    """
    if model_type != "mlp":
        raise NotImplementedError(
            f"log-density for model_type='{model_type}' is not implemented yet. "
            "Use model_type='mlp' or extend Flows/logprob.py."
        )

    model.eval()
    log_p0 = source_log_prob_fn(x0)
    return euler_integrate_with_logprob(
        model, x0, config, log_p0, num_steps=num_steps
    )


def make_source_log_prob_fn(flow_config) -> LogProbFn:
    """Build log p0 for Gaussian or Matérn graph source."""
    if flow_config.use_spectral and flow_config.use_matern_source:
        if flow_config.laplacian is None:
            raise ValueError("laplacian required for Matérn source log-density.")

        laplacian = flow_config.laplacian
        nu = flow_config.nu
        tau = flow_config.tau

        def matern_log_prob(x0: torch.Tensor) -> torch.Tensor:
            from Flows.matern_sampler import log_p_matern_graph

            return log_p_matern_graph(x0, laplacian, nu=nu, tau=tau)

        return matern_log_prob

    return log_p0_gaussian
