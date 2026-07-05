"""FALCON-style Boltzmann generator metrics (T-W2, E-W2, ESS, SNIS)."""

from __future__ import annotations

import numpy as np
import ot


def _periodic_diff(delta: np.ndarray) -> np.ndarray:
    """Shortest signed difference on the circle, in (-pi, pi]."""
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def torus_pairwise_cost(dihedrals_a: np.ndarray, dihedrals_b: np.ndarray) -> np.ndarray:
    """
    Squared torus distance between dihedral sets.

    dihedrals_a: (N, D), dihedrals_b: (M, D) in radians.
    Returns cost matrix (N, M).
    """
    a = np.asarray(dihedrals_a, dtype=np.float64)
    b = np.asarray(dihedrals_b, dtype=np.float64)
    diff = a[:, None, :] - b[None, :, :]
    periodic = _periodic_diff(diff)
    return np.sum(periodic ** 2, axis=-1)


def torus_wasserstein_w2(dihedrals_a: np.ndarray, dihedrals_b: np.ndarray) -> float:
    """2-Wasserstein distance on periodic dihedral angles (FALCON T-W2)."""
    cost = torus_pairwise_cost(dihedrals_a, dihedrals_b)
    n_a, n_b = cost.shape
    a = np.ones(n_a, dtype=np.float64) / n_a
    b = np.ones(n_b, dtype=np.float64) / n_b
    w2_sq = ot.emd2(a, b, cost)
    return float(np.sqrt(max(w2_sq, 0.0)))


def energy_wasserstein_w2(energies_a: np.ndarray, energies_b: np.ndarray) -> float:
    """1D 2-Wasserstein distance between energy samples (FALCON E-W2)."""
    a = np.asarray(energies_a, dtype=np.float64).ravel()
    b = np.asarray(energies_b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        raise ValueError("Energy arrays must be non-empty.")
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    n_a, n_b = a_sorted.size, b_sorted.size
    # Equal-mass coupling between empirical 1D distributions via quantile interpolation.
    quantiles = np.linspace(0.0, 1.0, max(n_a, n_b), endpoint=False)
    q_a = np.quantile(a_sorted, quantiles, method="linear")
    q_b = np.quantile(b_sorted, quantiles, method="linear")
    w2_sq = np.mean((q_a - q_b) ** 2)
    return float(np.sqrt(w2_sq))


def kish_ess(log_weights: np.ndarray) -> float:
    """
    Normalized effective sample size (Kish, 1957).

    log_weights: unnormalized log importance weights (any constant offset).
    Returns ESS / N in (0, 1].
    """
    lw = np.asarray(log_weights, dtype=np.float64).ravel()
    lw = lw - lw.max()
    w = np.exp(lw)
    w_sum = w.sum()
    if w_sum <= 0.0:
        return 0.0
    w_norm = w / w_sum
    n = w_norm.size
    ess = 1.0 / (n * np.sum(w_norm ** 2))
    return float(ess)


def snis_log_weights(
    energies_kj_mol: np.ndarray,
    log_p_theta: np.ndarray,
    kbt_kj_mol: float,
) -> np.ndarray:
    """
    Unnormalized log SNIS weights: log w = -E/kBT - log p_theta(x).

    energies_kj_mol: potential energies in kJ/mol.
    log_p_theta: log density under the generative model.
    """
    e = np.asarray(energies_kj_mol, dtype=np.float64).ravel()
    lp = np.asarray(log_p_theta, dtype=np.float64).ravel()
    if e.shape != lp.shape:
        raise ValueError(
            f"energies shape {e.shape} must match log_p_theta shape {lp.shape}"
        )
    return -e / kbt_kj_mol - lp


def snis_normalized_weights(log_weights: np.ndarray) -> np.ndarray:
    """Self-normalized importance weights summing to 1."""
    lw = np.asarray(log_weights, dtype=np.float64).ravel()
    lw = lw - lw.max()
    w = np.exp(lw)
    return w / w.sum()


def resample_weighted(
    samples: np.ndarray,
    log_weights: np.ndarray,
    n_out: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Multinomial resampling for SNIS-corrected empirical measures."""
    x = np.asarray(samples)
    probs = snis_normalized_weights(log_weights)
    n_out = int(n_out if n_out is not None else x.shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=n_out, replace=True, p=probs)
    return x[idx]
