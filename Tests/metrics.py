import ot
import numpy as np

# X_true: shape (M, N) - True graph signals
# X_gen:  shape (M, N) - Flow Matching generated signals
def wasserstein_distance(X_true, X_gen):
    # 1. Compute pairwise squared Euclidean distance matrix
    M = ot.dist(X_gen, X_true, metric='sqeuclidean')

    # 2. Assume uniform weights for empirical samples
    a, b = np.ones(M.shape[0]) / M.shape[0], np.ones(M.shape[1]) / M.shape[1]

    # 3. Compute W2 squared (exact)
    w2_squared = ot.emd2(a, b, M)
    w2 = np.sqrt(w2_squared)
    return w2


def sobolev_wasserstein_distance(X_true, X_gen, eigenvalues, eigenvectors, beta=1.0):
    """
    W2 under Sobolev spectral cost:
        d^2(x, y) = sum_k (1 + lambda_k)^(-beta) * (phi_k^T (x - y))^2

    eigenvalues: (K,) Laplacian eigenvalues (truncated)
    eigenvectors: (D, K) Laplacian eigenvectors (truncated)
    """
    phi = np.asarray(eigenvectors, dtype=np.float64)
    weights = (1.0 + np.asarray(eigenvalues, dtype=np.float64)) ** (-beta)

    y0 = X_gen @ phi
    y1 = X_true @ phi
    diff = y0[:, None, :] - y1[None, :, :]
    cost = np.sum((diff ** 2) * weights[None, None, :], axis=-1)

    a = np.ones(cost.shape[0], dtype=np.float64) / cost.shape[0]
    b = np.ones(cost.shape[1], dtype=np.float64) / cost.shape[1]
    w2_squared = ot.emd2(a, b, cost)
    return float(np.sqrt(w2_squared))


def rmsd_distribution(X_gen, X_true, n_atoms=22, feat_dim=3):
    """
    COM-aligned RMSD between generated and true conformers.

    X_gen, X_true: (B, n_atoms * feat_dim) flat feature vectors.
    Returns dict with mean and std of per-sample minimum RMSD to any true sample.
    """
    gen = np.asarray(X_gen, dtype=np.float64).reshape(-1, n_atoms, feat_dim)
    true = np.asarray(X_true, dtype=np.float64).reshape(-1, n_atoms, feat_dim)

    rmsds = []
    for g in gen:
        g_centered = g - g.mean(axis=0, keepdims=True)
        best = np.inf
        for t in true:
            t_centered = t - t.mean(axis=0, keepdims=True)
            diff = g_centered - t_centered
            rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
            best = min(best, rmsd)
        rmsds.append(best)

    rmsds = np.asarray(rmsds, dtype=np.float64)
    return {"mean": float(rmsds.mean()), "std": float(rmsds.std())}

"""
metrics for evaluating the quality of generated signals on the sphere
"""
def angular_power_spectrum(coeffs_batch, lm_pairs):
    """
    C_l = (1 / (2l+1)) * Σ_m  mean_over_samples(|a_lm|²)
    Returns: array of shape (L_max+1,) — one value per degree l
    """
    L_max = max(l for l, m in lm_pairs)
    Cl    = np.zeros(L_max + 1)
    pairs = np.array(lm_pairs)

    for l in range(L_max + 1):
        m_mask       = pairs[:, 0] == l          # all m indices at this degree
        Cl[l] = coeffs_batch[:, m_mask].var(axis=0).mean()  # avg variance over m

    return Cl

def spectral_divergence(gen_coeffs, true_std, lm_pairs):
    """L2 distance between generated and true per-coefficient std."""
    gen_std = gen_coeffs.std(axis=0)
    return float(np.mean((gen_std - true_std) ** 2))

def harmonic_mode_error(gen_coeffs, true_coeffs, lm_pairs):
    """
    Topological claim (Prop 1): the l=0 mode (λ=0) should be preserved.
    Compare the mean and std of the a_00 coefficient.
    """
    l0_mask  = np.array([l == 0 for l, m in lm_pairs])
    gen_a00  = gen_coeffs[:, l0_mask].ravel()
    true_a00 = true_coeffs[:, l0_mask].ravel()
    return {
        'true_mean':  float(true_a00.mean()),
        'true_std':   float(true_a00.std()),
        'gen_std_fm': None,   # fill in after generating
        'gen_std_sob': None,
    }