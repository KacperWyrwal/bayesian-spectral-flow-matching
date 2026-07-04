import torch


def build_path_laplacian(n: int, device=None) -> torch.Tensor:
    """1D path-graph Laplacian on n nodes (no periodic wrap)."""
    if n <= 0:
        raise ValueError("n must be positive.")
    if n == 1:
        return torch.zeros((1, 1), dtype=torch.float32, device=device)
    adj = torch.zeros((n, n), dtype=torch.float32, device=device)
    for i in range(n - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    degree = torch.diag(adj.sum(dim=1))
    return degree - adj


def build_cycle_laplacian(n: int, device=None) -> torch.Tensor:
    """1D cycle-graph Laplacian on n nodes (periodic wrap)."""
    if n <= 0:
        raise ValueError("n must be positive.")
    if n == 1:
        return torch.zeros((1, 1), dtype=torch.float32, device=device)
    adj = torch.zeros((n, n), dtype=torch.float32, device=device)
    for i in range(n):
        adj[i, (i + 1) % n] = 1.0
        adj[i, (i - 1) % n] = 1.0
    degree = torch.diag(adj.sum(dim=1))
    return degree - adj


def precompute_separable_grid_basis(
    n_theta: int,
    n_phi: int,
    modes1: int,
    modes2: int,
    device=None,
):
    """
    Separable 2D grid Laplacian eigenbasis: L = L_theta ⊗ I + I ⊗ L_phi.

    Returns per-axis eigenpairs for theta (path) and phi (cycle), matching the
    sphere theta/phi grid layout used in Dataset.sphere.generate_grid_laplacian.
    """
    eval_theta, phi_theta = precompute_laplacian_basis(
        build_path_laplacian(n_theta, device=device), modes1, device=device
    )
    eval_phi, phi_phi = precompute_laplacian_basis(
        build_cycle_laplacian(n_phi, device=device), modes2, device=device
    )
    return eval_theta, phi_theta, eval_phi, phi_phi


def precompute_laplacian_basis(laplacian, k_modes, device=None):
    """
    Eigendecompose graph Laplacian and return the first K modes.

    Args:
        laplacian: (N, N) symmetric Laplacian matrix
        k_modes: number of eigenmodes to keep
        device: optional target device for returned tensors

    Returns:
        eigenvalues: (K,)
        eigenvectors: (N, K) — columns are orthonormal eigenvectors
    """
    if device is None and torch.is_tensor(laplacian):
        device = laplacian.device
    if torch.is_tensor(laplacian):
        L = laplacian.to(device).float()
    else:
        L = torch.tensor(laplacian, dtype=torch.float32, device=device)

    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    k = min(int(k_modes), eigenvectors.shape[1])
    return eigenvalues[:k], eigenvectors[:, :k]
