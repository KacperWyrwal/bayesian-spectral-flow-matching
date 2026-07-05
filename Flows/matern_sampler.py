import torch


def _to_laplacian_tensor(laplacian, device):
    if torch.is_tensor(laplacian):
        return laplacian.to(device).float()
    return torch.tensor(laplacian, dtype=torch.float32, device=device)


def sample_matern_graph(num_samples, laplacian, nu=2.0, tau=1.0, device=None):
    """
    Sample x ~ N(0, (tau I + L)^(-nu)) in Laplacian spectral basis.
    This is the kappa=0 stable parameterization.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    L = _to_laplacian_tensor(laplacian, device)
    eigenvalues, eigenvectors = torch.linalg.eigh(L)

    spectral_var = (tau + eigenvalues).pow(-nu)
    spectral_std = torch.sqrt(spectral_var).unsqueeze(0)

    eps = torch.randn(num_samples, eigenvalues.shape[0], device=device)
    coeffs = eps * spectral_std
    samples = coeffs @ eigenvectors.T
    return samples


def log_p_matern_graph(x, laplacian, nu=2.0, tau=1.0):
    """
    Log density of N(0, (tau I + L)^(-nu)) in the Laplacian eigenbasis.

    x: (B, d) samples on the same space as sample_matern_graph.
    """
    if device := getattr(x, "device", None):
        device = x.device
    else:
        device = torch.device("cpu")

    L = _to_laplacian_tensor(laplacian, device)
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    spectral_var = (tau + eigenvalues).pow(-nu)

    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    else:
        x = x.to(device).float()

    coeffs = x @ eigenvectors
    log_p = -0.5 * (coeffs ** 2 / spectral_var.unsqueeze(0)).sum(dim=-1)
    log_p = log_p - 0.5 * torch.log(spectral_var).sum()
    d = eigenvalues.shape[0]
    log_p = log_p - 0.5 * d * torch.log(
        torch.tensor(2.0 * torch.pi, device=device, dtype=x.dtype)
    )
    return log_p

