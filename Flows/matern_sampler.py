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

