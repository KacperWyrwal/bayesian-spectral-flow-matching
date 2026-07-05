import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.FNO import fno2d_spectral_dims
from models.spectral_basis import (
    precompute_laplacian_basis,
    precompute_separable_grid_basis,
)


class LaplacianSpectralConv1d(nn.Module):
    """
    NORM-style spectral integral operator on graph Laplacian eigenbasis.
    Computes: Phi @ R @ (Phi^T v) truncated to K modes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        eigenvectors: torch.Tensor,
    ):
        super().__init__()
        self.modes = modes
        phi = eigenvectors[:, :modes].float()
        self.register_buffer("phi", phi)  # (N, K)

        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, N)"""
        x_hat = torch.einsum("bcn,nk->bck", x, self.phi)
        out_hat = torch.einsum("bck,iok->bok", x_hat, self.weights)
        return torch.einsum("bok,nk->bon", out_hat, self.phi)


class LaplacianSpectralConv2d(nn.Module):
    """
    Separable 2D Laplacian spectral operator on an H×W grid.

    Mirrors FNO2d's rfft2 layout: encode/decode along theta then phi with
    independent 1D graph-Laplacian eigenbases, and learn weights on a
    (modes1, modes2) coefficient grid instead of collapsing H×W to 1D modes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        phi_theta: torch.Tensor,
        phi_phi: torch.Tensor,
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.register_buffer("phi_theta", phi_theta[:, :modes1].float())  # (H, modes1)
        self.register_buffer("phi_phi", phi_phi[:, :modes2].float())  # (W, modes2)

        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, H, W)"""
        x_phi = torch.einsum("bchw,wq->bchq", x, self.phi_phi)
        x_hat = torch.einsum("bchq,hp->bcqp", x_phi, self.phi_theta)
        out_hat = torch.einsum("bcqp,iopq->boqp", x_hat, self.weights)
        out_theta = torch.einsum("boqp,hp->bohq", out_hat, self.phi_theta)
        return torch.einsum("bohq,wq->bohw", out_theta, self.phi_phi)


class SobolevSpectralConv1d(nn.Module):
    """
    Laplacian eigenbasis with whitened Sobolev coordinates.
    Encode:  c_k = (1+lambda_k)^(alpha/2) <x, phi_k>
    Decode:  x = sum_k (1+lambda_k)^(-alpha/2) c_k phi_k
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        sobolev_alpha: float = 0.0,
    ):
        super().__init__()
        self.modes = modes
        self.sobolev_alpha = sobolev_alpha

        evals = eigenvalues[:modes].float()
        phi = eigenvectors[:, :modes].float()
        self.register_buffer("phi", phi)
        self.register_buffer("encode_scale", (1.0 + evals).pow(sobolev_alpha / 2.0))
        self.register_buffer("decode_scale", (1.0 + evals).pow(-sobolev_alpha / 2.0))

        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, N)"""
        x_hat = torch.einsum("bcn,nk->bck", x, self.phi)
        x_hat = x_hat * self.encode_scale.view(1, 1, -1)
        out_hat = torch.einsum("bck,iok->bok", x_hat, self.weights)
        out_hat = out_hat * self.decode_scale.view(1, 1, -1)
        return torch.einsum("bok,nk->bon", out_hat, self.phi)


class SobolevSpectralConv2d(nn.Module):
    """
    Separable 2D Sobolev-whitened Laplacian spectral operator on an H×W grid.

    Uses Kronecker-sum eigenvalues lambda_{k1,k2} = lambda_theta[k1] + lambda_phi[k2]
    on the (modes1, modes2) coefficient grid.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        eval_theta: torch.Tensor,
        eval_phi: torch.Tensor,
        phi_theta: torch.Tensor,
        phi_phi: torch.Tensor,
        sobolev_alpha: float = 0.0,
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.sobolev_alpha = sobolev_alpha

        self.register_buffer("phi_theta", phi_theta[:, :modes1].float())
        self.register_buffer("phi_phi", phi_phi[:, :modes2].float())

        eval_grid = (
            eval_theta[:modes1, None].float()
            + eval_phi[None, :modes2].float()
        )
        self.register_buffer(
            "encode_scale", (1.0 + eval_grid).pow(sobolev_alpha / 2.0)
        )
        self.register_buffer(
            "decode_scale", (1.0 + eval_grid).pow(-sobolev_alpha / 2.0)
        )

        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, H, W)"""
        x_phi = torch.einsum("bchw,wq->bchq", x, self.phi_phi)
        x_hat = torch.einsum("bchq,hp->bcqp", x_phi, self.phi_theta)
        x_hat = x_hat * self.encode_scale.view(1, 1, self.modes1, self.modes2)
        out_hat = torch.einsum("bcqp,iopq->boqp", x_hat, self.weights)
        out_hat = out_hat * self.decode_scale.view(1, 1, self.modes1, self.modes2)
        out_theta = torch.einsum("boqp,hp->bohq", out_hat, self.phi_theta)
        return torch.einsum("bohq,wq->bohw", out_theta, self.phi_phi)


class ManifoldFNOLayer(nn.Module):
    """One manifold FNO layer: 2D eigenbasis spectral path + 2D bypass, then activation."""

    def __init__(self, width: int, spectral: nn.Module, use_norm: bool = True):
        super().__init__()
        self.spectral = spectral
        self.bypass = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(1, width) if use_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)"""
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


class ManifoldFNO(nn.Module):
    """
    Graph-spectral neural operator on an H×W grid.
    Spectral convolutions use separable 2D Laplacian eigenbases on (B, C, H, W),
    matching FNO2d's rfft2 tensor layout and mode grid.
    """

    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 64,
        n_layers: int = 4,
        d_a: int = 1,
        d_u: int = 1,
        grid_shape: tuple[int, int] = None,
        eval_theta: torch.Tensor = None,
        eval_phi: torch.Tensor = None,
        phi_theta: torch.Tensor = None,
        phi_phi: torch.Tensor = None,
        sobolev_alpha: float = None,
        proj_dim: int = None,
        use_norm: bool = True,
    ):
        super().__init__()
        if grid_shape is None:
            raise ValueError("ManifoldFNO requires grid_shape=(H, W).")
        proj_dim = proj_dim or width
        self.lifting = nn.Conv2d(d_a + 2, width, kernel_size=1)

        use_sobolev = sobolev_alpha is not None
        layers = []
        for _ in range(n_layers):
            if use_sobolev:
                spectral = SobolevSpectralConv2d(
                    width,
                    width,
                    modes1,
                    modes2,
                    eval_theta,
                    eval_phi,
                    phi_theta,
                    phi_phi,
                    sobolev_alpha,
                )
            else:
                spectral = LaplacianSpectralConv2d(
                    width,
                    width,
                    modes1,
                    modes2,
                    phi_theta,
                    phi_phi,
                )
            layers.append(ManifoldFNOLayer(width, spectral, use_norm=use_norm))
        self.fno_layers = nn.ModuleList(layers)

        self.projection = (
            nn.Conv2d(width, d_u, kernel_size=1)
            if proj_dim <= 0 or proj_dim == width
            else nn.Sequential(
                nn.Conv2d(width, proj_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(proj_dim, d_u, kernel_size=1),
            )
        )

    def forward(self, a: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        a:      (B, H, W, d_a)
        coords: (B, H, W, 2)
        returns (B, H, W, d_u)
        """
        inp = torch.cat([a, coords], dim=-1).permute(0, 3, 1, 2)  # (B, d_a+2, H, W)
        v = self.lifting(inp)
        for layer in self.fno_layers:
            v = layer(v)
        return self.projection(v).permute(0, 2, 3, 1)  # (B, H, W, d_u)


def _synthesis_grid_coords(H: int, W: int) -> torch.Tensor:
    """Real (theta, phi) grid matching Dataset.sphere.build_synthesis_matrix."""
    theta_vals = torch.linspace(1e-4, math.pi - 1e-4, H)
    phi_vals = torch.linspace(0.0, 2.0 * math.pi, W + 1)[:-1]
    theta, phi = torch.meshgrid(theta_vals, phi_vals, indexing="ij")
    return torch.stack([theta, phi], dim=-1).unsqueeze(0)  # (1, H, W, 2)


def _resolve_manifold_basis(model_config):
    laplacian = getattr(model_config, "laplacian", None)
    if laplacian is None:
        raise ValueError(
            "Manifold FNO requires model_config.laplacian to be set."
        )
    spectral_modes = getattr(model_config, "spectral_modes", None)
    if spectral_modes is None:
        spectral_modes = model_config.fno_modes
    if torch.is_tensor(laplacian):
        device = laplacian.device
    else:
        device = getattr(model_config, "device", None)
    eigenvalues, eigenvectors = precompute_laplacian_basis(
        laplacian, spectral_modes, device=device
    )
    return eigenvalues, eigenvectors, int(spectral_modes)


def _resolve_separable_grid_basis(model_config):
    """Per-axis Laplacian eigenbases for 2D manifold FNO (matches FNO2d mode grid)."""
    if model_config.grid_shape is None:
        raise ValueError(
            "Separable manifold FNO requires model_config.grid_shape=(H, W)."
        )
    H, W = model_config.grid_shape
    spectral_modes = getattr(model_config, "spectral_modes", None)
    if spectral_modes is None:
        spectral_modes = model_config.fno_modes
    modes1, modes2 = fno2d_spectral_dims(
        spectral_modes, getattr(model_config, "fno_modes2", None)
    )
    laplacian = getattr(model_config, "laplacian", None)
    device = laplacian.device if torch.is_tensor(laplacian) else None
    eval_theta, phi_theta, eval_phi, phi_phi = precompute_separable_grid_basis(
        H, W, modes1, modes2, device=device
    )
    return eval_theta, phi_theta, eval_phi, phi_phi, modes1, modes2


class _ManifoldFNOVelocityFieldBase(nn.Module):
    """Shared 2D adapter for Laplacian / Sobolev manifold FNO velocity fields."""

    def __init__(self, model_config, sobolev_alpha=None):
        super().__init__()
        self.model_config = model_config
        if model_config.grid_shape is None:
            raise ValueError(
                "Manifold FNO velocity fields require model_config.grid_shape=(H, W)."
            )
        self.H, self.W = model_config.grid_shape
        grid_size = self.H * self.W
        if model_config.input_dim != grid_size or model_config.output_dim != grid_size:
            raise ValueError(
                f"grid_shape {model_config.grid_shape} implies dim {grid_size}, "
                f"but got input_dim={model_config.input_dim}, "
                f"output_dim={model_config.output_dim}."
            )

        if getattr(model_config, "laplacian", None) is None:
            raise ValueError(
                "Manifold FNO velocity fields require model_config.laplacian."
            )
        self.register_buffer("coords", _synthesis_grid_coords(self.H, self.W))

        d_a = 2 if model_config.enable_time else 1
        proj_dim = getattr(model_config, "fno_proj_dim", None)
        use_norm = getattr(model_config, "fno_use_norm", True)
        (
            eval_theta,
            phi_theta,
            eval_phi,
            phi_phi,
            modes1,
            modes2,
        ) = _resolve_separable_grid_basis(model_config)
        self.fno = ManifoldFNO(
            modes1=modes1,
            modes2=modes2,
            width=model_config.fno_width,
            n_layers=model_config.fno_layers,
            d_a=d_a,
            d_u=1,
            grid_shape=(self.H, self.W),
            eval_theta=eval_theta,
            eval_phi=eval_phi,
            phi_theta=phi_theta,
            phi_phi=phi_phi,
            sobolev_alpha=sobolev_alpha,
            proj_dim=proj_dim,
            use_norm=use_norm,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        B = x.shape[0]
        a = x.view(B, self.H, self.W).unsqueeze(-1)
        if t is not None and self.model_config.enable_time:
            t_map = t.view(B, 1, 1, 1).expand(B, self.H, self.W, 1)
            a = torch.cat([a, t_map], dim=-1)
        coords = self.coords.expand(B, -1, -1, -1).to(x.device)
        v = self.fno(a, coords)
        return v.squeeze(-1).reshape(B, self.H * self.W)


class LaplacianFNOVelocityField(_ManifoldFNOVelocityFieldBase):
    """FM adapter using Laplacian eigenbasis spectral convolutions on a 2D grid."""

    def __init__(self, model_config):
        super().__init__(model_config, sobolev_alpha=None)


class SobolevFNOVelocityField(_ManifoldFNOVelocityFieldBase):
    """FM adapter using whitened Sobolev spectral convolutions on a 2D grid."""

    def __init__(self, model_config):
        sobolev_alpha = getattr(model_config, "sobolev_alpha", 0.0)
        super().__init__(model_config, sobolev_alpha=sobolev_alpha)


class GraphFNOLayer(nn.Module):
    """One graph FNO layer: Laplacian spectral path + 1D bypass."""

    def __init__(self, width: int, spectral: nn.Module, use_norm: bool = True):
        super().__init__()
        self.spectral = spectral
        self.bypass = nn.Conv1d(width, width, kernel_size=1)
        self.norm = nn.InstanceNorm1d(width) if use_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, N)"""
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


class GraphFNO(nn.Module):
    """
    Graph-spectral neural operator on N nodes (1D flattened graph).

    Input:  (B, N, d_a)   function values at nodes
    Coords: (B, N, 1)     normalized node coordinates
    Output: (B, N, d_u)
    """

    def __init__(
        self,
        modes: int = 16,
        width: int = 64,
        n_layers: int = 4,
        d_a: int = 1,
        d_u: int = 1,
        eigenvalues: torch.Tensor = None,
        eigenvectors: torch.Tensor = None,
        sobolev_alpha: float = None,
        proj_dim: int = None,
        use_norm: bool = True,
    ):
        super().__init__()
        proj_dim = proj_dim or width
        self.lifting = nn.Linear(d_a + 1, width)

        use_sobolev = sobolev_alpha is not None
        layers = []
        for _ in range(n_layers):
            if use_sobolev:
                spectral = SobolevSpectralConv1d(
                    width, width, modes, eigenvalues, eigenvectors, sobolev_alpha
                )
            else:
                spectral = LaplacianSpectralConv1d(width, width, modes, eigenvectors)
            layers.append(GraphFNOLayer(width, spectral, use_norm=use_norm))
        self.fno_layers = nn.ModuleList(layers)

        if proj_dim <= 0 or proj_dim == width:
            self.projection = nn.Linear(width, d_u)
        else:
            self.projection = nn.Sequential(
                nn.Linear(width, proj_dim),
                nn.GELU(),
                nn.Linear(proj_dim, d_u),
            )

    def forward(self, a: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        a:      (B, N, d_a)
        coords: (B, N, 1)
        returns (B, N, d_u)
        """
        inp = torch.cat([a, coords], dim=-1)
        v = self.lifting(inp)
        v = v.permute(0, 2, 1)
        for layer in self.fno_layers:
            v = layer(v)
        v = v.permute(0, 2, 1)
        return self.projection(v)


class _GraphFNOVelocityFieldBase(nn.Module):
    """Shared 1D adapter for graph Laplacian / Sobolev FNO velocity fields."""

    def __init__(self, model_config, sobolev_alpha=None):
        super().__init__()
        self.model_config = model_config
        self.N = model_config.input_dim
        if model_config.output_dim != self.N:
            raise ValueError("Graph FNO velocity fields require input_dim == output_dim.")

        eigenvalues, eigenvectors, modes = _resolve_manifold_basis(model_config)
        self.register_buffer(
            "coords",
            torch.linspace(0.0, 1.0, self.N).view(1, self.N, 1),
        )

        d_a = 2 if model_config.enable_time else 1
        proj_dim = getattr(model_config, "fno_proj_dim", None)
        use_norm = getattr(model_config, "fno_use_norm", True)
        self.fno = GraphFNO(
            modes=modes,
            width=model_config.fno_width,
            n_layers=model_config.fno_layers,
            d_a=d_a,
            d_u=1,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            sobolev_alpha=sobolev_alpha,
            proj_dim=proj_dim,
            use_norm=use_norm,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        B, N = x.shape
        if N != self.N:
            raise ValueError(f"Expected state dimension {self.N}, got {N}.")
        a = x.unsqueeze(-1)
        if t is not None and self.model_config.enable_time:
            t_node = t.view(B, 1, 1).expand(B, N, 1)
            a = torch.cat([a, t_node], dim=-1)
        coords = self.coords.expand(B, -1, -1).to(x.device)
        v = self.fno(a, coords)
        return v.squeeze(-1)


class GraphLaplacianFNOVelocityField(_GraphFNOVelocityFieldBase):
    """FM adapter using Laplacian eigenbasis spectral convolutions on a graph."""

    def __init__(self, model_config):
        super().__init__(model_config, sobolev_alpha=None)


class GraphSobolevFNOVelocityField(_GraphFNOVelocityFieldBase):
    """FM adapter using whitened Sobolev spectral convolutions on a graph."""

    def __init__(self, model_config):
        sobolev_alpha = getattr(model_config, "sobolev_alpha", 0.0)
        super().__init__(model_config, sobolev_alpha=sobolev_alpha)
