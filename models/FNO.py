import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def fno2d_spectral_dims(fno_modes, fno_modes2=None):
    """Map total mode budget to per-axis FFT truncation (modes1, modes2)."""
    modes1 = max(1, math.isqrt(int(fno_modes)))
    raw2 = fno_modes if fno_modes2 is None else fno_modes2
    modes2 = max(1, math.isqrt(int(raw2)))
    return modes1, modes2

class SpectralConv1d(nn.Module):
    """
    1D Fourier integral operator layer.
    Computes: IFFT( R · FFT(v) ) truncated to k_max modes.
    R is a complex weight tensor of shape (in_ch, out_ch, k_max).
    """
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.modes = modes
        # Complex weights initialized with Xavier scaling
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, N)"""
        B, C, N = x.shape

        # 1. FFT along spatial dimension → (B, C, N//2+1) complex
        x_ft = torch.fft.rfft(x, norm='ortho')

        # 2. Multiply truncated modes by learned complex weights
        n_freq = x_ft.shape[-1]
        n_modes = min(self.modes, n_freq)
        out_ft = torch.zeros(B, self.weights.shape[1], n_freq,
                             dtype=torch.cfloat, device=x.device)
        # einsum: batch × in_channel → out_channel, for each mode
        out_ft[:, :, :n_modes] = torch.einsum(
            'bim,iom->bom', x_ft[:, :, :n_modes], self.weights[:, :, :n_modes]
        )

        # 3. IFFT back to physical space
        return torch.fft.irfft(out_ft, n=N, norm='ortho')   # (B, C_out, N)


class FNOLayer(nn.Module):
    """
    One FNO layer: spectral path + bypass local linear W, then activation.
    vₜ₊₁ = σ( SpectralConv(vₜ) + W vₜ )
    """
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.bypass   = nn.Conv1d(width, width, kernel_size=1)  # pointwise linear
        self.norm     = nn.InstanceNorm1d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


class FNO1d(nn.Module):
    """
    Fourier Neural Operator for 1D problems.
    Maps a(x) → u(x), discretization-invariant.

    Typical use: 1D Burgers' equation, ODE parameter-to-solution maps.
    Input: (B, N, d_a+1)  — function values + coordinate x concatenated
    Output: (B, N, d_u)
    """
    def __init__(
        self,
        modes: int   = 16,    # k_max: how many Fourier modes to keep
        width: int   = 64,    # d_v: hidden channel dimension
        n_layers: int = 4,
        d_a: int     = 1,     # input function channels (+ 1 for coordinate)
        d_u: int     = 1,     # output channels
    ):
        super().__init__()
        # P: lift (d_a+1) → width
        self.lifting = nn.Linear(d_a + 1, width)

        self.fno_layers = nn.ModuleList([
            FNOLayer(width, modes) for _ in range(n_layers)
        ])

        # Q: project width → d_u (two-layer MLP)
        self.projection = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, d_u),
        )

    def forward(self, a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        a: (B, N, d_a)   — input function values at grid points
        x: (B, N, 1)     — grid coordinates (normalized to [0,1])
        returns: (B, N, d_u)
        """
        # Concatenate coordinate to input, then lift
        inp = torch.cat([a, x], dim=-1)          # (B, N, d_a+1)
        v   = self.lifting(inp)                   # (B, N, width)

        # FNO layers expect (B, C, N) — channels first
        v = v.permute(0, 2, 1)                   # (B, width, N)
        for layer in self.fno_layers:
            v = layer(v)
        v = v.permute(0, 2, 1)                   # (B, N, width)

        return self.projection(v)                 # (B, N, d_u)


class SpectralConv2d(nn.Module):
    """
    2D Fourier integral operator layer.
    Computes: IFFT2( R · FFT2(v) ) truncated to (modes1, modes2) frequencies.
    """
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, H, W)"""
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            B, self.weights.shape[1], H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        m1 = min(self.modes1, H)
        m2 = min(self.modes2, x_ft.shape[-1])
        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :m1, :m2],
            self.weights[:, :, :m1, :m2],
        )
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNOLayer2d(nn.Module):
    """One 2D FNO layer: spectral path + local 1x1 bypass, then activation."""
    def __init__(self, width: int, modes1: int, modes2: int, use_norm: bool = True):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.bypass = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(1, width) if use_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


class FNO2d(nn.Module):
    """
    Fourier Neural Operator for 2D problems.
    Maps a(x, y) -> u(x, y), discretization-invariant on regular grids.

    Input:  (B, H, W, d_a)   function values at grid points
    Coords: (B, H, W, 2)     normalized grid coordinates in [0, 1]^2
    Output: (B, H, W, d_u)
    """
    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 64,
        n_layers: int = 4,
        d_a: int = 1,
        d_u: int = 1,
        proj_dim: int = None,
        use_norm: bool = True,
    ):
        super().__init__()
        proj_dim = proj_dim or width
        self.lifting = nn.Conv2d(d_a + 2, width, kernel_size=1)
        self.fno_layers = nn.ModuleList([
            FNOLayer2d(width, modes1, modes2, use_norm=use_norm) for _ in range(n_layers)
        ])
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
        v = self.projection(v).permute(0, 2, 3, 1)  # (B, H, W, d_u)
        return v


class FNO1dVelocityField(nn.Module):
    """
    FM adapter: maps flat state x_t and time t to velocity v_theta(x_t, t).

    Interface matches dummy_mlp: forward(x, t) with x,t shapes [B, D] and [B, 1].
    """
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        self.N = model_config.input_dim
        if model_config.output_dim != self.N:
            raise ValueError("FNO1dVelocityField requires input_dim == output_dim.")
        self.register_buffer(
            "coords",
            torch.linspace(0.0, 1.0, self.N).view(1, self.N, 1),
        )
        d_a = 2 if model_config.enable_time else 1
        self.fno = FNO1d(
            modes=model_config.fno_modes,
            width=model_config.fno_width,
            n_layers=model_config.fno_layers,
            d_a=d_a,
            d_u=1,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        B, N = x.shape
        if N != self.N:
            raise ValueError(f"Expected state dimension {self.N}, got {N}.")
        a = x.unsqueeze(-1)
        if t is not None and self.model_config.enable_time:
            t_node = t.view(B, 1, 1).expand(B, N, 1)
            a = torch.cat([a, t_node], dim=-1)
        coords = self.coords.expand(B, -1, -1)
        v = self.fno(a, coords)
        return v.squeeze(-1)


class FNO2dVelocityField(nn.Module):
    """
    FM adapter for 2D grid fields (e.g. sphere theta/phi discretization).

    model_config.fno_modes is a total mode budget; each axis uses sqrt(fno_modes).
    Optional fno_modes2 sets a separate budget for the second axis.

    Interface matches dummy_mlp: forward(x, t) with x,t shapes [B, H*W] and [B, 1].
    """
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        if model_config.grid_shape is None:
            raise ValueError("FNO2dVelocityField requires model_config.grid_shape=(H, W).")
        self.H, self.W = model_config.grid_shape
        grid_size = self.H * self.W
        if model_config.input_dim != grid_size or model_config.output_dim != grid_size:
            raise ValueError(
                f"grid_shape {model_config.grid_shape} implies dim {grid_size}, "
                f"but got input_dim={model_config.input_dim}, output_dim={model_config.output_dim}."
            )

        grid_y = torch.linspace(0.0, 1.0, self.H)
        grid_x = torch.linspace(0.0, 1.0, self.W)
        yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        coords = torch.stack([yy, xx], dim=-1).unsqueeze(0)  # (1, H, W, 2)
        self.register_buffer("coords", coords)

        modes1, modes2 = fno2d_spectral_dims(
            model_config.fno_modes, model_config.fno_modes2
        )
        d_a = 2 if model_config.enable_time else 1
        proj_dim = getattr(model_config, "fno_proj_dim", None)
        use_norm = getattr(model_config, "fno_use_norm", True)
        self.fno = FNO2d(
            modes1=modes1,
            modes2=modes2,
            width=model_config.fno_width,
            n_layers=model_config.fno_layers,
            d_a=d_a,
            d_u=1,
            proj_dim=proj_dim,
            use_norm=use_norm,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        B = x.shape[0]
        a = x.view(B, self.H, self.W).unsqueeze(-1)
        if t is not None and self.model_config.enable_time:
            t_map = t.view(B, 1, 1, 1).expand(B, self.H, self.W, 1)
            a = torch.cat([a, t_map], dim=-1)
        coords = self.coords.expand(B, -1, -1, -1)
        v = self.fno(a, coords)
        return v.squeeze(-1).reshape(B, self.H * self.W)