import torch
import torch.nn as nn

class IntegralKernelLayer(nn.Module):
    """
    One layer of a general neural operator.
    Approximates ∫ κ(x, y, a(x), a(y)) v(y) dy
    via a simple attention-style discretization.
    """
    def __init__(self, d_v: int, d_a: int):
        super().__init__()
        # Kernel network: takes (v(x), v(y), a(x), a(y)) → scalar weight
        self.kernel_net = nn.Sequential(
            nn.Linear(2 * d_v + 2 * d_a, 128),
            nn.GELU(),
            nn.Linear(128, d_v * d_v),
        )
        self.W = nn.Linear(d_v, d_v)   # local bypass

    def forward(self, v, a):
        """
        v: (B, N, d_v)   — current hidden function, evaluated at N points
        a: (B, N, d_a)   — input function, evaluated at N points
        returns: (B, N, d_v)
        """
        B, N, dv = v.shape
        da = a.shape[-1]

        # Build pairwise feature: (v(x), v(y), a(x), a(y))
        vx = v.unsqueeze(2).expand(-1, -1, N, -1)    # (B, N, N, dv)
        vy = v.unsqueeze(1).expand(-1, N, -1, -1)
        ax = a.unsqueeze(2).expand(-1, -1, N, -1)    # (B, N, N, da)
        ay = a.unsqueeze(1).expand(-1, N, -1, -1)

        features = torch.cat([vx, vy, ax, ay], dim=-1)  # (B, N, N, 2dv+2da)

        # Evaluate kernel: κ(x, y, a(x), a(y)) as a d_v×d_v matrix
        kappa = self.kernel_net(features)                # (B, N, N, dv*dv)
        kappa = kappa.view(B, N, N, dv, dv)

        # Integrate (sum over y, divide by N for normalization)
        # integral ≈ (1/N) Σ_y κ(x,y) v(y)
        integral = torch.einsum('bxyij,byj->bxi', kappa, v) / N  # (B, N, dv)

        # Local bypass
        local = self.W(v)

        return local + integral


class NeuralOperator(nn.Module):
    """
    General neural operator: P ∘ (L_T ∘ ... ∘ L_1) ∘ Q
    Maps function a → function u via stacked integral kernel layers.
    """
    def __init__(self, d_a: int, d_u: int, d_v: int = 64, n_layers: int = 4):
        super().__init__()
        self.lifting   = nn.Linear(d_a, d_v)   # P: lift input to hidden dim
        self.layers    = nn.ModuleList([
            IntegralKernelLayer(d_v, d_a) for _ in range(n_layers)
        ])
        self.projection = nn.Sequential(        # Q: project hidden to output
            nn.Linear(d_v, 128),
            nn.GELU(),
            nn.Linear(128, d_u),
        )
        self.act = nn.GELU()

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """
        a: (B, N, d_a)
        returns u: (B, N, d_u)
        """
        v = self.lifting(a)          # (B, N, d_v)
        for layer in self.layers:
            v = self.act(layer(v, a))
        return self.projection(v)    # (B, N, d_u)