import torch 
import numpy as np 
import Dataset.MoG as MoG

""""
we need to construct conditional prob path and velocity field
"""
class ConditionalFM():
    def __init__(self, model, config):
        self.model = model
        self.config = config 
        self.sigma_min = config.sigma_min
        self.sigma_max = config.sigma_max
    #we'll start with simple linear path aka optimal transport conditional VF
    def compute_sigma(self, t):
        return 1-(self.sigma_max-self.sigma_min)*t
    def compute_mu(self, x ,t):
        return t*x
    def compute_velocity(self, x,x1, t):
        sigma = self.compute_sigma(t)
        return (x1-(self.sigma_max-self.sigma_min)* x) / sigma
    def flow(self, x, x1, t):
        sigma = self.compute_sigma(t)
        mu = self.compute_mu(x1, t)
        return mu + x * sigma
    def sample_from_cond_path(self, x0, x1, t):
        return  self.flow(x0, x1, t)
    def compute_loss(self, x0,x1):
        # sample uniform t, from dataset and from cond prob path and compute loss
        batch_size = x0.shape[0]
        t = torch.rand(batch_size, device=self.config.device).unsqueeze(1)
        
        # sample from path 
        xt = self.sample_from_cond_path(x0, x1, t)
        score = self.model(xt,t)
        # general form of conditional velocity target
        target=self.compute_velocity(xt, x1, t)
        # in the case of linear interpolation
        #target = (x1-x0) (where sigma min 0 and sigma max 1)
        #mse CFM loss
        loss = torch.mean((score - target)**2)
        return loss
    def get_path(self):
        pass
    def step():
        pass
    def preprocess_batch():
        pass

class SpectralFM(ConditionalFM):
    def __init__(self, model, config):
        super(SpectralFM, self).__init__(model, config)
        self.alpha = config.alpha
        self._spectral_basis = getattr(config, "spectral_loss_basis", "laplacian")
        self._grid_shape = getattr(config, "grid_shape", None)
        self._eigenvalues = None
        self._eigenvectors = None
        self._spectral_weights = None
        self._build_spectral_basis()

    def _sobolev_weights(self, eigenvalues):
        return (1.0 + eigenvalues) ** self.alpha

    def _build_laplacian_basis(self):
        laplacian = getattr(self.config, "laplacian", None)
        if laplacian is None:
            raise ValueError(
                "spectral_loss_basis='laplacian' requires FlowConfig.laplacian."
            )
        laplacian = laplacian.to(self.config.device).float()
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        k = getattr(self.config, "spectral_loss_modes", None)
        if k is not None:
            k = min(int(k), eigenvectors.shape[1])
            eigenvalues = eigenvalues[:k]
            eigenvectors = eigenvectors[:, :k]
        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors
        self._spectral_weights = self._sobolev_weights(eigenvalues)

    def _build_fourier_basis(self):
        device = self.config.device
        k = getattr(self.config, "spectral_loss_modes", None)

        if self._grid_shape is not None:
            H, W = self._grid_shape
            k1 = torch.arange(H, device=device, dtype=torch.float32)
            k1 = torch.where(k1 <= H // 2, k1, k1 - H)
            k2 = torch.arange(W // 2 + 1, device=device, dtype=torch.float32)
            K1, K2 = torch.meshgrid(k1, k2, indexing="ij")
            eigenvalues = (
                4.0
                - 2.0 * torch.cos(2.0 * torch.pi * K1 / H)
                - 2.0 * torch.cos(2.0 * torch.pi * K2 / W)
            )
            weights = self._sobolev_weights(eigenvalues)
            if k is not None:
                m1 = min(H, int(k))
                m2 = min(W // 2 + 1, int(k))
                mask = torch.zeros(H, W // 2 + 1, device=device)
                mask[:m1, :m2] = 1.0
                weights = weights * mask
            self._spectral_weights = weights
            return

        # 1D fallback (cycle graph): DFT basis with cycle-graph Laplacian eigenvalues.
        laplacian = getattr(self.config, "laplacian", None)
        if laplacian is None:
            raise ValueError(
                "spectral_loss_basis='fourier' requires grid_shape or laplacian."
            )
        n = laplacian.shape[0]
        k1 = torch.arange(n // 2 + 1, device=device, dtype=torch.float32)
        eigenvalues = 2.0 - 2.0 * torch.cos(2.0 * torch.pi * k1 / n)
        weights = self._sobolev_weights(eigenvalues)
        if k is not None:
            m = min(n // 2 + 1, int(k))
            mask = torch.zeros(n // 2 + 1, device=device)
            mask[:m] = 1.0
            weights = weights * mask
        self._spectral_weights = weights

    def _build_spectral_basis(self):
        if self._spectral_basis == "fourier":
            self._build_fourier_basis()
        else:
            self._build_laplacian_basis()

    def compute_sigma(self, t):
        return super().compute_sigma(t)

    def compute_mu(self, x ,t):
        return t * x

    def sample_from_cond_path(self, x0, x1, t):
        # kappa = 0 path: x_t = (1-t)x0 + t x1 + sigma_t eps
        sigma = self.compute_sigma(t)
        eps = torch.randn_like(x0)
        xt = (1 - t) * x0 + t * x1 + sigma * eps
        return xt, eps

    def compute_velocity(self, x0, x1, t, eps):
        dsigma_dt = -(self.sigma_max - self.sigma_min)
        return (x1 - x0) + dsigma_dt * eps

    def compute_loss(self, x0, x1):
        
        batch_size = x0.shape[0]
        t = torch.rand(batch_size, device=self.config.device).unsqueeze(1)

        xt, eps = self.sample_from_cond_path(x0, x1, t)
        pred_velocity = self.model(xt, t)
        target_velocity = self.compute_velocity(x0, x1, t, eps)
        residual = pred_velocity - target_velocity

        if self._spectral_basis == "fourier":
            return self._fourier_weighted_loss(residual)
        residual_hat = residual @ self._eigenvectors
        weighted_sq = (residual_hat ** 2) * self._spectral_weights.unsqueeze(0)
        return weighted_sq.sum(dim=1).mean()

    def _fourier_weighted_loss(self, residual):
        weights = self._spectral_weights
        if self._grid_shape is not None:
            H, W = self._grid_shape
            residual_grid = residual.view(residual.shape[0], H, W)
            residual_hat = torch.fft.rfft2(residual_grid, norm="ortho")
            weighted_sq = (residual_hat.abs() ** 2) * weights.unsqueeze(0)
            return weighted_sq.sum(dim=(-2, -1)).mean()
        residual_hat = torch.fft.rfft(residual, norm="ortho")
        weighted_sq = (residual_hat.abs() ** 2) * weights.unsqueeze(0)
        return weighted_sq.sum(dim=1).mean()