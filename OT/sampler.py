import torch 
import numpy as np
import ot
from scipy.optimize import linear_sum_assignment

class OTDisplacementMap():
    def __init__(self, config, eigenvalues=None, eigenvectors=None):
        self.config = config
        self.beta = getattr(config, "beta", 1.0)
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

    def set_spectral_basis(self, eigenvalues, eigenvectors):
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

    def _sobolev_cost(self, x0, x1):
        y0 = x0 @ self.eigenvectors
        y1 = x1 @ self.eigenvectors
        weights = (1.0 + self.eigenvalues) ** (-self.beta)
        diff = y0[:, None, :] - y1[None, :, :]
        return torch.sum((diff ** 2) * weights.view(1, 1, -1), dim=-1)

    def pair_minibatch(self, x0, x1):
        """
        Pair x0 and x1 minibatches through balanced OT under Sobolev spectral cost.
        Returns x0 unchanged and x1 permuted to match x0 rows.
        """
        cost = self._sobolev_cost(x0, x1)
        n0, n1 = cost.shape
        a = np.full(n0, 1.0 / n0, dtype=np.float64)
        b = np.full(n1, 1.0 / n1, dtype=np.float64)
        plan = ot.emd(a, b, cost.detach().cpu().numpy().astype(np.float64))
        plan_t = torch.from_numpy(plan).to(cost.device).float()

        if n0 == n1:
            # Use Hungarian assignment on negative transport mass to recover a strict permutation.
            row_ind, col_ind = linear_sum_assignment((-plan_t).detach().cpu().numpy())
            row_order = torch.tensor(row_ind, device=cost.device, dtype=torch.long)
            col_order = torch.tensor(col_ind, device=cost.device, dtype=torch.long)
            x0 = x0[row_order]
            indices = col_order
        else:
            indices = torch.argmax(plan_t, dim=1)
        x1_paired = x1[indices]
        return x0, x1_paired

    def compute_displacement(self, x, t):
        return t * x