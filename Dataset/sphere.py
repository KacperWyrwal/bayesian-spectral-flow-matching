"""implement data living on S2 sphere that we can test the spectral method
the data set would ideally be a set of sognals generated on a topological structure with different spectral weighting (sum of sines wave or something)
"""
import argparse
import math
import numpy as np
import torch
from scipy.special import sph_harm
import matplotlib.pyplot as plt
from matplotlib import cm

# PART 1 — SPHERICAL HARMONIC BASIS
def get_lm_pairs(L_max):
    return [(l, m) for l in range(L_max + 1) for m in range(-l, l + 1)]

def get_eigenvalues(lm_pairs):
    return np.array([l * (l + 1) for l, m in lm_pairs], dtype=np.float32)

def real_sph_harm(l, m, theta, phi):
    
    if m == 0:
        return sph_harm(0, l, phi, theta).real.astype(np.float32)
    elif m > 0:
        # R_l^m = sqrt(2) * (-1)^m * Re[Y_l^m]
        return (np.sqrt(2) * (-1)**m * sph_harm(m, l, phi, theta).real).astype(np.float32)
    else:  # m < 0
        # R_l^m = sqrt(2) * (-1)^m * Im[Y_l^|m|]
        return (np.sqrt(2) * (-1)**m * sph_harm(-m, l, phi, theta).imag).astype(np.float32)

def build_synthesis_matrix(L_max, n_theta=64, n_phi=128):
   
    # Gauss-Legendre quadrature points in theta is ideal; regular grid is fine for viz
    theta_vals = np.linspace(1e-4, np.pi - 1e-4, n_theta)
    phi_vals   = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    THETA, PHI = np.meshgrid(theta_vals, phi_vals, indexing='ij')  # (n_theta, n_phi)

    pairs   = get_lm_pairs(L_max)
    n_coeffs = len(pairs)
    n_grid   = n_theta * n_phi

    Y_mat = np.zeros((n_grid, n_coeffs), dtype=np.float32)
    for idx, (l, m) in enumerate(pairs):
        Y_mat[:, idx] = real_sph_harm(l, m, THETA.ravel(), PHI.ravel())

    return Y_mat, THETA, PHI   # Y_mat: (n_grid, n_coeffs)


def generate_grid_laplacian(
    n_theta,
    n_phi,
    neighbor_radius=1,
    metric="geodesic",
    sigma=None,
):
    """
    Build a graph Laplacian on the theta/phi grid with extended neighborhood.
    
    Args:
        n_theta: Number of polar grid points (theta)
        n_phi: Number of azimuthal grid points (phi)
        neighbor_radius: Radius of neighborhood connectivity
                        1: 8-neighborhood (original)
                        2: Extended neighborhood (includes cells up to distance 2)
                        Higher values: even more connected graph
        metric: Distance/weighting metric for edges.
                - "geodesic": great-circle distance on S2 + heat-kernel weights
                - "manhattan": legacy inverse Manhattan weighting on grid offsets
        sigma: Heat-kernel width for geodesic metric. If None, set from median
               local geodesic edge distance.
    
    Each cell is connected to nearby cells based on neighbor_radius:
      - Connections wrap periodically in phi (longitude)
      - No wrap in theta (latitude)
      - Weights depend on chosen metric
    """
    if metric not in {"geodesic", "manhattan"}:
        raise ValueError(f"Unsupported metric '{metric}'. Use 'geodesic' or 'manhattan'.")

    n_nodes = n_theta * n_phi
    adjacency = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)

    def idx(i, j):
        return i * n_phi + j

    edge_pairs = []
    geodesic_distances = []
    xyz_t = None
    if metric == "geodesic":
        theta_vals = np.linspace(1e-4, np.pi - 1e-4, n_theta, dtype=np.float32)
        phi_vals = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False, dtype=np.float32)
        THETA, PHI = np.meshgrid(theta_vals, phi_vals, indexing="ij")
        x = np.sin(THETA) * np.cos(PHI)
        y = np.sin(THETA) * np.sin(PHI)
        z = np.cos(THETA)
        xyz = np.stack([x, y, z], axis=-1).reshape(n_nodes, 3)
        xyz_t = torch.from_numpy(xyz).float()

    for i in range(n_theta):
        for j in range(n_phi):
            u = idx(i, j)

            # Connect to all cells within neighbor_radius
            for di in range(-neighbor_radius, neighbor_radius + 1):
                for dj in range(-neighbor_radius, neighbor_radius + 1):
                    if di == 0 and dj == 0:
                        continue  # Skip self
                    
                    ni = i + di
                    nj = (j + dj) % n_phi  # Periodic in phi
                    
                    # No wrap in theta (latitude)
                    if 0 <= ni < n_theta:
                        v = idx(ni, nj)
                        if metric == "manhattan":
                            # Legacy inverse Manhattan weighting.
                            distance = abs(di) + abs(dj)
                            weight = 1.0 / distance
                            adjacency[u, v] = max(adjacency[u, v], weight)
                        else:
                            edge_pairs.append((u, v))
                            dot = float(torch.dot(xyz_t[u], xyz_t[v]).item())
                            dot = max(-1.0, min(1.0, dot))
                            geodesic_distances.append(math.acos(dot))

    if metric == "geodesic":
        print("Computing geodesic Laplacian...")
        if not geodesic_distances:
            raise RuntimeError("No edges were generated for geodesic Laplacian.")
        distances = np.array(geodesic_distances, dtype=np.float32)
        sigma_eff = float(np.median(distances)) if sigma is None else float(sigma)
        sigma_eff = max(sigma_eff, 1e-6)

        for (u, v), d in zip(edge_pairs, distances):
            weight = float(np.exp(-(d * d) / (2.0 * sigma_eff * sigma_eff)))
            adjacency[u, v] = max(adjacency[u, v], weight)

    # Ensure exact symmetry
    adjacency = torch.maximum(adjacency, adjacency.T)
    degree = torch.diag(adjacency.sum(dim=1))
    return degree - adjacency

# PART 2 — DATA GENERATION  (entirely in spectral domain)

def spectral_std_true(lm_pairs, s=2.0, l0_var=1.0):
    """
    Per-coefficient std: σ_lm = (l(l+1))^{-s/2}  for l > 0
                               = sqrt(l0_var)       for l = 0
    Controls how fast power falls off with degree l.
    s=2 → very smooth;  s=0.5 → rougher
    """
    std = np.zeros(len(lm_pairs), dtype=np.float32)
    for idx, (l, m) in enumerate(lm_pairs):
        if l == 0:
            std[idx] = np.sqrt(l0_var)
        else:
            std[idx] = (l * (l + 1)) ** (-s / 2)
    return std

def generate_coeff(N, lm_pairs, s=2.0, l0_var=1.0):
    std = spectral_std_true(lm_pairs, s, l0_var)
    return np.random.randn(N, len(lm_pairs)).astype(np.float32) * std[None, :]

# let's try to vizualize some of these signals to sanity check
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sphere signal visualization')
    parser.add_argument('--L_max', type=int, default=16, help='Maximum spherical harmonic degree')
    parser.add_argument('--n_theta', type=int, default=40, help='Number of polar grid points (theta)')
    parser.add_argument('--n_phi', type=int, default=80, help='Number of azimuthal grid points (phi)')
    parser.add_argument('--high_res', action='store_true', help='Use higher resolution grid')
    args = parser.parse_args()

    if args.high_res:
        args.n_theta = 80
        args.n_phi = 160

    lm_pairs = get_lm_pairs(args.L_max)
    Y_mat, THETA, PHI = build_synthesis_matrix(args.L_max, n_theta=args.n_theta, n_phi=args.n_phi)
    coeffs = generate_coeff(1, lm_pairs, s=2.0)[0]
    signal = Y_mat @ coeffs

    x = np.sin(THETA) * np.cos(PHI)
    y = np.sin(THETA) * np.sin(PHI)
    z = np.cos(THETA)

    signal_grid = signal.reshape(args.n_theta, args.n_phi)
    values = (signal_grid - signal_grid.min()) / (signal_grid.max() - signal_grid.min())
    colors = cm.viridis(values)

    # Close the seam at phi=0/2π by repeating the first column at the end.
    x = np.concatenate([x, x[:, :1]], axis=1)
    y = np.concatenate([y, y[:, :1]], axis=1)
    z = np.concatenate([z, z[:, :1]], axis=1)
    colors = np.concatenate([colors, colors[:, :1, :]], axis=1)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
        x,
        y,
        z,
        facecolors=colors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    plt.title('Sphere signal visualization')
    plt.tight_layout()
    plt.show()