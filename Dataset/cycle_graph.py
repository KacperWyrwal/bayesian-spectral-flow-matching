import torch
import matplotlib.pyplot as plt

def generate_cycle_graph_laplacian(N):
    """Creates the Laplacian matrix for a cycle graph with N nodes."""
    # Degree matrix is 2 * Identity since every node has 2 neighbors
    D = 2 * torch.eye(N)
    
    # Adjacency matrix for a cycle (1s on the cyclic off-diagonals)
    A = torch.zeros(N, N)
    for i in range(N):
        A[i, (i + 1) % N] = 1
        A[i, (i - 1) % N] = 1
        
    L = D - A
    return L

def generate_band_limited_signals(N, K, s, num_samples=1):
    
    assert K <= N, "K cannot be greater than the number of nodes N"
    
    # 1. Get Laplacians, Eigenvalues (lambda), and Eigenvectors (phi)
    L = generate_cycle_graph_laplacian(N)
    eigenvalues, eigenvectors = torch.linalg.eigh(L) # eigh is optimized for symmetric matrices
    
    # 2. Slice to keep only the first K components
    lambda_K = eigenvalues[:K]        # Shape: (K,)
    phi_K = eigenvectors[:, :K]        # Shape: (N, K)
    
    # 3. Compute the variance (standard deviation) for each coefficient a_i
    
    variances = (1 + lambda_K) ** (-s)
    std_deviations = torch.sqrt(variances) # Shape: (K,)
    
    # 4. Sample standard normal random variables and scale them by the StdDev
    # a_i ~ N(0, std_dev^2)
    epsilon = torch.randn(num_samples, K) # Shape: (num_samples, K)
    a = epsilon * std_deviations           # Broad-casts to scale each column
    
    # 5. Reconstruct the signals: x = Sum(a_i * phi_i) -> Matrix multiplication
    # (num_samples, K) x (N, K)^T -> (num_samples, N)
    x = torch.matmul(a, phi_K.t())
    
    return x, eigenvalues, eigenvectors
if __name__ == "__main__":
    # --- Example Usage ---
    N = 100        # 100 nodes in our ring/cycle
    K = 20         # Only use the 10 lowest-frequency waves
    s = 2.0        # Relatively high decay = very smooth signals
    num_signals = 5

    signals, lambdas, phis = generate_band_limited_signals(N, K, s, num_samples=num_signals)

    print("Generated signals shape:", signals.shape) # Should be (3, 100)
    # Plotting the generated signals to see the smoothness
    plt.figure(figsize=(10, 5))
    node_indices = torch.arange(N)

    for i in range(num_signals):
        # Append the first node to the end just to close the loop visually in the plot
        y_vals = torch.cat([signals[i], signals[i, :1]])
        x_vals = torch.arange(N + 1)
        plt.plot(x_vals, y_vals.numpy(), label=f'Signal {i+1}')

    plt.title(f"Band-Limited Signals on a Cycle Graph (K={K}, s={s})")
    plt.xlabel("Node Index (Wrapped around)")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.show()