import torch

def wasserstein_distance(x, y):
    # Compute the 1D Wasserstein distance between two distributions represented by samples
    x_sorted, _ = torch.sort(x)
    y_sorted, _ = torch.sort(y)
    return torch.mean(torch.abs(x_sorted - y_sorted))