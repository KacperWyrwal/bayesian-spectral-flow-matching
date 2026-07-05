"""
this is a dummy dataset for testing the code, it is a mixture of 3 gaussians in 1D space
"""

import torch
import torch.distributions as dist
import matplotlib.pyplot as plt

def sample_mog(num_samples):
    # Define the parameters of the mixture of Gaussians
    means = torch.tensor([-5.0, 0.0, 5.0])  # Means of the Gaussians
    stds = torch.tensor([1.0, 1.0, 1.0])     # Standard deviations of the Gaussians
    weights = torch.tensor([0.3, 0.4, 0.3])   # Weights of the Gaussians

    # Create a categorical distribution to sample which Gaussian to use
    component_distribution = dist.Normal(loc=means, scale=stds)
    categorical = dist.Categorical(weights)
    mog = dist.MixtureSameFamily(
        mixture_distribution=categorical,
        component_distribution=component_distribution
    )
    samples = mog.sample((num_samples,))
    return samples

if __name__ == "__main__":
    samples = sample_mog(100000)
    plt.hist(samples.numpy(), bins=1000, density=True)
    plt.title("Mixture of Gaussians")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.show()
