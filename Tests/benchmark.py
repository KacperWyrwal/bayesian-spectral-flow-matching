import os, sys; 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch 
import Dataset.cycle_graph as cycle_graph
import Flows.config as config
from Flows.trainer import Trainer
from Tests.metrics import wasserstein_distance
import matplotlib.pyplot as plt

def benchmark_graph_cycle(plot=False):
    N = 100
    K = 10
    s = 2.0
    
    #num_signals = 10
    # signals, lambdas, phis = cycle_graph.generate_band_limited_signals(N, K, s, num_samples=num_signals)
    # print("Generated signals shape:", signals.shape) # Should be (3, 100)
    trainer = Trainer(
        flow_config=config.FlowConfig(num_steps=5, lr=5e-5, sigma_min=0.001, sigma_max=1.0),
        model_config=config.ModelConfig(input_dim=N, output_dim=N, intermediate_dim=1024),
        dataset_size=64,
        num_epochs=1,
        batch_size=32,
        target_sampler=lambda num_samples, output_dim, device: cycle_graph.generate_band_limited_signals(output_dim, K, s, num_samples=num_samples)[0].to(device),
        source_sampler=lambda num_samples, input_dim, device: torch.randn(num_samples, input_dim, device=device)
        )
    #trainer.train()
    x_generated = trainer.infer(num_samples=64)
    # Extract final time step: [T, B, D] -> [B, D]
    x_generated = x_generated[-1] if x_generated.dim() == 3 else x_generated
    x_true, _, _ = cycle_graph.generate_band_limited_signals(N, K, s, num_samples=64)
    distance= wasserstein_distance(x_generated.detach().cpu().numpy(), x_true.cpu().numpy())
    print(f"Wasserstein distance between generated and true signals: {distance}")
    
    if plot:
        plt.figure(figsize=(10, 5))

        for i in range(5):
            y_vals_gen = torch.cat([x_generated[i], x_generated[i, :1]])
            x_vals_gen = torch.arange(N + 1)
            plt.plot(x_vals_gen, y_vals_gen.numpy(), label=f'Generated Signal {i+1}', linestyle='--')

            y_vals_true = torch.cat([x_true[i], x_true[i, :1]])
            x_vals_true = torch.arange(N + 1)
            plt.plot(x_vals_true, y_vals_true.numpy(), label=f'True Signal {i+1}')

        plt.title(f'Generated vs True Signals (Wasserstein Distance: {distance:.4f})')
        plt.xlabel('Node Index')
        plt.ylabel('Signal Value')
        plt.legend()
        plt.show()
    return distance

if __name__ == "__main__":
    benchmark_graph_cycle(plot=True)

     