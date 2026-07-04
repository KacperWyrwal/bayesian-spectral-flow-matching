import os, sys; 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from matplotlib import cm
import torch 
import Dataset.cycle_graph as cycle_graph
import Dataset.sphere as sphere
import Flows.config as config
import models.architecture as MA
from Flows.trainer import Trainer
from Tests.metrics import wasserstein_distance
import matplotlib.pyplot as plt
import numpy as np

def benchmark_graph_cycle(
    plot=False,
    use_sobolev=False,
    alpha=1.0,
    beta=1.0,
    nu=2.0,
    tau=1.0,
    checkpoint_path=None,
    from_checkpoint=False,
):
    N = 100
    K = 10
    s = 2.0
    laplacian = cycle_graph.generate_cycle_graph_laplacian(N)
    
    #num_signals = 10
    # signals, lambdas, phis = cycle_graph.generate_band_limited_signals(N, K, s, num_samples=num_signals)
    # print("Generated signals shape:", signals.shape) # Should be (3, 100)
    source_sampler = lambda num_samples, input_dim, device: torch.randn(
        num_samples, input_dim, device=device
    )
    target_sampler = lambda num_samples, output_dim, device: cycle_graph.generate_band_limited_signals(
        output_dim, K, s, num_samples=num_samples
    )[0].to(device)

    if checkpoint_path is None:
        tag = "sobolev" if use_sobolev else "standard"
        checkpoint_path = f"checkpoints/benchmark_cycle_{tag}.pth"

    if from_checkpoint:
        trainer = Trainer.from_checkpoint(
            checkpoint_path,
            source_sampler=source_sampler,
            target_sampler=target_sampler,
        )
    else:
        trainer = Trainer(
            flow_config=config.FlowConfig(
                num_steps=20,
                lr=5e-5,
                sigma_min=0.001,
                sigma_max=1.0,
                use_spectral=use_sobolev,
                use_sobolev_ot=use_sobolev,
                use_matern_source=False,
                alpha=alpha,
                beta=beta,
                nu=nu,
                tau=tau,
                laplacian=laplacian if use_sobolev else None,
            ),
            model_config=config.ModelConfig(input_dim=N, output_dim=N, intermediate_dim=1024),
            dataset_size=50000,
            num_epochs=100,
            batch_size=128,
            checkpoint_path=checkpoint_path,
            target_sampler=target_sampler,
            source_sampler=source_sampler,
        )
        trainer.train()
        trainer.save_checkpoint(checkpoint_path)
    x_generated = trainer.infer(num_samples=64)

    # Extract final time step: [T, B, D] -> [B, D]
    x_generated = x_generated[-1] if x_generated.dim() == 3 else x_generated
    x_true, _, _ = cycle_graph.generate_band_limited_signals(N, K, s, num_samples=64)
    distance= wasserstein_distance(x_generated.detach().cpu().numpy(), x_true.cpu().numpy())
    print(f"Wasserstein distance between generated and true signals: {distance}")
    
    if plot:
        plt.figure(figsize=(10, 5))

        for i in range(3):
            y_vals_gen = torch.cat([x_generated.detach().cpu()[i], x_generated.detach().cpu()[i, :1]])
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

def benchmark_s2_signals(
    plot=False,
    use_sobolev=False,
    alpha=1.0,
    beta=1.0,
    nu=2.0,
    tau=1.0,
    L_max=5,
    checkpoint_path=None,
    from_checkpoint=False,
    model_type="mlp",
    balanced_models=True,
    fno_light=False,
    fno_modes=64,
    fno_modes2=64,
    fno_width=32,
    fno_layers=2,
    spectral_loss_modes=64,
    spectral_loss_basis="laplacian",
):
    
    test_samples= 256

    #L_max = 5
    n_theta = 20
    n_phi = 40

    lm_pairs = sphere.get_lm_pairs(L_max)
    n_grid = n_theta * n_phi

    # compute spherical harmonic once to reuse for both sampling and visualization
    Y_mat, THETA, PHI = sphere.build_synthesis_matrix(L_max, n_theta=n_theta, n_phi=n_phi)
    Y_mat_torch = torch.from_numpy(Y_mat).float()
    laplacian = sphere.generate_grid_laplacian(n_theta, n_phi,metric="geodesic")

    def target_sampler(num_samples, output_dim, device):
        assert output_dim == n_grid, "Output dimension must match the number of grid cells"
        coeffs = sphere.generate_coeff(num_samples, lm_pairs, s=2.0)
        coeffs_t = torch.from_numpy(coeffs).float()
        grid_signals = coeffs_t @ Y_mat_torch.T
        return grid_signals.to(device)

    manifold_types = {"laplacian_fno", "sobolev_fno"}

    if checkpoint_path is None:
        tag = "sobolev" if use_sobolev else "standard"
        model_tag = "" if model_type == "mlp" else f"_{model_type}"
        basis_tag = "" if spectral_loss_basis == "laplacian" else f"_{spectral_loss_basis}"
        checkpoint_path = f"checkpoints/benchmark_s2_{tag}{model_tag}{basis_tag}.pth"

    if from_checkpoint:
        trainer = Trainer.from_checkpoint(
            checkpoint_path,
            target_sampler=target_sampler,
        )
    else:
        if balanced_models:
            model_cfg = MA.balanced_s2_model_config(
                n_theta,
                n_phi,
                laplacian,
                model_type=model_type,
                sobolev_alpha=alpha,
            )
        elif model_type == "fno2d" and fno_light:
            model_cfg = MA.light_fno2d_config(n_theta, n_phi)
        elif model_type in manifold_types and fno_light:
            model_cfg = MA.light_manifold_fno_config(
                n_theta,
                n_phi,
                laplacian,
                model_type=model_type,
                sobolev_alpha=alpha,
            )
        elif model_type in manifold_types:
            model_cfg = config.ModelConfig(
                input_dim=n_grid,
                output_dim=n_grid,
                intermediate_dim=256,
                model_type=model_type,
                grid_shape=(n_theta, n_phi),
                laplacian=laplacian,
                spectral_modes=spectral_loss_modes or fno_modes,
                sobolev_alpha=alpha,
                fno_modes=fno_modes,
                fno_width=fno_width,
                fno_layers=fno_layers,
            )
        else:
            model_cfg = config.ModelConfig(
                input_dim=n_grid,
                output_dim=n_grid,
                intermediate_dim=256,
                model_type=model_type,
                grid_shape=(n_theta, n_phi) if model_type == "fno2d" else None,
                fno_modes=fno_modes,
                fno_modes2=fno_modes2,
                fno_width=fno_width,
                fno_layers=fno_layers,
            )

        trainer = Trainer(
            flow_config=config.FlowConfig(
                num_steps=4,
                lr=5e-5,
                sigma_min=0.001,
                sigma_max=1.0,
                use_spectral=use_sobolev,
                use_sobolev_ot=False,
                use_matern_source=use_sobolev,  # Use Matern source if using Sobolev OT
                alpha=alpha,
                beta=beta,
                nu=nu,
                tau=tau,
                laplacian=laplacian if use_sobolev else None,
                spectral_loss_modes=spectral_loss_modes if use_sobolev else None,
                spectral_loss_basis=spectral_loss_basis if use_sobolev else "laplacian",
                grid_shape=(n_theta, n_phi) if use_sobolev else None,
            ),
            model_config=model_cfg,
            dataset_size=4096,
            num_epochs=1000,
            batch_size=256,
            checkpoint_path=checkpoint_path,
            target_sampler=target_sampler,
        )
        trainer.train()
        trainer.save_checkpoint(checkpoint_path)
    x_generated = trainer.infer(num_samples=test_samples)

    # Extract final time step: [T, B, D] -> [B, D]
    x_generated_grid = x_generated[-1] if x_generated.dim() == 3 else x_generated
    x_true_grid = target_sampler(test_samples, n_grid, trainer.flow_config.device)
    distance = wasserstein_distance(x_generated_grid.detach().cpu().numpy(), x_true_grid.detach().cpu().numpy())
    print(f"Wasserstein distance between generated and true signals: {distance}")
    
    if plot:
        x = np.sin(THETA) * np.cos(PHI)
        y = np.sin(THETA) * np.sin(PHI)
        z = np.cos(THETA)
        
        signal_grid = x_generated_grid[0].detach().cpu().numpy().reshape(n_theta, n_phi)
        denom = max(signal_grid.max() - signal_grid.min(), 1e-16)
        values = (signal_grid - signal_grid.min()) / denom
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
    return distance

def compare_manifold_fno_s2(
    plot=False,
    L_max=5,
    alpha=-1,
    beta=2.0,
    nu=3.0,
    tau=2.0,
    balanced_models=True,
    fno_modes=64,
    fno_width=32,
    fno_layers=2,
    spectral_loss_modes=800,
    from_checkpoint=False,
):
    """Compare mlp / fno2d / laplacian_fno / sobolev_fno on the S2 benchmark."""
    n_theta = 20
    n_phi = 40
    laplacian = sphere.generate_grid_laplacian(n_theta, n_phi, metric="geodesic")
    if balanced_models:
        MA.print_s2_balanced_param_counts(n_theta, n_phi, laplacian, sobolev_alpha=alpha)

    common = dict(
        plot=plot,
        use_sobolev=False,
        L_max=L_max,
        alpha=alpha,
        beta=beta,
        nu=nu,
        tau=tau,
        balanced_models=balanced_models,
        fno_modes=fno_modes,
        spectral_loss_basis="laplacian",
        fno_width=fno_width,
        fno_layers=fno_layers,
        spectral_loss_modes=spectral_loss_modes,
        from_checkpoint=from_checkpoint,
    )
    results = {}
    for model_type in ("mlp", "fno2d", "laplacian_fno", "sobolev_fno"):
        print(f"\n--- S2 benchmark: {model_type} ---")
        results[model_type] = benchmark_s2_signals(model_type=model_type, **common)
        print(f"{model_type} Wasserstein: {results[model_type]:.4f}")
    return results


if __name__ == "__main__":

    #print("Running Sobolev spectral FM benchmark on cycle graph...")
    #d_sob = benchmark_graph_cycle(plot=True, use_sobolev=True, alpha=1.0, beta=1.0, nu=2.0, tau=1.0)
    #print(f"Sobolev spectral FM Wasserstein distance: {d_sob:.4f}")
    
    print("Comparing S2 architectures (mlp / fno2d / manifold FNO variants)...")
    results = compare_manifold_fno_s2(
        plot=False,
        L_max=5,
        alpha=-1,
        beta=2.0,
        nu=3.0,
        tau=2.0,
    )
    for model_type, distance in results.items():
        print(f"  {model_type}: {distance:.4f}")
    

     