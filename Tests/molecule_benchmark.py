import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import torch

import Dataset.molecule_graph as molecule_graph
import Flows.config as config
import models.architecture as MA
from Flows.trainer import Trainer
from Tests.metrics import (
    rmsd_distribution,
    sobolev_wasserstein_distance,
    wasserstein_distance,
)

N_ATOMS = molecule_graph.N_ATOMS_AD3
STATE_DIM = molecule_graph.STATE_DIM
GRAPH_FNO_TYPES = {"graph_laplacian_fno", "graph_sobolev_fno"}


def benchmark_molecule_ad3(
    plot=False,
    use_sobolev=False,
    feature_mode="com_relative",
    alpha=-1.0,
    beta=2.0,
    nu=3.0,
    tau=2.0,
    model_type="mlp",
    balanced_models=True,
    fno_light=False,
    fno_modes=22,
    fno_width=32,
    fno_layers=2,
    spectral_loss_modes=66,
    max_frames=8192,
    dataset_size=4096,
    num_epochs=100,
    batch_size=128,
    test_samples=1024,
    num_steps=10,
    checkpoint_path=None,
    from_checkpoint=False,
    train_npz=None,
    test_npz=None,
    pdb_path=None,
    graph_type="bond",
    heat_sigma=None,
    heat_cutoff_nm=None,
):
    pdb_path = pdb_path or molecule_graph.DEFAULT_TRAIN_PDB
    train_npz = train_npz or molecule_graph.DEFAULT_TRAIN_NPZ
    test_npz = test_npz or molecule_graph.DEFAULT_TEST_NPZ

    _, laplacian, edges = molecule_graph.build_ad3_laplacian(
        pdb_path,
        graph_type=graph_type,
        heat_sigma=heat_sigma,
        heat_cutoff_nm=heat_cutoff_nm,
    )
    print(
        f"AD3 {graph_type} graph: {len(edges)} edges, "
        f"state dim {STATE_DIM}, L shape {tuple(laplacian.shape)}"
    )

    train_sampler = molecule_graph.AD3TrajectorySampler(
        npz_path=train_npz,
        pdb_path=pdb_path,
        feature_mode=feature_mode,
        max_frames=max_frames,
    )
    eval_sampler = molecule_graph.AD3TrajectorySampler(
        npz_path=test_npz,
        pdb_path=pdb_path,
        feature_mode=feature_mode,
        max_frames=max_frames,
        seed=1,
    )

    def target_sampler(num_samples, output_dim, device):
        return train_sampler(num_samples, output_dim, device)

    if checkpoint_path is None:
        tag = "sobolev" if use_sobolev else "standard"
        model_tag = "" if model_type == "mlp" else f"_{model_type}"
        feat_tag = "" if feature_mode == "com_relative" else f"_{feature_mode}"
        graph_tag = "" if graph_type == "bond" else f"_{graph_type}"
        checkpoint_path = (
            f"checkpoints/benchmark_molecule_{tag}{model_tag}{feat_tag}{graph_tag}.pth"
        )

    if from_checkpoint:
        trainer = Trainer.from_checkpoint(
            checkpoint_path,
            target_sampler=target_sampler,
        )
    else:
        if balanced_models:
            model_cfg = MA.balanced_ad3_model_config(
                STATE_DIM,
                laplacian,
                model_type=model_type,
                sobolev_alpha=alpha,
            )
        elif model_type in GRAPH_FNO_TYPES and fno_light:
            model_cfg = MA.light_graph_fno_config(
                STATE_DIM,
                laplacian,
                model_type=model_type,
                sobolev_alpha=alpha,
            )
        elif model_type in GRAPH_FNO_TYPES:
            model_cfg = config.ModelConfig(
                input_dim=STATE_DIM,
                output_dim=STATE_DIM,
                intermediate_dim=256,
                model_type=model_type,
                laplacian=laplacian,
                spectral_modes=spectral_loss_modes or fno_modes,
                sobolev_alpha=alpha,
                fno_modes=fno_modes,
                fno_width=fno_width,
                fno_layers=fno_layers,
            )
        elif model_type == "fno1d":
            model_cfg = config.ModelConfig(
                input_dim=STATE_DIM,
                output_dim=STATE_DIM,
                intermediate_dim=256,
                model_type="fno1d",
                fno_modes=fno_modes,
                fno_width=fno_width,
                fno_layers=fno_layers,
            )
        else:
            model_cfg = config.ModelConfig(
                input_dim=STATE_DIM,
                output_dim=STATE_DIM,
                intermediate_dim=512,
                model_type=model_type,
            )

        trainer = Trainer(
            flow_config=config.FlowConfig(
                num_steps=num_steps,
                lr=5e-5,
                sigma_min=0.001,
                sigma_max=1.0,
                use_spectral=use_sobolev,
                use_sobolev_ot=False,
                use_matern_source=use_sobolev,
                alpha=alpha,
                beta=beta,
                nu=nu,
                tau=tau,
                laplacian=laplacian if use_sobolev else None,
                spectral_loss_modes=spectral_loss_modes if use_sobolev else None,
                spectral_loss_basis="laplacian" if use_sobolev else "laplacian",
            ),
            model_config=model_cfg,
            dataset_size=dataset_size,
            num_epochs=num_epochs,
            batch_size=batch_size,
            checkpoint_path=checkpoint_path,
            target_sampler=target_sampler,
        )
        trainer.train()
        trainer.save_checkpoint(checkpoint_path)

    x_generated = trainer.infer(num_samples=test_samples)
    x_generated = x_generated[-1] if x_generated.dim() == 3 else x_generated
    x_true = eval_sampler(test_samples, STATE_DIM, trainer.flow_config.device)

    x_gen_np = x_generated.detach().cpu().numpy()
    x_true_np = x_true.detach().cpu().numpy()

    w2_eucl = wasserstein_distance(x_true_np, x_gen_np)
    print(f"Euclidean Wasserstein distance: {w2_eucl:.4f}")

    metrics = {"w2_euclidean": w2_eucl}

    if use_sobolev and trainer.cfm._eigenvalues is not None:
        evals = trainer.cfm._eigenvalues.detach().cpu().numpy()
        evecs = trainer.cfm._eigenvectors.detach().cpu().numpy()
        w2_sob = sobolev_wasserstein_distance(
            x_true_np, x_gen_np, evals, evecs, beta=beta
        )
        print(f"Sobolev spectral Wasserstein distance (beta={beta}): {w2_sob:.4f}")
        metrics["w2_sobolev"] = w2_sob

    rmsd_stats = rmsd_distribution(x_gen_np, x_true_np, n_atoms=N_ATOMS)
    print(
        f"COM-aligned RMSD (nm): mean={rmsd_stats['mean']:.4f}, "
        f"std={rmsd_stats['std']:.4f}"
    )
    metrics["rmsd"] = rmsd_stats

    if plot:
        _plot_conformers(x_gen_np, x_true_np, title="Generated vs true AD3 conformers")

    return metrics


def _plot_conformers(x_gen, x_true, title="Conformers", n_show=2):
    fig = plt.figure(figsize=(10, 4))
    for col, (label, batch) in enumerate((("Generated", x_gen), ("True", x_true))):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        for i in range(min(n_show, batch.shape[0])):
            coords = batch[i].reshape(N_ATOMS, 3)
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=20, label=f"sample {i}")
        ax.set_title(label)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_zlabel("z (nm)")
        if col == 1:
            ax.legend(fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def compare_models_molecule(
    plot=False,
    use_sobolev=False,
    feature_mode="com_relative",
    alpha=-1.0,
    beta=2.0,
    nu=3.0,
    tau=2.0,
    balanced_models=True,
    fno_modes=22,
    fno_width=32,
    fno_layers=2,
    spectral_loss_modes=66,
    max_frames=8192,
    dataset_size=4096,
    num_epochs=100,
    batch_size=128,
    test_samples=1024,
    num_steps=10,
    from_checkpoint=False,
    graph_type="bond",
    heat_sigma=None,
    heat_cutoff_nm=None,
):
    """Compare mlp / fno1d / graph_laplacian_fno / graph_sobolev_fno on AD3.

    Set use_sobolev=True for spectral FM with Matern source; False for standard FM.
    """
    pdb_path = molecule_graph.DEFAULT_TRAIN_PDB
    _, laplacian, _ = molecule_graph.build_ad3_laplacian(
        pdb_path,
        graph_type=graph_type,
        heat_sigma=heat_sigma,
        heat_cutoff_nm=heat_cutoff_nm,
    )
    if balanced_models:
        MA.print_ad3_balanced_param_counts(STATE_DIM, laplacian, sobolev_alpha=alpha)

    common = dict(
        plot=plot,
        use_sobolev=use_sobolev,
        feature_mode=feature_mode,
        alpha=alpha,
        beta=beta,
        nu=nu,
        tau=tau,
        balanced_models=balanced_models,
        fno_modes=fno_modes,
        fno_width=fno_width,
        fno_layers=fno_layers,
        spectral_loss_modes=spectral_loss_modes,
        max_frames=max_frames,
        dataset_size=dataset_size,
        num_epochs=num_epochs,
        batch_size=batch_size,
        test_samples=test_samples,
        num_steps=num_steps,
        from_checkpoint=from_checkpoint,
        graph_type=graph_type,
        heat_sigma=heat_sigma,
        heat_cutoff_nm=heat_cutoff_nm,
    )
    model_types = ("mlp", "fno1d", "graph_laplacian_fno", "graph_sobolev_fno")
    results = {}
    for model_type in model_types:
        print(f"\n--- AD3 molecule benchmark: {model_type} ---")
        results[model_type] = benchmark_molecule_ad3(model_type=model_type, **common)
        w2 = results[model_type]["w2_euclidean"]
        print(f"{model_type} Euclidean W2: {w2:.4f}")
    return results


if __name__ == "__main__":
    print(
        "Comparing AD3 architectures (mlp / fno1d / graph FNO variants), "
        "spectral FM with heat-kernel graph..."
    )
    results = compare_models_molecule(
        use_sobolev=False,
        graph_type="heat",
        num_epochs=1000,
        dataset_size=16384,
        batch_size=1024,
        test_samples=2048,
        num_steps=8,
    )
    for model_type, metrics in results.items():
        w2 = metrics["w2_euclidean"]
        w2_sob = metrics.get("w2_sobolev")
        rmsd_mean = metrics["rmsd"]["mean"]
        line = f"  {model_type}: W2={w2:.4f}, RMSD={rmsd_mean:.4f}"
        if w2_sob is not None:
            line += f", Sobolev W2={w2_sob:.4f}"
        print(line)
