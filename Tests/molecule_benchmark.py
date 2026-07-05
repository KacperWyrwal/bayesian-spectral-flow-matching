import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional

import Dataset.molecule_graph as molecule_graph
import Dataset.dihedrals as dihedrals
import Dataset.energy as energy
import Flows.config as config
import models.architecture as MA
from Flows.trainer import Trainer
from Tests.metrics import (
    rmsd_distribution,
    sobolev_wasserstein_distance,
    wasserstein_distance,
)
from Tests.boltzmann_metrics import (
    energy_wasserstein_w2,
    kish_ess,
    resample_weighted,
    snis_log_weights,
    torus_wasserstein_w2,
)

N_ATOMS = molecule_graph.N_ATOMS_AD3
STATE_DIM = molecule_graph.STATE_DIM
GRAPH_FNO_TYPES = {"graph_laplacian_fno", "graph_sobolev_fno"}


def _evaluate_falcon_metrics(
    x_gen_np: np.ndarray,
    x_true_np: np.ndarray,
    pdb_path,
    temperature_k: float = 310.0,
    log_p_gen: Optional[np.ndarray] = None,
    snis: bool = False,
    snis_seed: int = 0,
) -> dict:
    """Compute T-W2, E-W2, and optional SNIS-reweighted metrics."""
    pos_gen = dihedrals.features_to_positions(x_gen_np)
    pos_true = dihedrals.features_to_positions(x_true_np)

    dih_gen = dihedrals.ad3_backbone_dihedrals(pos_gen)
    dih_true = dihedrals.ad3_backbone_dihedrals(pos_true)

    energy_eval = energy.AD3EnergyEvaluator(
        pdb_path=pdb_path, temperature_k=temperature_k
    )
    e_gen = energy_eval.evaluate_potential_energy(pos_gen)
    e_true = energy_eval.evaluate_potential_energy(pos_true)
    kbt = energy.kbt_kj_mol(temperature_k)

    t_w2_raw = torus_wasserstein_w2(dih_gen, dih_true)
    e_w2_raw = energy_wasserstein_w2(e_gen, e_true)
    print(f"T-W2 (raw proposal): {t_w2_raw:.4f}")
    print(f"E-W2 (raw proposal): {e_w2_raw:.4f}")

    result = {
        "t_w2_raw": t_w2_raw,
        "e_w2_raw": e_w2_raw,
    }

    if not snis:
        return result

    if log_p_gen is None:
        raise ValueError("snis=True requires log_p_gen from infer_with_logprob.")

    log_w = snis_log_weights(e_gen, log_p_gen, kbt)
    ess = kish_ess(log_w)
    print(f"SNIS ESS (normalized): {ess:.4f}")
    if ess < 0.01:
        print(
            "Warning: ESS < 0.01 — SNIS reweighted metrics may be unreliable. "
            "Consider more ODE steps or Dopri5 integration."
        )

    x_resampled = resample_weighted(x_gen_np, log_w, n_out=x_gen_np.shape[0], seed=snis_seed)
    pos_resampled = dihedrals.features_to_positions(x_resampled)
    dih_resampled = dihedrals.ad3_backbone_dihedrals(pos_resampled)
    e_resampled = energy_eval.evaluate_potential_energy(pos_resampled)

    t_w2_reweighted = torus_wasserstein_w2(dih_resampled, dih_true)
    e_w2_reweighted = energy_wasserstein_w2(e_resampled, e_true)
    print(f"T-W2 (SNIS-reweighted): {t_w2_reweighted:.4f}")
    print(f"E-W2 (SNIS-reweighted): {e_w2_reweighted:.4f}")

    result.update(
        {
            "ess": ess,
            "t_w2_reweighted": t_w2_reweighted,
            "e_w2_reweighted": e_w2_reweighted,
        }
    )
    return result


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
    falcon_metrics=False,
    snis=False,
    temperature_k: float = 310.0,
    snis_seed: int = 0,
):
    if falcon_metrics and test_samples < 10000:
        test_samples = 10000
        print(f"falcon_metrics=True: using test_samples={test_samples}")

    if snis and not falcon_metrics:
        falcon_metrics = True

    if snis and model_type != "mlp":
        raise NotImplementedError(
            "SNIS benchmarking is currently supported for model_type='mlp' only."
        )

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

    log_p_gen = None
    if snis:
        x_generated, log_p_gen_t, _ = trainer.infer_with_logprob(num_samples=test_samples)
        x_generated = x_generated.detach()
        log_p_gen = log_p_gen_t.detach().cpu().numpy()
    else:
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

    if falcon_metrics:
        falcon = _evaluate_falcon_metrics(
            x_gen_np,
            x_true_np,
            pdb_path=pdb_path,
            temperature_k=temperature_k,
            log_p_gen=log_p_gen,
            snis=snis,
            snis_seed=snis_seed,
        )
        metrics.update(falcon)

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
    falcon_metrics=False,
    snis=False,
    temperature_k: float = 310.0,
    snis_seed: int = 0,
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
        falcon_metrics=falcon_metrics,
        snis=snis,
        temperature_k=temperature_k,
        snis_seed=snis_seed,
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
