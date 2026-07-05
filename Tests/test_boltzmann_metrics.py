"""Unit tests for FALCON-style Boltzmann metrics and dihedrals."""

import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Dataset import dihedrals
from Tests.boltzmann_metrics import (
    energy_wasserstein_w2,
    kish_ess,
    snis_log_weights,
    torus_pairwise_cost,
    torus_wasserstein_w2,
)


class TestBoltzmannMetrics(unittest.TestCase):
    def test_torus_distance_identical_is_zero(self):
        angles = np.array([[0.5, 1.0], [1.2, 2.0]])
        cost = torus_pairwise_cost(angles, angles)
        np.testing.assert_allclose(np.diag(cost), 0.0, atol=1e-12)

    def test_torus_distance_periodic_shift_is_zero(self):
        a = np.array([[0.1, 0.2]])
        b = np.array([[0.1 + 2.0 * np.pi, 0.2 + 2.0 * np.pi]])
        cost = torus_pairwise_cost(a, b)
        self.assertAlmostEqual(cost[0, 0], 0.0, places=10)

    def test_torus_wasserstein_identical_is_zero(self):
        angles = np.random.default_rng(0).uniform(0, 2 * np.pi, size=(32, 2))
        w2 = torus_wasserstein_w2(angles, angles)
        self.assertAlmostEqual(w2, 0.0, places=10)

    def test_energy_wasserstein_identical_is_zero(self):
        e = np.linspace(-50, 50, 64)
        w2 = energy_wasserstein_w2(e, e)
        self.assertAlmostEqual(w2, 0.0, places=10)

    def test_kish_ess_uniform_is_one(self):
        log_w = np.zeros(100)
        self.assertAlmostEqual(kish_ess(log_w), 1.0, places=10)

    def test_kish_ess_single_dominant_weight_is_small(self):
        log_w = np.full(100, -1e9)
        log_w[0] = 0.0
        ess = kish_ess(log_w)
        self.assertAlmostEqual(ess, 1.0 / 100.0, places=5)

    def test_snis_log_weights_shape(self):
        e = np.array([1.0, 2.0, 3.0])
        lp = np.array([-1.0, -2.0, -3.0])
        lw = snis_log_weights(e, lp, kbt_kj_mol=2.5)
        self.assertEqual(lw.shape, (3,))

    def test_ad3_dihedrals_shape(self):
        pos = np.random.default_rng(1).normal(size=(8, 22, 3))
        dih = dihedrals.ad3_backbone_dihedrals(pos)
        self.assertEqual(dih.shape, (8, 2))
        self.assertTrue(np.all((dih >= 0.0) & (dih < 2.0 * np.pi)))

    def test_features_to_positions_roundtrip_shape(self):
        feats = np.random.default_rng(2).normal(size=(4, 66))
        pos = dihedrals.features_to_positions(feats)
        self.assertEqual(pos.shape, (4, 22, 3))


class TestLogProb(unittest.TestCase):
    def test_logprob_gaussian_integration_shape(self):
        import torch

        from Flows.config import FlowConfig, ModelConfig
        from Flows.logprob import infer_with_logprob, log_p0_gaussian
        import models.architecture as MA

        config = FlowConfig(num_steps=4)
        model_cfg = ModelConfig(input_dim=4, output_dim=4, intermediate_dim=16)
        model = MA.build_model(model_cfg)
        x0 = torch.randn(8, 4)
        x1, log_p1 = infer_with_logprob(
            model, x0, config, log_p0_gaussian, model_type="mlp", num_steps=4
        )
        self.assertEqual(tuple(x1.shape), (8, 4))
        self.assertEqual(tuple(log_p1.shape), (8,))
        self.assertTrue(torch.isfinite(log_p1).all())

    def test_matern_log_p_finite(self):
        import torch

        from Flows.matern_sampler import log_p_matern_graph, sample_matern_graph

        laplacian = torch.eye(6)
        samples = sample_matern_graph(500, laplacian, nu=2.0, tau=1.0)
        log_p = log_p_matern_graph(samples, laplacian, nu=2.0, tau=1.0)
        self.assertEqual(tuple(log_p.shape), (500,))
        self.assertTrue(torch.isfinite(log_p).all())


def _openmm_available() -> bool:
    try:
        import openmm  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_openmm_available(), "OpenMM not installed")
class TestOpenMMEnergy(unittest.TestCase):
    def test_openmm_energy_crosscheck_smoke(self):
        from Dataset.energy import crosscheck_npz_energies
        from Dataset.molecule_graph import DEFAULT_TEST_NPZ

        result = crosscheck_npz_energies(DEFAULT_TEST_NPZ, n_frames=3, rtol=0.15)
        self.assertIn("max_rel_error", result)
        self.assertTrue(np.isfinite(result["max_rel_error"]))


@unittest.skipUnless(_openmm_available(), "OpenMM not installed")
class TestFalconIntegration(unittest.TestCase):
    def test_falcon_metrics_integration_smoke(self):
        from Tests.molecule_benchmark import benchmark_molecule_ad3

        metrics = benchmark_molecule_ad3(
            model_type="mlp",
            dataset_size=128,
            num_epochs=1,
            batch_size=32,
            test_samples=64,
            num_steps=4,
            falcon_metrics=True,
            snis=True,
            balanced_models=True,
        )
        for key in (
            "t_w2_raw",
            "e_w2_raw",
            "ess",
            "t_w2_reweighted",
            "e_w2_reweighted",
        ):
            self.assertIn(key, metrics)
            self.assertTrue(np.isfinite(metrics[key]))


if __name__ == "__main__":
    unittest.main()
