"""OpenMM potential-energy evaluation for AD3 alanine dipeptide."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import numpy as np

from Dataset.molecule_graph import DEFAULT_TRAIN_PDB, N_ATOMS_AD3

# kB in kJ/(mol·K); AD3 README uses T=310 K.
KBT_310K_KJ_MOL = 8.314462618e-3 * 310.0

_OPENMM_IMPORT_ERROR: Optional[Exception] = None
try:
    import openmm
    import openmm.app as app
    import openmm.unit as unit
except ImportError as exc:
    _OPENMM_IMPORT_ERROR = exc


def kbt_kj_mol(temperature_k: float = 310.0) -> float:
    return 8.314462618e-3 * temperature_k


def _require_openmm() -> None:
    if _OPENMM_IMPORT_ERROR is not None:
        raise ImportError(
            "OpenMM is required for energy evaluation. "
            "Install with: pip install openmm"
        ) from _OPENMM_IMPORT_ERROR


@lru_cache(maxsize=4)
def _build_context(pdb_path: str, temperature_k: float):
    """Cached OpenMM context for repeated energy evaluations."""
    _require_openmm()

    pdb = app.PDBFile(pdb_path)
    forcefield = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
    )
    integrator = openmm.LangevinMiddleIntegrator(
        temperature_k * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picoseconds,
    )
    platform = openmm.Platform.getPlatformByName("CPU")
    context = openmm.Context(system, integrator, platform)
    return context, pdb.topology.getNumAtoms()


class AD3EnergyEvaluator:
    """
    Evaluate potential energies (kJ/mol) for AD3 conformers via OpenMM.

    Expects positions in nm with shape (B, 22, 3), COM-centered or absolute
    (energy is translation-invariant).
    """

    def __init__(
        self,
        pdb_path: Union[str, Path] = DEFAULT_TRAIN_PDB,
        temperature_k: float = 310.0,
    ):
        self.pdb_path = str(Path(pdb_path).resolve())
        self.temperature_k = temperature_k
        self._context = None
        self._n_atoms = None

    def _ensure_context(self) -> None:
        if self._context is None:
            self._context, self._n_atoms = _build_context(
                self.pdb_path, self.temperature_k
            )

    def evaluate_potential_energy(
        self,
        positions_nm: np.ndarray,
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Args:
            positions_nm: (B, n_atoms, 3) or (n_atoms, 3) in nm.
        Returns:
            (B,) potential energies in kJ/mol.
        """
        self._ensure_context()
        pos = np.asarray(positions_nm, dtype=np.float64)
        if pos.ndim == 2:
            pos = pos[np.newaxis, ...]
        if pos.shape[1] != N_ATOMS_AD3 or pos.shape[2] != 3:
            raise ValueError(
                f"Expected positions (B, {N_ATOMS_AD3}, 3), got {pos.shape}"
            )

        energies = []
        for start in range(0, pos.shape[0], batch_size):
            batch = pos[start : start + batch_size]
            for frame in batch:
                self._context.setPositions(
                    frame * unit.nanometer
                )
                state = self._context.getState(getEnergy=True)
                pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                energies.append(pe)
        return np.asarray(energies, dtype=np.float64)


def evaluate_potential_energy(
    positions_nm: np.ndarray,
    pdb_path: Union[str, Path] = DEFAULT_TRAIN_PDB,
    temperature_k: float = 310.0,
    batch_size: int = 64,
) -> np.ndarray:
    """Convenience wrapper using a module-level evaluator cache."""
    evaluator = AD3EnergyEvaluator(pdb_path=pdb_path, temperature_k=temperature_k)
    return evaluator.evaluate_potential_energy(positions_nm, batch_size=batch_size)


def crosscheck_npz_energies(
    npz_path: Union[str, Path],
    pdb_path: Union[str, Path] = DEFAULT_TRAIN_PDB,
    n_frames: int = 5,
    rtol: float = 0.05,
) -> dict:
    """
    Compare OpenMM energies against stored NPZ potential energies for sanity checks.

    Returns dict with max/mean relative error over sampled frames.
    """
    data = np.load(npz_path)
    positions = data["positions"][:n_frames].astype(np.float64)
    ref = data["energies"][:n_frames, 0].astype(np.float64)
    pred = evaluate_potential_energy(positions, pdb_path=pdb_path)
    rel_err = np.abs(pred - ref) / np.maximum(np.abs(ref), 1e-6)
    return {
        "max_rel_error": float(rel_err.max()),
        "mean_rel_error": float(rel_err.mean()),
        "within_rtol": bool(rel_err.max() <= rtol),
    }
