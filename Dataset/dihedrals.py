"""Backbone dihedral angles for AD3 alanine dipeptide."""

from __future__ import annotations

import numpy as np

from Dataset.molecule_graph import FEAT_DIM, N_ATOMS_AD3

# 0-indexed atom indices after parse_pdb_atoms ordering.
AD3_PHI_ATOMS = (4, 6, 8, 14)   # ACE C, ALA N, ALA CA, ALA C
AD3_PSI_ATOMS = (6, 8, 14, 16)  # ALA N, ALA CA, ALA C, NME N


def features_to_positions(
    features: np.ndarray,
    n_atoms: int = N_ATOMS_AD3,
    feat_dim: int = FEAT_DIM,
) -> np.ndarray:
    """(B, n_atoms * feat_dim) COM-relative features -> (B, n_atoms, 3) positions in nm."""
    feats = np.asarray(features, dtype=np.float64)
    if feats.ndim == 1:
        feats = feats[np.newaxis, ...]
    expected = n_atoms * feat_dim
    if feats.shape[-1] != expected:
        raise ValueError(f"Expected feature dim {expected}, got {feats.shape[-1]}")
    return feats.reshape(feats.shape[0], n_atoms, feat_dim)


def compute_dihedral(
    positions_nm: np.ndarray,
    i: int,
    j: int,
    k: int,
    l: int,
) -> np.ndarray:
    """
    IUPAC dihedral for atom quadruple (i, j, k, l).

    positions_nm: (B, n_atoms, 3) or (n_atoms, 3)
  Returns angles in [0, 2*pi).
    """
    pos = np.asarray(positions_nm, dtype=np.float64)
    if pos.ndim == 2:
        pos = pos[np.newaxis, ...]

    b1 = pos[:, j, :] - pos[:, i, :]
    b2 = pos[:, k, :] - pos[:, j, :]
    b3 = pos[:, l, :] - pos[:, k, :]

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1_norm = np.linalg.norm(n1, axis=-1, keepdims=True)
    n2_norm = np.linalg.norm(n2, axis=-1, keepdims=True)
    n1 = n1 / np.maximum(n1_norm, 1e-12)
    n2 = n2 / np.maximum(n2_norm, 1e-12)

    b2_unit = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-12)
    m1 = np.cross(n1, b2_unit)

    x = np.sum(n1 * n2, axis=-1)
    y = np.sum(m1 * n2, axis=-1)
    angles = np.arctan2(y, x)
    return np.mod(angles, 2.0 * np.pi)


def ad3_backbone_dihedrals(positions_nm: np.ndarray) -> np.ndarray:
    """Return (B, 2) array of [phi, psi] in radians."""
    phi = compute_dihedral(positions_nm, *AD3_PHI_ATOMS)
    psi = compute_dihedral(positions_nm, *AD3_PSI_ATOMS)
    return np.stack([phi, psi], axis=-1)
