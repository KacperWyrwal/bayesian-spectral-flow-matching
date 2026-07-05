"""Alanine dipeptide (AD-3) bond-graph utilities and trajectory feature loader."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

N_ATOMS_AD3 = 22
FEAT_DIM = 3
STATE_DIM = N_ATOMS_AD3 * FEAT_DIM

AD3_ROOT = Path(__file__).parent / "AD3" / "AD-3"
DEFAULT_TRAIN_PDB = AD3_ROOT / "train" / "ad1-traj-state0.pdb"
DEFAULT_TRAIN_NPZ = AD3_ROOT / "train" / "ad1-traj-arrays.npz"
DEFAULT_TEST_NPZ = AD3_ROOT / "test" / "ad2-traj-arrays.npz"

# Covalent radii in nm (approximate, for bond inference).
_COV_RADII_NM = {
    "H": 0.031,
    "C": 0.076,
    "N": 0.071,
    "O": 0.066,
    "S": 0.105,
    "P": 0.107,
}
_BOND_TOLERANCE_NM = 0.045

# Hardcoded fallback bonds for AD3 (0-indexed pairs), if distance heuristic fails.
_AD3_FALLBACK_BONDS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (1, 3), (1, 4), (4, 5), (4, 6),
    (6, 7), (6, 8), (8, 9), (8, 10), (8, 14), (14, 15),
    (10, 11), (10, 12), (10, 13), (14, 16), (16, 17), (16, 18),
    (18, 19), (18, 20), (18, 21),
]


def _element_from_name(atom_name: str) -> str:
    name = atom_name.strip()
    if not name:
        return "C"
    if name[0].isdigit():
        name = name.lstrip("0123456789")
    element = name[0].upper()
    if len(name) > 1 and name[1].islower():
        element += name[1].lower()
    return element if element in _COV_RADII_NM else element[0]


def parse_pdb_atoms(pdb_path: Union[str, Path]) -> Tuple[np.ndarray, List[str]]:
    """
    Parse ATOM/HETATM records from a PDB file.

    Returns:
        positions_nm: (n_atoms, 3) coordinates in nm
        atom_names: element-like names per atom (length n_atoms)
    """
    pdb_path = Path(pdb_path)
    serials: List[int] = []
    names: List[str] = []
    coords_ang: List[List[float]] = []

    with pdb_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            serial = int(line[6:11])
            atom_name = line[12:16].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            serials.append(serial)
            names.append(atom_name)
            coords_ang.append([x, y, z])

    if not serials:
        raise ValueError(f"No atoms found in PDB: {pdb_path}")

    order = np.argsort(serials)
    positions_nm = np.asarray(coords_ang, dtype=np.float64)[order] * 0.1
    atom_names = [names[i] for i in order]
    return positions_nm, atom_names


def parse_pdb_conect(pdb_path: Union[str, Path]) -> List[Tuple[int, int]]:
    """Parse CONECT records; returns 0-indexed undirected bond pairs."""
    pdb_path = Path(pdb_path)
    bonds: set[Tuple[int, int]] = set()

    with pdb_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("CONECT"):
                continue
            fields = [int(x) for x in line.split()[1:]]
            src = fields[0] - 1
            for dst_serial in fields[1:]:
                dst = dst_serial - 1
                if src == dst:
                    continue
                bonds.add((min(src, dst), max(src, dst)))
    return sorted(bonds)


def infer_bonds_distance(
    positions_nm: np.ndarray,
    atom_names: Sequence[str],
    tolerance_nm: float = _BOND_TOLERANCE_NM,
) -> List[Tuple[int, int]]:
    """Infer covalent bonds from interatomic distances."""
    n_atoms = positions_nm.shape[0]
    radii = np.array(
        [_COV_RADII_NM.get(_element_from_name(name), 0.076) for name in atom_names],
        dtype=np.float64,
    )
    bonds: set[Tuple[int, int]] = set()
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(positions_nm[i] - positions_nm[j])
            cutoff = radii[i] + radii[j] + tolerance_nm
            if dist <= cutoff:
                bonds.add((i, j))
    return sorted(bonds)


def infer_bonds(
    positions_nm: np.ndarray,
    atom_names: Sequence[str],
    pdb_path: Optional[Union[str, Path]] = None,
) -> List[Tuple[int, int]]:
    """Merge CONECT (if PDB given) with distance-based bond inference."""
    bonds: set[Tuple[int, int]] = set(infer_bonds_distance(positions_nm, atom_names))
    if pdb_path is not None:
        bonds.update(parse_pdb_conect(pdb_path))

    bond_list = sorted(bonds)
    if len(bond_list) < 18:
        bond_list = _AD3_FALLBACK_BONDS
    return bond_list


def build_graph_laplacian(n_atoms: int, bonds: Iterable[Tuple[int, int]]) -> torch.Tensor:
    """Unnormalized graph Laplacian L = D - A from unweighted bonds."""
    adj = torch.zeros(n_atoms, n_atoms)
    for i, j in bonds:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    return build_weighted_graph_laplacian(adj)


def build_weighted_graph_laplacian(adjacency: torch.Tensor) -> torch.Tensor:
    """Unnormalized graph Laplacian L = D - W for a symmetric weight matrix."""
    adjacency = 0.5 * (adjacency + adjacency.T)
    degree = torch.diag(adjacency.sum(dim=1))
    return degree - adjacency


def build_heat_kernel_adjacency(
    positions_nm: np.ndarray,
    sigma: Optional[float] = None,
    cutoff_nm: Optional[float] = None,
) -> torch.Tensor:
    """
    Symmetric heat-kernel adjacency from 3D interatomic distances.

    w_ij = exp(-d_ij^2 / (2 sigma^2)), with sigma defaulting to the median
    distance among included pairs (same heuristic as Dataset.sphere geodesic L).
    """
    pos = np.asarray(positions_nm, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions_nm must be (n_atoms, 3), got {pos.shape}")

    dists = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    mask = dists > 0.0
    if cutoff_nm is not None:
        mask &= dists <= float(cutoff_nm)

    pair_dists = dists[mask]
    if pair_dists.size == 0:
        raise RuntimeError("No edges were generated for heat-kernel graph.")

    sigma_eff = float(np.median(pair_dists)) if sigma is None else float(sigma)
    sigma_eff = max(sigma_eff, 1e-6)

    weights = np.exp(-(dists ** 2) / (2.0 * sigma_eff ** 2))
    np.fill_diagonal(weights, 0.0)
    if cutoff_nm is not None:
        weights *= mask.astype(np.float64)

    adjacency = torch.from_numpy(weights.astype(np.float32))
    return torch.maximum(adjacency, adjacency.T)


def _adjacency_to_edge_list(
    adjacency: torch.Tensor,
    min_weight: float = 1e-8,
) -> List[Tuple[int, int]]:
    """Return undirected edges with weight > min_weight (for logging)."""
    adj = adjacency.detach().cpu().numpy()
    n = adj.shape[0]
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > min_weight:
                edges.append((i, j))
    return edges


def build_feature_laplacian(
    laplacian_22: torch.Tensor,
    feat_dim: int = FEAT_DIM,
) -> torch.Tensor:
    """Build D-dimensional feature Laplacian via Kronecker product L ⊗ I_feat."""
    eye = torch.eye(feat_dim, dtype=laplacian_22.dtype, device=laplacian_22.device)
    return torch.kron(laplacian_22, eye)


def center_of_mass(positions: np.ndarray) -> np.ndarray:
    """positions: (..., n_atoms, 3) -> COM (..., 3)."""
    return positions.mean(axis=-2, keepdims=True)


def positions_to_features(
    positions_nm: np.ndarray,
    feature_mode: str = "com_relative",
    reference_nm: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert (B, n_atoms, 3) positions to flat (B, 66) feature vectors.

    feature_mode:
        - "com_relative": subtract per-frame center of mass
        - "ref_displacement": subtract reference structure (n_atoms, 3)
    """
    if positions_nm.ndim == 2:
        positions_nm = positions_nm[np.newaxis, ...]

    if feature_mode == "com_relative":
        feats = positions_nm - center_of_mass(positions_nm)
    elif feature_mode == "ref_displacement":
        if reference_nm is None:
            raise ValueError("reference_nm is required for ref_displacement mode.")
        feats = positions_nm - reference_nm[np.newaxis, ...]
    else:
        raise ValueError(
            f"Unknown feature_mode '{feature_mode}'. "
            "Expected 'com_relative' or 'ref_displacement'."
        )
    return feats.reshape(feats.shape[0], -1)


def build_ad3_laplacian(
    pdb_path: Union[str, Path] = DEFAULT_TRAIN_PDB,
    graph_type: str = "bond",
    heat_sigma: Optional[float] = None,
    heat_cutoff_nm: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """
    Build L_22 and L_66 for AD3 from PDB topology.

    Args:
        pdb_path: Reference structure for topology / 3D coordinates.
        graph_type: "bond" (covalent graph) or "heat" (heat-kernel on 3D distances).
        heat_sigma: Heat-kernel width in nm; median pair distance if None.
        heat_cutoff_nm: Optional distance cutoff in nm for heat-kernel edges.

    Returns:
        L22, L66, edges — edges are bonds for graph_type="bond", or weighted
        heat-kernel pairs (weight > 1e-8) for graph_type="heat".
    """
    if graph_type not in {"bond", "heat"}:
        raise ValueError(
            f"Unsupported graph_type '{graph_type}'. Expected 'bond' or 'heat'."
        )

    positions_nm, atom_names = parse_pdb_atoms(pdb_path)
    if positions_nm.shape[0] != N_ATOMS_AD3:
        raise ValueError(
            f"Expected {N_ATOMS_AD3} atoms, got {positions_nm.shape[0]} from {pdb_path}"
        )

    if graph_type == "bond":
        bonds = infer_bonds(positions_nm, atom_names, pdb_path=pdb_path)
        L22 = build_graph_laplacian(N_ATOMS_AD3, bonds)
        edges = bonds
    else:
        adjacency = build_heat_kernel_adjacency(
            positions_nm,
            sigma=heat_sigma,
            cutoff_nm=heat_cutoff_nm,
        )
        L22 = build_weighted_graph_laplacian(adjacency)
        edges = _adjacency_to_edge_list(adjacency)

    L66 = build_feature_laplacian(L22)
    return L22, L66, edges


class AD3TrajectorySampler:
    """Lazy AD-3 trajectory loader for Trainer target sampling."""

    def __init__(
        self,
        npz_path: Union[str, Path] = DEFAULT_TRAIN_NPZ,
        pdb_path: Union[str, Path] = DEFAULT_TRAIN_PDB,
        feature_mode: str = "com_relative",
        max_frames: Optional[int] = 8192,
        stride: int = 1,
        seed: int = 0,
    ):
        self.npz_path = Path(npz_path)
        self.pdb_path = Path(pdb_path)
        self.feature_mode = feature_mode
        self.max_frames = max_frames
        self.stride = stride
        self.seed = seed

        self._positions: Optional[np.ndarray] = None
        self._reference_nm: Optional[np.ndarray] = None
        self._frame_indices: Optional[np.ndarray] = None

    def _load(self) -> None:
        if self._positions is not None:
            return
        data = np.load(self.npz_path)
        positions = data["positions"].astype(np.float64)
        n_total = positions.shape[0]
        indices = np.arange(0, n_total, self.stride)
        if self.max_frames is not None and len(indices) > self.max_frames:
            rng = np.random.default_rng(self.seed)
            indices = np.sort(rng.choice(indices, size=self.max_frames, replace=False))
        self._positions = positions[indices]
        self._frame_indices = indices
        if self.feature_mode == "ref_displacement":
            ref_pos, _ = parse_pdb_atoms(self.pdb_path)
            self._reference_nm = ref_pos

    @property
    def n_available_frames(self) -> int:
        self._load()
        return int(self._positions.shape[0])

    def sample_frames(self, num_samples: int) -> np.ndarray:
        """Return positions (num_samples, n_atoms, 3) in nm."""
        self._load()
        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, self._positions.shape[0], size=num_samples)
        return self._positions[idx]

    def __call__(self, num_samples: int, output_dim: int, device) -> torch.Tensor:
        if output_dim != STATE_DIM:
            raise ValueError(
                f"AD3 molecule state dim is {STATE_DIM}, got output_dim={output_dim}"
            )
        frames = self.sample_frames(num_samples)
        feats = positions_to_features(
            frames,
            feature_mode=self.feature_mode,
            reference_nm=self._reference_nm,
        )
        return torch.from_numpy(feats).float().to(device)
