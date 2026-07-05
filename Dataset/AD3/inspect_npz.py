"""Quick peek at an AD3 trajectory .npz file."""
import numpy as np
from pathlib import Path

NPZ = Path(__file__).parent / "AD-3" / "train" / "ad1-traj-arrays.npz"
data = np.load(NPZ)
print(f"File: {NPZ.name}")
print(f"Keys: {list(data.keys())}")
print(f"Shapes: { {k: v.shape for k, v in data.items()} }")
print(f"Frames: {data['positions'].shape[0]}, atoms: {data['positions'].shape[1]}\n")
for frame in range(3):
    FRAME = frame  # change to inspect another frame

    print(f"--- frame {FRAME} ---")
    print(f"step:      {data['step'][FRAME]}")
    print(f"time (ps): {data['time'][FRAME]:.2f}")
    print(f"energies:  potential={data['energies'][FRAME, 0]:.2f}, kinetic={data['energies'][FRAME, 1]:.2f} kJ/mol")
    print(f"positions (nm), atom 0: {data['positions'][FRAME, 0]}")
    print(f"velocities (nm/ps), atom 0: {data['velocities'][FRAME, 0]}")
    print(f"forces (kJ/mol/nm), atom 0: {data['forces'][FRAME, 0]}")
