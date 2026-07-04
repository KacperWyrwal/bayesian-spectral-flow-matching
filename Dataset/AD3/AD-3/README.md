# Alanine Dipeptide trajectory data

This folder contains a minimal data set of two long MD trajectories for alanine
dipeptide, the simplest dipeptide (22 atoms).

For each trajectory two files are available:

* `protein-state0.pdb`: contains the topology and initial 3D XYZ coordinates.
* `protein-arrays.npz`: contains trajectory information.

## NPZ Information

The NPZ file contains detailed information for a subset of simulation steps.
There are T such frames and the NPZ file contains the following arrays:

* 'step': `(T,)` array, Md step number. 
* 'energies': `(T,2)` array, each row containing [potential, kinetic] energies
  in kJ/mol.
* 'positions': `(T,num_atoms,3)` array, positions in nm.
* 'velocities': `(T,num_atoms,3)` array, velocities in nm/ps.
* 'forces': `(T,num_atoms,3)` array, forces in kJ/(mol nm).


## Dataset construction

The dataset was constructed in the following way:

1. For the included `alanine-dipeptide.pdb` PDB file, perform a molecular
   dynamics simulation:
   a.) Use OpenMM with the AMBER14 force field and implicit water model.
   b.) Perform an energy minimization (relaxation) from the initial PDB
       configuration.
   c.) Use a Langevin integrator at temperature T=310K, friction=0.3/ps,
       timestep=0.5fs for 2e6 steps to equilibriate ("burn-in phase").
   d.) Use a Langevin integrator at temperature T=310K, friction=0.3/ps,
       timestep=0.5fs for 2e8 steps to sample a trajectory ("sample phase").
2. Save trajectory information every 1,000 steps (0.5ps) to an `arrays.npz`
   file.


## Credit and Authors

This dataset was created in March 2022 as part of the Molecular Simulation
initiative.

