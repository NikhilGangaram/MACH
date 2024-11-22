# File Structure

The project contains the following structure:

- `input_master.lmp` - This is the master LAMMPS input file.
- `trigger/` - A folder containing:
  - `FeH_Wen.eam.fs` - Source code for an interatomic potential.
  - `run.sh` - A bash script used for running the simulation on the computer server. **Note**: The script is currently constrained to `hpcs87` for the purposes of this project but can be updated inside `run.sh` for use on other systems.
  - `my_lattice_prep.data` - A file containing lattice information.

### Important Notes:

- **Lattice Structure**: If you need to change your input lattice structure, remember to update the master LAMMPS input file (`input_master.lmp`).
- **Simulation Preparation**: To run the simulation, you must generate the `id.txt` file. This file is needed for identifying atoms in the simulations.

---

# Example Data Collection Workflow

1. An example of the file structure for data collection is the `examples/VF_Tilt_3/` folder. Note that `vf.py` requires the list of atom IDs from `id.txt`.
2. If you want to augment the data using `inflate_vacancy.py`, you must run this script first, following the documentation provided above.
3. Then, run the `generate_id.py` script before running `vf.py` to generate the `id.txt` file.
4. For each atom, `vf.py` will:
   - Create a simulation folder.
   - Copy necessary files from the `trigger/` folder.
   - Prepare the input file (`inp.lmp`).
5. The script then runs the LAMMPS simulation for each atom.
6. Once the simulation is finished, it reads the total energy from `Tot_e_2.data` and calculates the formation energy for each atom.
7. The formation energies are stored in an Excel file (`formation_energy.xlsx`).
8. The atom with the lowest formation energy is recorded in `lowest_energy_atom.txt`.
