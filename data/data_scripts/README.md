# generate_id.py

This Python script extracts atom IDs from a LAMMPS data file and writes them to an output text file.

## Input

- **File**: `trigger/my_lattice_prep.data`
- **Description**: A LAMMPS data file containing atomic data. The script expects the file to have an "Atoms # atomic" section, where each atom is listed with an ID.
- **Format**: The script reads the file line by line, starting from the "Atoms # atomic" section, and extracts the first column (atom ID) from each line.

## Output

- **File**: `id.txt`
- **Description**: A text file containing the extracted atom IDs, one per line. The file is saved in the same directory as the script.

# inflate_vacancy.py

This Python script randomly removes a specified percentage of atom IDs from a list and updates a LAMMPS lattice file by removing corresponding atoms.

## Input

- **File**: `id.txt`
  - **Description**: A text file containing a list of atom IDs, one per line. These IDs are the ones that will be removed.
  - **Format**: Each line contains a single atom ID (integer).

- **File**: `my_lattice_prep.data`
  - **Description**: A LAMMPS lattice file containing atomic data, including an "Atoms # atomic" section.
  - **Format**: The file should contain a list of atoms, each with an atom ID. The script will remove atoms based on IDs from `id.txt`.

## Output

- **File**: `trigger/modified_lattice_prep.data`
  - **Description**: A modified version of the original lattice file with the specified atoms removed. The number of atoms will be updated accordingly.

- **File**: `removed_ids.txt`
  - **Description**: A text file containing the atom IDs that were removed, one per line.

## Functionality

1. **Remove IDs from `id.txt`**: The script removes a specified percentage of IDs from the file (`id.txt`). Alternatively, you can specify a separate file containing the list of IDs to remove.
2. **Update Lattice File**: The script then updates the LAMMPS lattice file (`my_lattice_prep.data`) by removing atoms with the removed IDs and adjusts the total atom count accordingly.
3. **File Renaming**: The script renames the original lattice file and saves the modified version as the new lattice file.
4. **Multiple Input Types** if the command is run with a percentage, the script will remove a certain percentage of atoms from the lattice. However, it can also be run with a path to a txt file of ids. This was done in an effort to improve reproducibility. 

## Notes

- **Percentage Removal**: The script allows you to specify a percentage of IDs to randomly remove from `id.txt`. Alternatively, you can provide a separate file with specific IDs to remove.
- **File Structure**: The script assumes the LAMMPS data file follows the standard format with the "Atoms # atomic" section.

# vf.py

This Python script automates the process of setting up and running multiple LAMMPS simulations for atoms identified from `id.txt`, calculating the formation energy for each atom, and saving the results in an Excel file.

## Input

- **File**: `id.txt`
  - **Description**: A text file containing a list of atom IDs, one per line. These IDs represent the atoms to be simulated in the LAMMPS setup.
  - **Format**: Each line contains a single atom ID (integer).

- **File**: `input_master.lmp`
  - **Description**: A master input file for LAMMPS simulations. The script will modify this file for each atom by adding a group ID for the atom.
  - **Format**: Standard LAMMPS input format, with a placeholder `#Add_group_id` where the atom group is to be added.

- **Directory**: `trigger/`
  - **Description**: A folder containing files necessary for the simulation. The script will copy these files into newly created subfolders for each atom.
  
## Output

- **File**: `formation_energy.xlsx`
  - **Description**: An Excel file containing the calculated formation energies for all simulated atoms. Each row corresponds to one atom with its formation energy.

- **File**: `lowest_energy_atom.txt`
  - **Description**: A text file that stores the ID and formation energy of the atom with the lowest formation energy.

## Functionality

1. **Setup Simulation Directories**: 
   - The script reads atom IDs from `id.txt` and creates a simulation subfolder (`sim_{atom_id}`) for each atom.
   - It copies necessary files from the `trigger/` folder into the subfolder.

2. **Modify LAMMPS Input**: 
   - For each atom, the script prepares a LAMMPS input file (`inp.lmp`) by modifying the `input_master.lmp` file. The atom ID is added as a group ID in the input file.

3. **Run LAMMPS Simulations**:
   - The script runs the LAMMPS simulation using `qsub run.sh` in each atom's simulation folder.

4. **Process Simulation Results**:
   - After each simulation, the script waits for the `Tot_e_2.data` file to appear, extracts the total energy from this file, and calculates the formation energy.
   - The formation energy is computed using the formula:  
     `formation_energy = -total_energy - ((N - n) * Efe)`,  
     where `N`, `n`, and `Efe` are predefined constants.

5. **Store and Export Results**:
   - The formation energy for each atom is stored in a dictionary (`dict_final_result`).
   - The results are saved in an Excel file (`formation_energy.xlsx`) and the atom with the lowest formation energy is saved to `lowest_energy_atom.txt`.

## Notes

- The script uses the `qsub` command to submit the LAMMPS simulation jobs, which implies that it's intended to run in a job scheduler environment (e.g., HPC cluster with PBS/Torque).
- Ensure that the `trigger/` folder contains the necessary files for the LAMMPS simulations and that `input_master.lmp` is properly formatted.
- The script automatically cleans up temporary files (`Tot_e_2.data`, `inp.lmp`) after processing each atom.


# Example Data Collection Workflow 

1. An example of the file structure for data collection is the `examples/VF_Tilt_3/` folder. Note that `vf.py` requires the list of atom IDs from `id.txt`.
2. Thus, if you would like to augment the data using `inflate_vacancy.py`, you must run this first, following the documentation above.
3. Then, you must run the `generate_id.py` script before running `vf.py` since you need to generate `id.txt`. 
4. For each atom, `vf.py` creates a simulation folder, copies necessary files from `trigger/`, and prepares the input file (`inp.lmp`).
5. It runs the LAMMPS simulation for each atom.
6. Once the simulation finishes, the script reads the total energy from `Tot_e_2.data` and calculates the formation energy for each atom.
7. The formation energies are stored in an Excel file (`formation_energy.xlsx`). The atom with the lowest formation energy is recorded in `lowest_energy_atom.txt`.