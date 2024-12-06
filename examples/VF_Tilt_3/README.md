# README for `examples/VF_tilt_3/` folder

This folder contains an example workflow for generating, modifying, and analyzing vacancy formation energies using the scripts and tools provided in the project. Below is an overview of the folder structure and the steps required to run the example, along with the relevant instructions for each Python script involved in the process.

## Folder Structure

The `examples/VF_tilt_3/` folder contains the necessary data and scripts for simulating and analyzing the vacancy formation energies of atoms in a LAMMPS lattice.

### Key Files in the Folder:
- `trigger/` - Contains required files for running LAMMPS simulations, such as lattice data and interatomic potential files.
- `formation_energy.xlsx` - An Excel file where the formation energies for atoms are stored after running the simulation.
- `lowest_energy_atom.txt` - A text file that records the atom with the lowest formation energy.
- `input_master.lmp` - A master LAMMPS input file used for running simulations on individual atoms.
- `id.txt` - A file containing a list of atom IDs to be used in the simulations.

## Workflow for Vacancy Formation Energy Simulation

### 1. **Augment Data (Optional):**
   If you would like to augment your lattice by randomly removing atoms to simulate vacancy formation, you can use the `inflate_vacancy.py` script.

   - **Command**:  
     ```bash
     python inflate_vacancy.py --input id.txt --output modified_id.txt --remove_percentage 10
     ```
   - **Description**: This script will read the atom IDs from `id.txt` and randomly remove a specified percentage of atoms (e.g., 10%) from the list. The modified atom IDs will be saved to a new file (`modified_id.txt`).

### 2. **Generate Atom IDs:**
   Once the lattice has been augmented or if you have an existing lattice, you can generate the list of atom IDs with the `generate_id.py` script.

   - **Command**:  
     ```bash
     python generate_id.py --lattice_file trigger/my_lattice_prep.data --output id.txt
     ```
   - **Description**: This script will extract the atom IDs from the LAMMPS data file (`my_lattice_prep.data`) and save them into the file `id.txt`.

### 3. **Run LAMMPS Simulations (using `vf.py`):**
   After generating the atom IDs, you can use the `vf.py` script to set up and run LAMMPS simulations for each atom. This will calculate the formation energy for each atom in the lattice.

   - **Command**:  
     ```bash
     python vf.py --id_file id.txt --lattice_file trigger/my_lattice_prep.data --output formation_energy.xlsx --lowest_energy lowest_energy_atom.txt
     ```
   - **Description**: The script will:
     1. **Setup Simulation Directories**: For each atom, create a separate folder and copy the necessary files from the `trigger/` directory.
     2. **Modify LAMMPS Input**: The script modifies the master LAMMPS input file (`input_master.lmp`) to include the specific atom ID and prepare the simulation.
     3. **Run LAMMPS Simulations**: Executes the simulation using the `qsub` job scheduler for each atom.
     4. **Process Results**: After each simulation, the script extracts the total energy from the resulting `Tot_e_2.data` file, computes the formation energy, and stores the result in `formation_energy.xlsx`.
     5. **Track Lowest Formation Energy**: The atom with the lowest formation energy is recorded in `lowest_energy_atom.txt`.

### 4. **Check Results:**
   After the simulation runs are completed, the `formation_energy.xlsx` file will contain the calculated formation energies for each atom, and `lowest_energy_atom.txt` will contain the atom ID with the lowest formation energy.

---

## Training the EGNN Model

Once you have the formation energy data, you can use the provided code to train a model to predict the formation energy based on the atom positions and their interactions.

The training process begins by loading the formation energy data stored in an Excel file. The `formation_energy.xlsx` file contains atom IDs and their corresponding formation energies, which are mapped into a dictionary for easy access during training. 

Next, the script reads the atomic positions from `trigger/my_lattice_prep.data`, which contains the 3D coordinates of the atoms. These positions are used to calculate the pairwise distances between atoms. The script identifies the nearest neighbors for each atom and prepares input features required by the neural network, which include the pairwise difference matrix, the distance matrix, and Radial Basis Function (RBF) features. These features capture the spatial relationships between atoms, which are crucial for accurately predicting the formation energy given a vacancy at a particular atomic position.

The model used for this task is built using the layers from the TFN implementation. This architecture is designed to handle the symmetries inherent in molecular systems, where atomic positions can undergo rotations, translations, and reflections without changing the physical system's energy. Notably, pre-trained weights can also be loaded to perform rudimentary transfer learning. 

Training the model involves using the Adam optimizer and Mean Squared Error (MSE) loss. The model processes the inputs (atomic positions and features) to compute predictions for the formation energies. The loss function measures the discrepancy between the model’s predictions and the true formation energies, and backpropagation is used to adjust the model parameters to minimize this error. During the training process, checkpoints of model weights are saved at a frequency of 10 times throughout the training run. 

---

## Notes:
- **LAMMPS Simulations**: Ensure that the LAMMPS setup on your machine (or HPC cluster) is configured correctly for running simulations with `vf.py`. The `run.sh` script in the `trigger/` folder may need to be adjusted based on your system configuration.
- **Model Training**: The training process can be computationally expensive, so ensure that you have access to a machine with sufficient resources (GPU recommended for neural network training).
- **File Paths**: Make sure to adjust file paths in the scripts if your directory structure differs from the example provided.

---

This README provides an overview of how to run the vacancy formation energy workflow and train the EGNN model using the example data in `examples/VF_tilt_3/`. The scripts and notebook provided allow you to generate atom IDs, run simulations, calculate formation energies, and train a neural network model to predict those energies based on atom positions.
