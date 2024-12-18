# MACH - M6 Atomic Computation Hub

Welcome to the M6 Atomic Computation Hub (MACH). This repository provides tools for simulating vacancy formation energies in atomic lattices using LAMMPS simulations and machine learning models, specifically an Edge-Conditioned Graph Neural Network (EGNN).

## Overview

MACH facilitates the calculation of vacancy formation energies for atoms in a lattice and uses a neural network model (EGNN) to predict these energies based on atomic positions and interactions. The core of the workflow involves:
1. **Data Collection**: Generating atom IDs, simulating vacancy formation, and extracting formation energies.
2. **LAMMPS Simulations**: Running LAMMPS simulations to model vacancies and calculate formation energies.
3. **EGNN Model Training**: Using the collected data to train an EGNN model that predicts formation energies from atomic data.

---

## Installation

### Conda Environment Setup

To ensure that all dependencies are compatible, create and activate a **conda** environment:

1. Create a new environment with Python 3.9:
   - `conda create --name mach python=3.9`
   
2. Activate the environment:
   - `conda activate mach`

### Install Python Dependencies

Once the environment is activated, install the required dependencies using **pip**:
   - `pip install torch pandas scipy matplotlib numpy openpyxl`
   
---

## Data Collection Workflow

Data collection involves generating atom IDs, simulating vacancies, and calculating formation energies. The process can be broken down into three main steps:

1. **Generate Atom IDs**: Use the `generate_id.py` script to extract atom IDs from a LAMMPS data file. This will create a file `id.txt` containing the atom IDs needed for the next steps.

2. **Augment Data (Optional)**: If you'd like to simulate vacancies by removing a specific percentage of atoms, use the `inflate_vacancy.py` script to randomly remove atom IDs from `id.txt`. This step is optional but allows for augmented vacancy data.

3. **Run LAMMPS Simulations**: Use the `vf.py` script to run LAMMPS simulations for each atom. This script will generate formation energies by performing calculations for atoms identified in `id.txt`. The results will be saved in an Excel file (`formation_energy.xlsx`), with the atom having the lowest formation energy also recorded in `lowest_energy_atom.txt`.

---

## EGNN Model Training

Once you have the formation energy data from the simulations, you can proceed with training the EGNN model. 

### Key Steps:

1. **Load Data**: The `formation_energy.xlsx` file contains the formation energies for all atoms. You will load this data and also extract atomic positions from the LAMMPS lattice file (`my_lattice_prep.data`).

2. **Prepare Input Features**: For each atom, generate input features for the EGNN, including pairwise atomic distances and radial basis function (RBF) values. These features describe the atomic interactions and will serve as the input to the model.

3. **Define the EGNN Model**: The EGNN model is based on the **TensorField-Torch** implementation. It uses a series of layers, including convolutional and self-interaction layers, to process the atomic data. The model predicts the formation energy in the case of a vacancy at a given atomic position. Notably, users can load pre-trained weights to perform rudimentary transfer learning. 

4. **Training the Model**: Once the input features are ready, you can train the EGNN model. The model will learn to predict the formation energies of atoms based on their positions and interactions with other atoms.

---

## GitHub Repositories Used / Referenced

1. **TensorField-Torch**:
   - GitHub: [https://github.com/semodi/tensorfield-torch](https://github.com/semodi/tensorfield-torch)
   - This repository provides a PyTorch implementation of Tensor Field Networks (TFNs), which we use for the EGNN model in this project. The repository defines the layers as classes, making it easy to stack and experiment with different network architectures. Modifications were made to the original implementation found at the linked github repository to enable the use of the CUDA architecture to make use of any available NVidia GPU's. 

2. **TensorFieldNetworks**:
   - GitHub: [https://github.com/tensorfieldnetworks/tensorfieldnetworks](https://github.com/tensorfieldnetworks/tensorfieldnetworks)
   - The original implementation of TFN, which inspired the TensorField-Torch repository. Although we initially considered using this version, we opted for the more flexible PyTorch-based implementation provided by the TensorField-Torch repository due to its concise, class-based design.

---
