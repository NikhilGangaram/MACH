import os

# Define the paths for input and output files
input_file = 'trigger/my_lattice_prep.data'
output_file = 'id.txt'

# List to hold atom IDs
atom_ids = []

# Flag to identify the start of the Atoms section
start_atoms = False

# Open the input LAMMPS data file
with open(input_file, 'r') as f:
    lines = f.readlines()

# Iterate through lines to extract atom IDs
for line in lines:
    # Check for the "Atoms # atomic" line to start processing
    if 'Atoms # atomic' in line:
        start_atoms = True
        continue  # Skip this line
    
    # If we are in the Atoms section, process atom IDs
    if start_atoms:
        # Skip empty lines
        if not line.strip():
            continue
        
        # Extract atom ID (the first column in each line)
        tokens = line.split()
        if len(tokens) > 0:
            atom_id = tokens[0]  # The first token is the atom ID
            atom_ids.append(atom_id)

# Write the atom IDs to the output file
with open(output_file, 'w') as f:
    for atom_id in atom_ids:
        f.write(atom_id + '\n')

# Final message to confirm the task completion
print(f"Atom IDs have been written to '{output_file}'.")
