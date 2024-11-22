import os
import random

def remove_ids_from_file(file_path, percentage=None, input_ids_file=None):
    """
    Remove a certain percentage of IDs from the file or based on an input list of IDs.
    - If percentage is given, it removes that percentage of IDs.
    - If input_ids_file is provided, it removes the specified IDs from the file.
    """
    # Read the ids from the file
    with open(file_path, 'r') as file:
        ids = file.readlines()

    # Clean up the ids list (remove any empty lines or extra spaces)
    ids = [id.strip() for id in ids if id.strip()]

    if percentage:
        # Calculate how many IDs to remove based on the percentage
        num_to_remove = int(len(ids) * (percentage / 100))
        removed_ids = random.sample(ids, num_to_remove)  # Randomly sample ids to remove
    elif input_ids_file:
        # If there's a file with specific IDs to remove, load those IDs
        with open(input_ids_file, 'r') as input_file:
            removed_ids = [id.strip() for id in input_file.readlines() if id.strip()]
    else:
        raise ValueError("Either percentage or input_ids_file must be provided")

    # Remove the selected ids from the original list
    remaining_ids = [id for id in ids if id not in removed_ids]

    # Write back the remaining ids to the original file (id.txt)
    with open(file_path, 'w') as file:
        file.write('\n'.join(remaining_ids) + '\n')

    return removed_ids

def update_lattice_file(atom_file, removed_ids):
    # Assuming the input file is in the 'trigger' folder
    atom_file_path = os.path.join('trigger', atom_file)
    
    # Read the contents of the original file
    with open(atom_file_path, 'r') as file:
        lines = file.readlines()

    # Find the line with the number of atoms (e.g., '528 atoms')
    atom_line_index = next(i for i, line in enumerate(lines) if 'atoms' in line)
    num_atoms = int(lines[atom_line_index].split()[0])

    # Update the atom count
    num_removed_atoms = len(removed_ids)
    new_atom_count = num_atoms - num_removed_atoms
    lines[atom_line_index] = f"{new_atom_count} atoms\n"

    # Find the start index of the atom section ("Atoms # atomic")
    atom_start_index = next(i for i, line in enumerate(lines) if line.startswith("Atoms # atomic"))
    
    # Everything after "Atoms # atomic"
    atom_lines = lines[atom_start_index + 1:]

    # Remove the lines for the atoms with IDs in removed_ids
    remaining_atom_lines = []
    for line in atom_lines:
        # Skip empty lines or lines that don't have a proper format
        if line.strip() == '':
            continue
        
        # Skip any lines before the atom data (e.g., "Masses", "Velocities", etc.)
        if line.startswith(('Masses', 'Velocities', 'Bonds', 'Pair Coeffs')):
            continue
        
        try:
            # The first column should be the atom ID
            atom_id = int(line.split()[0])  # The first column is the atom ID
            if atom_id not in removed_ids:
                remaining_atom_lines.append(line)
        except IndexError:
            # If a line doesn't have enough data to split into columns, skip it
            continue
        except ValueError:
            # If line doesn't start with a valid integer (e.g., "Masses"), skip it
            continue

    # Now write the modified data to the new file: trigger/modified_lattice_prep.data
    modified_file = 'trigger/modified_lattice_prep.data'

    # Ensure the 'trigger' directory exists
    if not os.path.exists('trigger'):
        os.makedirs('trigger')

    with open(modified_file, 'w') as file:
        # Write everything up to "Atoms # atomic"
        file.writelines(lines[:atom_start_index + 1])  
        # Write the remaining atom lines
        file.writelines(remaining_atom_lines)

    print(f"Updated '{modified_file}' by removing {len(removed_ids)} atoms and creating a new file.")

    # Save removed atom IDs to 'removed_ids.txt'
    with open('removed_ids.txt', 'w') as file:
        file.write('\n'.join(map(str, removed_ids)) + '\n')  # Convert IDs to strings

    print(f"Removed atom IDs saved to 'removed_ids.txt'.")

    # Rename the original file (inside the 'trigger' folder)
    original_file = os.path.join('trigger', 'original_' + atom_file)
    if os.path.exists(atom_file_path):  # Rename the original lattice file
        os.rename(atom_file_path, original_file)
        print(f"Original lattice file renamed to 'original_{atom_file}'.")

    # Rename the modified file to the original name
    os.rename(modified_file, atom_file_path)

    # Delete the original file after renaming
    os.remove(original_file)

def main():
    # Example usage: Define file paths and call the functions

    # Define the input file paths
    id_file = "id.txt"  # Path to the file containing IDs (id.txt)
    lattice_file = "my_lattice_prep.data"  # Path to the lattice data file in the trigger folder

    # Percentage of IDs to remove (e.g., 10% of IDs to remove)
    percentage_to_remove = 10  # Adjust this as needed

    # First, remove IDs from the id.txt file (you can also use a separate file of IDs to remove)
    removed_ids = remove_ids_from_file(id_file, percentage=percentage_to_remove)

    # Now, update the lattice file by removing atoms from the lattice data file
    update_lattice_file(lattice_file, [int(id) for id in removed_ids])


if __name__ == "__main__":
    main()
    
# PYTHONBUFFERED=1 nohup python3 vf.py > output.log 2?&1 & 