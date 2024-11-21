
import shlex
import os
import subprocess
import csv
import shutil
from multiprocessing import Pool
from operator import itemgetter
import time
import functools
from contextlib import ExitStack

# Execute a shell command
def run_command(command):
    print(command)
    proc = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    output = proc.stdout.read()
    return output.decode("utf-8")

# Create a folder based on coordinate values
def create_folder(coord):
    folder_name = f"coord_{coord['x']}_{coord['y']}_{coord['z']}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

# Copy contents from 'trigger' folder to new folders
def copy_trigger_contents(new_folder):
    trigger_folder = 'trigger'
    for item in os.listdir(trigger_folder):
        source_item = os.path.join(trigger_folder, item)
        destination_item = os.path.join(new_folder, item)
        if os.path.isdir(source_item):
            shutil.copytree(source_item, destination_item)
        else:
            shutil.copy2(source_item, destination_item)

atom_type = 2



def add_to_input(atom_type, coord, folder_name, additional_coords=None):
    lammps_input_path = os.path.join(folder_name, "inp.lmp")
    
    with open("input_master.lmp", "r") as master_file:
        lines = master_file.readlines()

    with open(lammps_input_path, "w+") as new_file:
        for line in lines:
            new_file.write(line)
            if "#H in plane_Fe " in line:
                new_line = f"create_atoms {atom_type} single {coord['x']} {coord['y']} {coord['z']}\n"
                new_file.write(new_line)
    

# Initialize a dictionary to store results
energy_results = {}


# Function to run LAMMPS in a folder and return total energy
# def run_lammps_in_folder_and_get_energy(folder_name):


def run_lammps_in_folder_and_get_energy(folder_name, stack=None):
    if stack is None:
        stack = ExitStack()
    os.chdir(folder_name)
    lammps_output = run_command("qsub run.sh").split("\n")
    os.chdir("..")
    
    # Poll for the 'Tot_e_2.data' file to be created
    tot_e_2_path = os.path.join(folder_name, "Tot_e_2.data")
    max_sleep_duration = 10200
    elapsed_time = 0
    while elapsed_time < max_sleep_duration:
        if elapsed_time > 0:
            max_sleep_duration += 120  # Dynamically increase max_sleep_duration by 120
            print(f"File not found. Extending waiting time to {max_sleep_duration} seconds.")
        print(f"Waiting for file creation in {folder_name}...")
        
        if os.path.exists(tot_e_2_path):
            # File exists, read it and extract the total energy value
            with stack.enter_context(open(tot_e_2_path, "r")) as tot_e_2_file:
                tot_e_2_lines = tot_e_2_file.readlines()
                if tot_e_2_lines:
                    total_energy = float(tot_e_2_lines[0].strip().split(":")[1])
                    print(f"Total Energy in {folder_name}: {total_energy}")
                else:
                    print(f"Tot_e_2.data file is empty in {folder_name}")
                    total_energy = None
            return total_energy
        
        time.sleep(30)  # Short sleep interval
        elapsed_time += 30
    
    # If the file was not found within the given time, raise an exception
    raise FileNotFoundError(f"Tot_e_2.data file was not created in {folder_name} within the given time")


next_suitable_atom_coord = []

iteration_total_energy_data = -6869216.3123375 # update the energy corresponding to the data structure


# Calculate trap energy for a given total energy and iteration_total_energy_data
def calculate_trap_energy(total_energy, iteration_total_energy_data):
    e_fe = -6869222.15606093
    e_fe_h = -6869224.15539286
    e_trap_current_xyz = -(total_energy - iteration_total_energy_data + e_fe - e_fe_h)
    print('iteration_total_energy_data', iteration_total_energy_data)
    return e_trap_current_xyz


# Read coordinates from the CSV file
with open('filtered_data.csv') as file:
    reader = csv.reader(file, delimiter='\t')
    for row in reader:
        if len(row) >= 4:
            next_suitable_atom_coord.append({'x': row[1], 'y': row[2], 'z': row[3]})

# Define the number of iterations
num_iterations = 1  # You can adjust this value


# Initialize a list to store removed coordinates for each iteration
removed_coords_history = []

batch_size = 300
batch_time_gap = 2  # in seconds


# Main iteration loop
for iteration in range(num_iterations):
    print(f"Iteration {iteration + 1}/{num_iterations}")
    
    # Initialize the list to store removed coordinates
    removed_coords = []
    

    # Create folders first
    for coord in next_suitable_atom_coord:
        folder_name = f"coord_{coord['x']}_{coord['y']}_{coord['z']}"
        
        # Only create folders if they don't exist
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    
    # Create and update LAMMPS input file
    for batch_start in range(0, len(next_suitable_atom_coord), batch_size):
        batch_coords = next_suitable_atom_coord[batch_start:batch_start + batch_size]

        # Process each coordinate in the batch
        for coord in batch_coords:
            folder_name = create_folder(coord)
            copy_trigger_contents(folder_name)
            add_to_input(atom_type, coord, folder_name)

            # Wait for the specified time gap before processing the next coordinate
            time.sleep(batch_time_gap)    
        
    # Print the updated removed_coords list
    print(f"Updated removed_coords: {removed_coords}")
    
    # Create a list of folder names based on coordinates
    folder_names = [f"coord_{coord['x']}_{coord['y']}_{coord['z']}" for coord in next_suitable_atom_coord]

    # Print the folder names to see what's being processed
    print("Folder names being processed:", folder_names)
    
    # Run LAMMPS in each folder in parallel and store total energies
#    max_processes = min(len(next_suitable_atom_coord), os.cpu_count())
#    with Pool(processes=max_processes) as pool:
    with Pool(processes=len(next_suitable_atom_coord)) as pool:
        total_energies = pool.map(run_lammps_in_folder_and_get_energy, [f"coord_{coord['x']}_{coord['y']}_{coord['z']}" for coord in next_suitable_atom_coord])
#        time.sleep(1)   
    
    # Print the calculated total energies
    print("Total energies:", total_energies)
    
    # Initialize a dictionary to store trap energies for this iteration
    trap_energy_results_iteration = {}
    
    # Initialize a dictionary to store information
    result_dict = {}
    
    # Calculate trap energies for each coordinate
    for coord, total_energy in zip(next_suitable_atom_coord, total_energies):
        # Calculate trap energy using the provided function
        trap_energy = calculate_trap_energy(total_energy, iteration_total_energy_data)
        
        # Store trap energy in the dictionary
        trap_energy_results_iteration[(coord['x'], coord['y'], coord['z'])] = trap_energy
    
        # Store information in the dictionary
        result_dict[(coord['x'], coord['y'], coord['z'])] = {
            'total_energy': total_energy,
            'trap_energy': trap_energy
        }
 
    # Print the result dictionary
    print(result_dict)    
    
    # Print trap energy results for this iteration
    print(f"Iteration {iteration + 1} trap energy results: {trap_energy_results_iteration}")
    
    # Find the coordinate with the largest trap energy
    coord_with_largest_energy = max(trap_energy_results_iteration, key=trap_energy_results_iteration.get)
    print(f"Coordinate with largest trap energy: {coord_with_largest_energy}")
    
    
    print ('iteration_total_energy_data', iteration_total_energy_data)
    



    # Remove the coordinate with the largest energy from next_suitable_atom_coord
    removed_coords = [coord for coord in next_suitable_atom_coord if tuple(coord.values()) == coord_with_largest_energy]
  
    # Get the total energy and trap energy for the removed coordinates
    removed_coords_data = []
    for removed_coord in removed_coords:
        removed_total_energy = result_dict.get(
            (removed_coord['x'], removed_coord['y'], removed_coord['z']),
            {}).get('total_energy')
        removed_trap_energy = result_dict.get(
            (removed_coord['x'], removed_coord['y'], removed_coord['z']),
            {}).get('trap_energy')
        removed_coords_data.append({
            'x': removed_coord['x'],
            'y': removed_coord['y'],
            'z': removed_coord['z'],
            'total_energy': removed_total_energy,
            'trap_energy': removed_trap_energy
        })

    # Append the removed coordinates and their corresponding energies to the history
    removed_coords_history.append(removed_coords_data)
    
    next_suitable_atom_coord = [coord for coord in next_suitable_atom_coord if tuple(coord.values()) != coord_with_largest_energy]

    print("Removed coords:", removed_coords)
    
    for removed_coord in removed_coords:
        removed_total_energy = result_dict.get(
            (removed_coord['x'], removed_coord['y'], removed_coord['z']),
            {}).get('total_energy')
        if removed_total_energy is not None:
            iteration_total_energy_data = removed_total_energy
            break  # No need to continue searching once found  
    print("Updated iteration_total_energy_data:", iteration_total_energy_data)

     # Loop over coordinates and create/update files
    for coord in next_suitable_atom_coord:
         folder_name = f"coord_{coord['x']}_{coord['y']}_{coord['z']}"

         # Only update inp.lmp file
         add_to_input(atom_type, coord, folder_name)

         # Remove all files except the specified ones
         for item in os.listdir(folder_name):
             if item not in ['run.sh', 'V_H.data', 'FeH_Wen.eam.fs']:
                 item_path = os.path.join(folder_name, item)
                 if os.path.isfile(item_path):
                     os.remove(item_path)  

    trigger_folder = 'trigger'
    
    # Remove folders after extracting energy values for this iteration
    for coord in removed_coords:
        folder_name = f"coord_{coord['x']}_{coord['y']}_{coord['z']}"

        # Remove 'my_lattice_prep.data' in the 'trigger' folder
        if os.path.exists(os.path.join(trigger_folder, 'my_lattice_prep.data')):
            os.remove(os.path.join(trigger_folder, 'my_lattice_prep.data'))
            print("Removed 'my_lattice_prep.data' in 'trigger' folder")
        
        # Remove 'my_lattice_prep.data' in the folder
        my_lattice_prep_file = os.path.join(folder_name, 'my_lattice_prep.data')
        if os.path.exists(my_lattice_prep_file):
            os.remove(my_lattice_prep_file)
            print(f"Removed 'my_lattice_prep.data' in {folder_name}")

        source_file = os.path.join(folder_name, 'V_H.data')
        target_file = os.path.join(folder_name, 'my_lattice_prep_2.data')
    
        # Rename 'V_H.data' to 'my_lattice_prep_2.data'
        if os.path.exists(source_file):
            os.rename(source_file, target_file)
            print(f"Renamed 'V_H.data' to 'my_lattice_prep.data' in {folder_name}")
            
            # Copy 'my_lattice_prep.data' to the 'trigger' folder
        if os.path.exists(target_file):
            shutil.copy2(target_file, os.path.join(trigger_folder, 'my_lattice_prep.data'))
            print(f"Copied 'my_lattice_prep.data' to 'trigger' folder")    
     
        # Remove the folder corresponding to the removed coordinate
        shutil.rmtree(folder_name)
        print(f"Removed folder: {folder_name}")    
        
        
       
        
    # Print the updated removed_coords list
    print(f"Updated removed_coords: {removed_coords}")   
    
    # Update the input_master.lmp file with the previously removed coordinates
    with open("input_master.lmp", "r") as master_file:
        lines = master_file.readlines()

        
# Write removed_coords_history to a text file
with open(f'removed_coords_iteration_{iteration + 1}.txt', 'w') as txt_file:
    for iteration_data in removed_coords_history:
        for coord_data in iteration_data:
            txt_file.write(f"{coord_data['x']},{coord_data['y']},{coord_data['z']},"
                           f"Total Energy: {coord_data['total_energy']},"
                           f"Trap Energy: {coord_data['trap_energy']}\n")