
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 23:06:11 2023

@author: mugilgeethan
"""

import shlex
import os
import subprocess
import csv
import pandas as pd
from openpyxl import Workbook
import time
import json
import shutil

# Excute a shell command
def run_command(command):
    print(command)
    proc = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    output = proc.stdout.read()
    return output.decode("utf-8")

def prepare_inp_file(atom_id):
    input_master_file = open("input_master.lmp", "r")
    lines = input_master_file.readlines()
    input_master_file.close()

    inp_file = open(f"sim_{atom_id}/inp.lmp", "w+")

    for line in lines:
        inp_file.write(line)
        if "#Add_group_id" in line:
            new_line = f"group {group_id} id {atom_id}\n"
            inp_file.write(new_line)

    inp_file.close()


# Modify the get_next_suitable_atoms function to read from 'id.txt'
def get_next_suitable_atoms(atom_index):
    next_suitable_atom_list = []  # Initialize an empty list
    with open('id.txt', 'r') as file:
        for line in file:
            atom_id = int(line.strip())  # Read each line and convert it to an integer
            next_suitable_atom_list.append(atom_id)  # Append the atom ID to the list
    return next_suitable_atom_list

# Define a function to create a subfolder and copy the content from 'trigger'
# Define a function to create a subfolder and copy the content from 'trigger'
# Define a function to create a subfolder and copy the content from 'trigger'
def create_sim_folder(atom_id):
    folder_name = f"sim_{atom_id}"
    
    # Check if the folder already exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        # Copy the contents from the 'trigger' folder to the new subfolder
        for item in os.listdir("trigger"):
            source = os.path.join("trigger", item)
            destination = os.path.join(folder_name, item)
            if os.path.isdir(source):
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
    else:
        print(f"Folder '{folder_name}' already exists. Skipping.")

def run_simulations(next_suitable_atoms):
    for atom_id in next_suitable_atoms:
        # 1. Create a subfolder and copy the content from 'trigger'
        create_sim_folder(atom_id)

        # 2. Prepare the input file within the subfolder
        prepare_inp_file(atom_id)

        # 3. Run LAMMPS simulation within the subfolder
        os.chdir(f"sim_{atom_id}")
        lammps_output = run_command("qsub run.sh")
        os.chdir("..")

# Continue with your code
group_id = 1
list_atom_id =  []
dict_final_result = {}
N = 846
n = 1
m = 36
Efe = -4.0153
# Egb = -3432.12482416104
# Only one iteration
# 1. Find suitable next atom
next_suitable_atoms = get_next_suitable_atoms(len(list_atom_id))

# Run the simulations in parallel
run_simulations(next_suitable_atoms)

# Continue with the rest of your code to process the results and save them

# Process the results
for atom_id in next_suitable_atoms:
    # 4. Access the 'Tot_e_2.data' file in each subfolder
    os.chdir(f"sim_{atom_id}")
    while not os.path.isfile("Tot_e_2.data"):
        time.sleep(10)

    # 5. Get total energy from 'Tot_e_2.data' file
    with open("Tot_e_2.data", "r") as file:
        Tot_e_2_lines = file.readlines()
        if not Tot_e_2_lines:
            print("The file is empty")
        else:
            total_energy = float(Tot_e_2_lines[0].strip().split('-')[1])
            print(f"Total Energy for Atom {atom_id}: {total_energy}")

    # 6. Calculate formation energy
    formation_energy = -total_energy - ((N - n) * Efe)
   # trap_energy = 5.78099078-(-total_energy - Egb)
    print('formation_energy', formation_energy)
    # print('trap_energy', trap_energy)
    # 7. Store the result in the dictionary
    dict_final_result[atom_id] = formation_energy
    # dict_final_result[atom_id] = trap_energy

    os.remove("Tot_e_2.data")
    os.remove("inp.lmp")
    os.chdir("..")

# Final result
print(dict_final_result)

# To write the dict_msd data in an Excel file
# # -----------------------------------------------------
df = pd.DataFrame(data=dict_final_result, index=[0])
df = df.T
df.columns = ["formation_energy"]
print(df)
df.to_excel('formation_energy.xlsx')
# -----------------------------------------------------


# df = pd.DataFrame(data=dict_final_result, index=[0])
# df = df.T
# df.columns = ["trap_energy"]
# print(df)
# df.to_excel('trap_energy.xlsx')
# # -----------------------------------------------------



# Find the atom with the lowest formation energy
min_energy_atom_id = min(dict_final_result, key=dict_final_result.get)
min_energy = dict_final_result[min_energy_atom_id]

# max_energy_atom_id = max(dict_final_result, key=dict_final_result.get)
# max_energy = dict_final_result[max_energy_atom_id]



# # Save the ID and formation energy of the atom with the lowest formation energy in a text file
with open("lowest_energy_atom.txt", "w") as file:
    file.write(f"Atom ID with the lowest formation energy: {min_energy_atom_id}\n")
    file.write(f"Formation Energy: {min_energy}")

# Save the ID and trapping energy of the atom with the largest trapping energy in a text file
# with open("Stronger_trap_site.txt", "w") as file:
#     file.write(f"Atom ID with the largest trapping energy: {max_energy_atom_id}\n")
#     file.write(f"trapping Energy: {max_energy}")