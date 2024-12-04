import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath('../../models/'))
import tensorfieldnetworks.layers as layers
import tensorfieldnetworks.utils as utils

# Setup for CUDA (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load formation energy and atom positions data from Excel file
file_path = 'formation_energy.xlsx'  # Adjust to your file path
df = pd.read_excel(file_path)
formation_energy_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

# Load lattice file containing atomic positions
lattice_file_path = 'trigger/my_lattice_prep.data'  # Adjust to your file path

# Read atom positions from the lattice file
with open(lattice_file_path, 'r') as f:
    lines = f.readlines()

# Find the start of atom data in the file
start_index = None
for i, line in enumerate(lines):
    if line.strip() == "Atoms # atomic":
        start_index = i + 1
        break

# Parse atom positions
atom_positions = {}
atom_ids = []
for line in lines[start_index:]:
    parts = line.split()
    if len(parts) >= 5:
        atom_id = int(parts[0])
        x_pos = float(parts[2])
        y_pos = float(parts[3])
        z_pos = float(parts[4])
        atom_ids.append(atom_id)
        atom_positions[atom_id] = (x_pos, y_pos, z_pos)

# Convert atom positions to a numpy array
positions_array = np.array([atom_positions[atom_id] for atom_id in atom_ids])

# Calculate pairwise distances between atoms using numpy
number_of_neighbors = 15
distances = np.linalg.norm(positions_array[:, np.newaxis] - positions_array, axis=2)  # Euclidean distance
np.fill_diagonal(distances, np.inf)  # Ignore diagonal (self-distances)
closest_indices = np.argsort(distances, axis=1)[:, :number_of_neighbors]  # Get indices of nearest neighbors
closest_positions_array = positions_array[closest_indices]

# Create a dictionary of closest atom positions for each atom
closest_atoms_dict = {atom_id: (positions_array[i], closest_positions_array[i]) 
                      for i, atom_id in enumerate(atom_ids)}

# Define Radial Basis Function (RBF) settings
rbf_low = 0.0
rbf_high = 3.5
rbf_count = 4
rbf_spacing = (rbf_high - rbf_low) / rbf_count
centers = torch.Tensor(np.linspace(rbf_low, rbf_high, rbf_count))

# Function to compute input features (rij, dij, rbf) from atom positions
def get_inputs(positions):
    rij = utils.difference_matrix(positions)
    dij = utils.distance_matrix(positions)
    gamma = 1. / rbf_spacing
    rbf = torch.exp(-gamma * (dij.unsqueeze(-1) - centers)**2)  # RBF kernel
    return rij, dij, rbf

# Prepare input data (rij, dij, rbf) and labels (formation energies)
inputs = []
labels = []
for atom_id, (atom_pos, closest_atoms_pos) in closest_atoms_dict.items():
    positions = np.vstack([atom_pos, closest_atoms_pos])
    rij, dij, rbf = get_inputs(torch.Tensor(positions))
    inputs.append((rij, rbf)) 
    labels.append(formation_energy_dict.get(atom_id, 0))  # Default to 0 if not found

# Convert inputs and labels to tensors
inputs_tensor = [(rij.unsqueeze(0), rbf.unsqueeze(0)) for rij, rbf in inputs]
labels_tensor = torch.Tensor(labels)

# Define the Readout layer for final output prediction
class Readout(torch.nn.Module):
    def __init__(self, input_dims, num_outputs):
        super(Readout, self).__init__()
        self.lin = torch.nn.Linear(input_dims, num_outputs)
        
    def forward(self, inputs):
        inputs = torch.mean(inputs.squeeze(), dim=0)  # Aggregate the node features
        inputs = self.lin(inputs).unsqueeze(0)  # Linear transformation for final output
        return inputs

# Define the EGNN (Equivariant Graph Neural Network) architecture
class EGNN(torch.nn.Module):
    def __init__(self, num_atoms, rbf_dim=rbf_count, num_outputs=1):
        super(EGNN, self).__init__()
        self.layer_dims = [1, num_atoms, num_atoms, 4]  # Hidden layer dimensions
        self.num_atoms = num_atoms
        self.num_layers = len(self.layer_dims) - 1
        self.rbf_dim = rbf_dim
        self.embed = layers.SelfInteractionLayer(input_dim=1, output_dim=1, bias=False)  # Embedding layer
        
        # Define the layers in the network (mirroring the network used in the shape classification example)
        self.layers = []
        for layer, (layer_dim_out, layer_dim_in) in enumerate(zip(self.layer_dims[1:], self.layer_dims[:-1])):
            self.layers.append(layers.Convolution(rbf_dim, layer_dim_in))
            self.layers.append(layers.Concatenation())
            self.layers.append(layers.SelfInteraction(layer_dim_in, layer_dim_out))
            self.layers.append(layers.NonLinearity(layer_dim_out))
        
        self.layers = torch.nn.ModuleList(self.layers)
        self.readout = Readout(self.layer_dims[-1], num_outputs)
        
    def forward(self, rbf, rij):
        # Initial embedding
        embed = self.embed(torch.ones(1, self.num_atoms, 1, 1).repeat([rbf.size(0), 1, 1, 1]))
        input_tensor_list = {0: [embed]}
        
        # Pass input through all layers of the network
        for il, layer in enumerate(self.layers[::4]):
            input_tensor_list = self.layers[4*il](input_tensor_list, rbf, rij)
            input_tensor_list = self.layers[4*il+1](input_tensor_list)
            input_tensor_list = self.layers[4*il+2](input_tensor_list)
            input_tensor_list = self.layers[4*il+3](input_tensor_list)
        
        return self.readout(input_tensor_list[0][0])

# Define the pre-trained weights path (set it to None if not loading pre-trained weights)
pre_trained_weights_path = None  

# Instantiate the model and move it to the device (GPU/CPU)
model = EGNN((number_of_neighbors+1), num_outputs=1).to(device)

# Load pre-trained weights if the path is provided
if pre_trained_weights_path:
    if os.path.exists(pre_trained_weights_path):
        print(f"Loading pre-trained weights from {pre_trained_weights_path}...")
        model.load_state_dict(torch.load(pre_trained_weights_path))
        print("Pre-trained weights loaded successfully.")
    else:
        print(f"Warning: The pre-trained weights file does not exist at {pre_trained_weights_path}. Continuing without loading weights.")
        
# Define loss function (MSE) and optimizer (Adam)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 2000
checkpoint_interval = epochs // 10  # Save model every 1/10th of the total epochs
for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, label) in enumerate(zip(inputs_tensor, labels_tensor)):
        rij, rbf = inputs
        label = label.unsqueeze(0).to(device)  # Move label to device
        rij = rij.to(device)  # Move rij to device
        rbf = rbf.to(device)  # Move rbf to device
        
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        outputs = model(rbf, rij)
        
        # Compute loss
        loss = criterion(outputs.squeeze(), label)
        loss.backward()  # Backward pass

        # Update weights
        optimizer.step()

        running_loss += loss.item()

    # Print the loss every epoch
    print(f'Epoch [{epoch}/{epochs}], Loss: {running_loss/len(inputs_tensor):.4f}')
    
    # Save the model every 1/10th of the epochs
    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint_path = f"checkpoints/epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved at epoch {epoch+1} to {checkpoint_path}")

# Training complete
print('Finished Training')
