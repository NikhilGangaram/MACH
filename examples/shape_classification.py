import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
sys.path.append(os.path.abspath('..'))
from models.linear import LinearModel
from models.tfn_torch import *

# Define the tetris shapes (your original dataset)
tetris = [[(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
          [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)], # chiral_shape_2
          [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
          [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
          [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
          [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # T
          [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # zigzag
          [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]]  # L

# Convert the list of tuples into torch tensors
dataset = [torch.tensor(points, dtype=torch.float32) for points in tetris]

# Flatten the 4 points (each point is 3D) into a 12-dimensional vector
flattened_data = [shape.flatten() for shape in dataset]
flattened_data = torch.stack(flattened_data)  # Shape: (8, 12)

# Create labels (from 0 to 7)
labels = torch.tensor(np.arange(len(dataset)))

# Create a TensorDataset and DataLoader for easy batching
train_data = TensorDataset(flattened_data, labels)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

# Initialize the model), loss function, and optimizer
model = LinearModel()
# model = TFN_Tetris_Model(layer_dims=[1, 4, 4, 4], rbf_low=0.0, rbf_high=3.5, rbf_count=4, num_classes=8)
criterion = nn.CrossEntropyLoss()  # Multi-class classification loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the number of epochs to train for 
num_epochs = 1000

# Training loop 
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for inputs, targets in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track the loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == targets).sum().item()
        total_preds += targets.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct_preds / total_preds

    # Print only every 100th epoch
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")