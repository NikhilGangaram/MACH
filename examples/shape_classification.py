import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
sys.path.append(os.path.abspath('..'))
from models.tfn_torch import (TensorFieldNetwork, difference_matrix,
                            distance_matrix, random_rotation_matrix)

# Define the tetris shapes
tetris = [[(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
          [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)], # chiral_shape_2
          [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
          [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
          [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
          [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # T
          [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # zigzag
          [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]]  # L

dataset = [torch.tensor(points, dtype=torch.float32) for points in tetris]
num_classes = len(dataset)

# RBF parameters
rbf_low = 0.0
rbf_high = 3.5
rbf_count = 4
rbf_spacing = (rbf_high - rbf_low) / rbf_count
centers = torch.linspace(rbf_low, rbf_high, rbf_count)

class ShapeClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.gamma = 1.0 / rbf_spacing
        
        # Network architecture
        layer_dims = [1, 4, 4, 4]
        
        # Initial embedding
        self.embed = nn.Parameter(torch.ones(1, 1))
        
        # TFN layers
        self.tfn_layers = nn.ModuleList([
            TensorFieldNetwork(
                input_dims=[1, 1] if i == 0 else [layer_dims[i], layer_dims[i]],
                hidden_dims=[layer_dims[i+1], layer_dims[i+1]],
                output_dims=[layer_dims[i+1], layer_dims[i+1]]
            ) for i in range(len(layer_dims)-1)
        ])
        
        # Final classification layer
        self.classifier = nn.Linear(layer_dims[-1], num_classes)
        
    def compute_rbf(self, dij):
        # Expand centers to match batch dimension if needed
        centers_expanded = centers.to(dij.device)
        return torch.exp(-self.gamma * (dij.unsqueeze(-1) - centers_expanded).pow(2))
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Compute relative positions and distances
        rij = difference_matrix(x)
        dij = distance_matrix(x)
        
        # Compute RBF features
        rbf = self.compute_rbf(dij)
        
        # Initial embedding
        input_tensor_list = {0: [self.embed.expand(batch_size, -1, -1)]}
        
        # Apply TFN layers
        for layer in self.tfn_layers:
            input_tensor_list = layer(input_tensor_list, rbf, rij)
            
        # Get scalar features and average them
        tfn_scalars = input_tensor_list[0][0]
        tfn_output = torch.mean(tfn_scalars.squeeze(-1), dim=0)
        
        # Final classification
        return self.classifier(tfn_output)

def train_model():
    model = ShapeClassifier(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    max_epochs = 2001
    print_freq = 100
    
    model.train()
    for epoch in range(max_epochs):
        loss_sum = 0.0
        for label, shape in enumerate(dataset):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(shape)
            loss = criterion(outputs.unsqueeze(0), torch.tensor([label]))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            loss_sum += loss.item()
            
        if epoch % print_freq == 0:
            print(f"Epoch {epoch}: validation loss = {loss_sum/len(dataset):.3f}")
            
    return model

def test_model(model, test_set_size=25):
    model.eval()
    rng = np.random.RandomState()
    
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for i in range(test_set_size):
            for label, shape in enumerate(dataset):
                # Apply random rotation and translation
                rotation = random_rotation_matrix(rng)
                rotated_shape = torch.mm(shape, rotation)
                translation = torch.tensor(np.random.uniform(
                    low=-3., high=3., size=(3)), dtype=torch.float32).unsqueeze(0)
                translated_shape = rotated_shape + translation
                
                # Get prediction
                outputs = model(translated_shape)
                _, predicted = torch.max(outputs, 0)
                
                total_predictions += 1
                if predicted.item() == label:
                    correct_predictions += 1
                    
    accuracy = float(correct_predictions) / total_predictions
    print(f'Test accuracy: {accuracy:.6f}')
    return accuracy

if __name__ == "__main__":
    model = train_model()
    test_model(model)