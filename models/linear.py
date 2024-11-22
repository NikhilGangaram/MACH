import torch
import torch.nn as nn

# Class for a basic linear model  
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(12, 64)  # Input dimension is 12 (3D x 4 points)
        self.fc2 = nn.Linear(64, 32)  # Hidden layer with 32 units
        self.fc3 = nn.Linear(32, 8)   # Output layer with 8 classes (for 8 shapes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
