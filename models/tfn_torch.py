import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg

EPSILON = 1e-8

def get_eijk():
    """
    Constant Levi-Civita tensor
    Returns:
        torch.Tensor of shape [3, 3, 3]
    """
    eijk_ = torch.zeros(3, 3, 3)
    eijk_[0, 1, 2] = eijk_[1, 2, 0] = eijk_[2, 0, 1] = 1.
    eijk_[0, 2, 1] = eijk_[2, 1, 0] = eijk_[1, 0, 2] = -1.
    return eijk_

def norm_with_epsilon(input_tensor, dim=None, keepdim=False):
    """
    Regularized norm
    Args:
        input_tensor: torch.Tensor
    Returns:
        torch.Tensor normed over dim
    """
    return torch.sqrt(torch.clamp(torch.sum(torch.square(input_tensor), dim=dim, keepdim=keepdim), min=EPSILON))

def ssp(x):
    """
    Shifted soft plus nonlinearity.
    """
    return torch.log(0.5 * torch.exp(x) + 0.5)

def unit_vectors(v, dim=-1):
    """Convert vectors to unit vectors."""
    return v / norm_with_epsilon(v, dim=dim, keepdim=True)

def Y_2(rij):
    """
    Spherical harmonics of degree l=2
    Args:
        rij: torch.Tensor [N, N, 3]
    Returns:
        torch.Tensor [N, N, 5]
    """
    x, y, z = rij[..., 0], rij[..., 1], rij[..., 2]
    r2 = torch.clamp(torch.sum(torch.square(rij), dim=-1), min=EPSILON)
    
    return torch.stack([
        x * y / r2,
        y * z / r2,
        (-torch.square(x) - torch.square(y) + 2. * torch.square(z)) / (2 * math.sqrt(3) * r2),
        z * x / r2,
        (torch.square(x) - torch.square(y)) / (2. * r2)
    ], dim=-1)

class RadialFunction(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
            
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, inputs):
        hidden = F.relu(self.linear1(inputs))
        return self.linear2(hidden)

class Filter0(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=1):
        super().__init__()
        self.radial = RadialFunction(input_dim, hidden_dim, output_dim)
        
    def forward(self, inputs):
        return torch.unsqueeze(self.radial(inputs), -1)

class Filter1(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=1):
        super().__init__()
        self.radial = RadialFunction(input_dim, hidden_dim, output_dim)
        
    def forward(self, inputs, rij):
        radial = self.radial(inputs)
        dij = torch.norm(rij, dim=-1)
        mask = torch.unsqueeze(dij < EPSILON, -1).expand_as(radial)
        masked_radial = torch.where(mask, torch.zeros_like(radial), radial)
        return torch.unsqueeze(unit_vectors(rij), -2) * torch.unsqueeze(masked_radial, -1)

class Filter2(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=1):
        super().__init__()
        self.radial = RadialFunction(input_dim, hidden_dim, output_dim)
        
    def forward(self, inputs, rij):
        radial = self.radial(inputs)
        dij = torch.norm(rij, dim=-1)
        mask = torch.unsqueeze(dij < EPSILON, -1).expand_as(radial)
        masked_radial = torch.where(mask, torch.zeros_like(radial), radial)
        return torch.unsqueeze(Y_2(rij), -2) * torch.unsqueeze(masked_radial, -1)

class SelfInteraction(nn.Module):
    def __init__(self, input_dim, output_dim, with_bias=True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=with_bias)
        
    def forward(self, inputs):
        # inputs: [N, C, 2L+1]
        # Transpose to [N, 2L+1, C], apply linear, then transpose back
        x = inputs.transpose(-2, -1)
        x = self.linear(x)
        return x.transpose(-2, -1)

def rotation_equivariant_nonlinearity(x, nonlin=ssp):
    """
    Rotation equivariant nonlinearity.
    Args:
        x: torch.Tensor with channels as -2 axis and M as -1 axis.
    """
    shape = x.shape
    representation_index = shape[-1]
    
    if representation_index == 1:
        return nonlin(x)
    else:
        norm = norm_with_epsilon(x, dim=-1)
        nonlin_out = nonlin(norm)
        factor = torch.div(nonlin_out, norm)
        return x * torch.unsqueeze(factor, -1)

class ConvolutionLayer(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=None):
        super().__init__()
        self.filter0 = Filter0(input_dims[0], hidden_dims, output_dims[0])
        self.filter1_to_0 = Filter1(input_dims[1], hidden_dims, output_dims[0])
        self.filter1_to_1 = Filter1(input_dims[1], hidden_dims, output_dims[1])
        
    def forward(self, input_tensor_list, rbf, rij):
        output_tensor_list = {0: [], 1: []}
        
        for key in input_tensor_list:
            for tensor in input_tensor_list[key]:
                # L x 0 -> L
                tensor_out = self.filter0(tensor)
                m = 0 if tensor_out.shape[-1] == 1 else 1
                output_tensor_list[m].append(tensor_out)
                
                if key == 1:
                    # L x 1 -> 0
                    tensor_out = self.filter1_to_0(tensor, rij)
                    m = 0 if tensor_out.shape[-1] == 1 else 1
                    output_tensor_list[m].append(tensor_out)
                
                if key == 0 or key == 1:
                    # L x 1 -> 1
                    tensor_out = self.filter1_to_1(tensor, rij)
                    m = 0 if tensor_out.shape[-1] == 1 else 1
                    output_tensor_list[m].append(tensor_out)
                    
        return output_tensor_list

class TensorFieldNetwork(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super().__init__()
        self.convolution = ConvolutionLayer(input_dims, hidden_dims)
        self.self_interaction0 = SelfInteraction(hidden_dims[0], output_dims[0], with_bias=True)
        self.self_interaction1 = SelfInteraction(hidden_dims[1], output_dims[1], with_bias=False)
        
    def forward(self, input_tensor_list, rbf, rij):
        # Convolution
        conv_out = self.convolution(input_tensor_list, rbf, rij)
        
        # Self-interaction
        output_tensor_list = {0: [], 1: []}
        for key in conv_out:
            for tensor in conv_out[key]:
                if key == 0:
                    tensor_out = self.self_interaction0(tensor)
                else:
                    tensor_out = self.self_interaction1(tensor)
                m = 0 if tensor_out.shape[-1] == 1 else 1
                output_tensor_list[m].append(tensor_out)
        
        # Nonlinearity
        for key in output_tensor_list:
            for i in range(len(output_tensor_list[key])):
                output_tensor_list[key][i] = rotation_equivariant_nonlinearity(output_tensor_list[key][i])
        
        return output_tensor_list

# Utility functions for geometry
def difference_matrix(geometry):
    """
    Get relative vector matrix for array of shape [N, 3].
    Args:
        geometry: torch.Tensor with Cartesian coordinates and shape [N, 3]
    Returns:
        Relative vector matrix with shape [N, N, 3]
    """
    ri = geometry.unsqueeze(1)  # [N, 1, 3]
    rj = geometry.unsqueeze(0)  # [1, N, 3]
    return ri - rj  # [N, N, 3]

def distance_matrix(geometry):
    """
    Get relative distance matrix for array of shape [N, 3].
    Args:
        geometry: torch.Tensor with Cartesian coordinates and shape [N, 3]
    Returns:
        Relative distance matrix with shape [N, N]
    """
    rij = difference_matrix(geometry)
    return norm_with_epsilon(rij, dim=-1)

def random_rotation_matrix(numpy_random_state):
    """
    Generates a random 3D rotation matrix from axis and angle.
    """
    rng = numpy_random_state
    axis = rng.randn(3)
    axis /= np.linalg.norm(axis) + EPSILON
    theta = 2 * np.pi * rng.uniform(0.0, 1.0)
    return torch.tensor(rotation_matrix(axis, theta))

def rotation_matrix(axis, theta):
    return scipy.linalg.expm(np.cross(np.eye(3), axis * theta))



#now, convert this shape_classification example which currently uses tensorflow into a pytorch example to test the code you just gave me 