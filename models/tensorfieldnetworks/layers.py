from math import sqrt
import math
import torch
import numpy as np
from tensorfieldnetworks import utils
from .utils import FLOAT_TYPE, EPSILON
from torch import nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional
from torch.nn.init import xavier_uniform_
from torch.nn.init import zeros_ as zeros_initializer
CONSTANT_BIAS = 0.0

class Dense(torch.nn.Linear):
    def __init__(self, input_dim, output_dim, bias=True, activation=None,
                 weight_init=xavier_uniform_, bias_init=zeros_initializer):
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.activation = activation

        super(Dense, self).__init__(input_dim, output_dim, bias)

    def reset_parameters(self):
        """
        Reinitialize model parameters.
        """
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output of the dense layer.
        """
        y = super(Dense, self).forward(inputs)
        if self.activation:
            y = self.activation(y)

        return y

class R(nn.Module):

    def __init__(self, input_dim, nonlin= functional.relu, hidden_dim=None, output_dim=1,
        weights_initializer=None, biases_initializer=None):

        """ input dimension is the rbf_dimension"""

        super(R, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.R_net = nn.Sequential( Dense(input_dim, hidden_dim, bias = True,
            activation = nonlin, bias_init = lambda x: init.constant_(x,CONSTANT_BIAS) ),
                        Dense(hidden_dim, output_dim, bias = True,
            activation = None,  bias_init = lambda x: init.constant_(x,CONSTANT_BIAS)))

    def forward(self, inputs):
        return self.R_net(inputs)

def unit_vectors(v, axis=-1):
    return v / utils.norm_with_epsilon(v, axis=axis, keep_dims=True)

def Y_2(rij):
    # rij : [N, N, 3]
    # x, y, z : [N, N]
    x = rij[..., 0]
    y = rij[..., 1]
    z = rij[..., 2]
    r2 = torch.max(torch.sum(rij**2, dim=-1), EPSILON)
    # return : [N, N, 5]
    output = torch.stack([x * y / r2,
                       y * z / r2,
                       (-x**2 - y**2 + 2. * z**2) / (2 * sqrt(3) * r2),
                       z * x / r2,
                       (x**2 - y**2) / (2. * r2)],
                      dim=-1)
    return output

class F(nn.Module):

    def __init__(self, l, input_dim, nonlin=functional.relu, hidden_dim=None, output_dim=1,
        weights_initializer=None, biases_initializer=None):

        super(F, self).__init__()
        self.radial = R(input_dim, nonlin=nonlin, hidden_dim=hidden_dim, output_dim=output_dim,
                       weights_initializer=weights_initializer, biases_initializer=biases_initializer)

        self.output_dim = output_dim
        """ input dimension is the rbf_dimension"""
        if l == 0 :
            self.rep = None
            self.forward = self.forward_no_rep
        elif l == 1:
            self.rep = unit_vectors
            self.forward = self.forward_rep
        elif l == 2:
            self.rep = Y_2
            self.forward = self.forward_rep

    def forward_no_rep(self, rbf_input, rij):
        radial = self.radial(rbf_input)
        return radial.unsqueeze(-1)

    def forward_rep(self, rbf_input, rij):
        radial = self.radial(rbf_input)
        dij = torch.norm(rij, dim=-1)
        condition = (dij < EPSILON).unsqueeze(-1).repeat(1, 1, 1, self.output_dim)
        masked_radial = torch.where(condition, torch.zeros_like(radial),
            radial)
        return self.rep(rij).unsqueeze(-2) * masked_radial.unsqueeze(-1)

class Filter(nn.Module):

    def __init__(self, l_filter, l_out, input_dim, nonlin=functional.relu, hidden_dim=None, output_dim=1,
        weights_initializer=None, biases_initializer=None):
        """ 4.1.3 Layer definition
            l1: l_input
            l2: l_output
            input_dim: rbf_dimension
        """

        super(Filter, self).__init__()
        self.F_out = F(l_filter, input_dim,
                      nonlin=nonlin,
                      hidden_dim=hidden_dim,
                      output_dim=output_dim,
                      weights_initializer=weights_initializer,
                      biases_initializer=biases_initializer)

        # Ensure eijk is on the same device as the model
        self.register_buffer('eijk', utils.get_eijk())
        self.l_filter = l_filter
        self.l_out = l_out

        if l_out == 0:
            if l_filter == 0:
                self.cg = None
                self.forward = self.forward_00
            elif l_filter == 1:
                # Use register_buffer to ensure device consistency
                self.register_buffer('cg', torch.eye(3).unsqueeze(0))
        elif l_out == 1:
            # Use register_buffer for device-consistent tensors
            self.register_buffer('cg_1', torch.eye(3).unsqueeze(-1))
            self.register_buffer('cg_3', self.eijk)
            self.cg = {1: self.cg_1, 3: self.cg_3}
            self.forward = self.forward_1
        elif l_out == 2:
            # Use register_buffer for device-consistent tensors
            self.register_buffer('cg', torch.eye(5).unsqueeze(-1))
        else:
            raise ValueError(f'l2 = {l_out} not implemented')

    def forward(self, layer_input, rbf_input, rij):
        cg = self.cg
        return torch.einsum('ijk,zabfj,zbfk->zafi', cg, self.F_out(rbf_input, rij),
            layer_input)

    def forward_00(self, layer_input, rbf_input, rij):
        cg = torch.eye(layer_input.size()[-1]).to(layer_input.device).unsqueeze(-2)
        return torch.einsum('ijk,zabfj,zbfk->zafi', cg, self.F_out(rbf_input, rij),
            layer_input)

    def forward_1(self, layer_input, rbf_input, rij):
        cg = self.cg[layer_input.size()[-1]]
        return torch.einsum('ijk,zabfj,zbfk->zafi', cg, self.F_out(rbf_input, rij),
            layer_input)

class Convolution(nn.Module):

    def __init__(self, input_dim, output_dim=1, weights_initializer=None,
        biases_initializer=None):
        """
        input_dim: rbf_dimension
        """
        super(Convolution, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.f00 = {}
        self.f10 = {}
        self.f11 = {}
        self.forward = self.forward_build

    def forward_build(self, input_tensor_list, rbf, rij):
        # Ensure the current device is used for new filters
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for key in input_tensor_list:
            for i, tensor in enumerate(input_tensor_list[key]):
                # L x 0 -> L
                name = f'{key}_{i}'
                self.f00[name] = Filter(0,0,input_dim=self.input_dim,
                    output_dim=self.output_dim).to(device)

                if key == 1:
                    # L x 1 -> 0
                    self.f10[name] = Filter(1,0,input_dim=self.input_dim,
                    output_dim=self.output_dim).to(device)

                if key in [0, 1]:
                    # L x 1 -> 1
                    self.f11[name] = Filter(1,1,input_dim=self.input_dim,
                    output_dim=self.output_dim).to(device)

        self.f00 = torch.nn.ModuleDict(self.f00)
        self.f10 = torch.nn.ModuleDict(self.f10)
        self.f11 = torch.nn.ModuleDict(self.f11)
        self.forward = self.forward_later

        return self.forward(input_tensor_list, rbf, rij)

    def forward_later(self, input_tensor_list, rbf, rij):
        output_tensor_list = {0:[], 1:[]}
        for key in input_tensor_list:
            for i, tensor in enumerate(input_tensor_list[key]):
                name = f'{key}_{i}'
                # L x 0 -> L
                tensor_out = self.f00[name](tensor,
                                      rbf,
                                      rij)
                output_tensor_list[key].append(tensor_out)
                if key == 1:
                    # L x 1 -> 0
                    tensor_out = self.f10[name](tensor,
                                          rbf,
                                          rij)
                    output_tensor_list[0].append(tensor_out)
                if key in [0, 1]:
                    # L x 1 -> 1
                    tensor_out = self.f11[name](tensor,
                                          rbf,
                                          rij)
                    output_tensor_list[1].append(tensor_out)
        return output_tensor_list

class SelfInteractionLayer(nn.Module):

    def __init__(self, input_dim, output_dim, bias = False,
            weights_initializer=None, biases_initializer=None):
        super(SelfInteractionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = Parameter(torch.Tensor(output_dim, input_dim))
        if bias:
            self.bias = Parameter(torch.Tensor(output_dim))
        else:
            # Ensure bias is a tensor on the correct device
            self.register_buffer('bias', torch.zeros(output_dim))
        self.reset_parameters()
        self.forward = self.first_forward

    def reset_parameters(self):
        init.orthogonal_(self.weight)
        if self.bias is not None:
            init.constant_(self.bias, CONSTANT_BIAS)

    def first_forward(self, layer_input):
        if not layer_input.size()[-2] == self.input_dim:
            self.input_dim = layer_input.size()[-2]
            self.weight = Parameter(torch.Tensor(self.output_dim, self.input_dim).to(layer_input.device))
            self.reset_parameters()
        self.forward = self.later_forward
        return self.forward(layer_input)

    def later_forward(self, layer_input):
        # Ensure weight and bias are on the same device as the input
        weight = self.weight.to(layer_input.device)
        bias = self.bias.to(layer_input.device)

        return (torch.einsum('zafi,gf->zaig',
            layer_input, weight) + bias).permute(0,1, 3, 2)

class SelfInteraction(nn.Module):

    def __init__(self, input_dim, output_dim, weights_initializer=None,
        biases_initializer=None):
        super(SelfInteraction, self).__init__()

        self.SI = {}
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights_init = weights_initializer
        self.biases_init = biases_initializer

        self.forward = self.forward_init

    def forward_init(self, input_tensor_list):
        # Determine the device from the first available tensor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for key in input_tensor_list:
            for i, tensor in enumerate(input_tensor_list[key]):
                name = f'{key}_{i}'
                if key == 0:
                    self.SI[name] = SelfInteractionLayer(self.input_dim, self.output_dim, True,
                        self.weights_init, self.biases_init).to(device)
                else:
                    self.SI[name] = SelfInteractionLayer(self.input_dim, self.output_dim, False,
                        self.weights_init, self.biases_init).to(device)

        self.SI = torch.nn.ModuleDict(self.SI)
        self.forward = self.forward_later
        return self.forward(input_tensor_list)

    def forward_later(self, input_tensor_list):
        output_tensor_list = {0: [], 1: []}
        for key in input_tensor_list:
            for i, tensor in enumerate(input_tensor_list[key]):
                tensor_out = self.SI[f'{key}_{i}'](tensor)
                output_tensor_list[key].append(tensor_out)
        return output_tensor_list


class NonLinearity(nn.Module):

    def __init__(self, channels, nonlin=functional.elu, biases_initializer=None):
        super(NonLinearity, self).__init__()
        self.biases_initializer = biases_initializer
        self.nonlin = nonlin
        self.channels = channels
        self.biases = {}
        self.forward = self.forward_init

    def forward_init(self, input_tensor_list):
        # Determine the device from the first available tensor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for key in input_tensor_list:
            for i, tensor in enumerate(input_tensor_list[key]):
                name = f'{key}_{i}'
                self.biases[name] = Parameter(torch.Tensor(self.channels).to(device))

        self.biases = torch.nn.ParameterDict(self.biases)
        self.forward = self.forward_later
        self.reset_parameters()
        return self.forward(input_tensor_list)

    def forward_later(self, input_tensor_list):
        output_tensor_list = {0: [], 1: []}
        for key in input_tensor_list:
            for i, tensor in enumerate(input_tensor_list[key]):
                # Ensure biases are on the same device as the tensor
                biases = self.biases[f'{key}_{i}'].to(tensor.device)
                tensor_out = utils.rotation_equivariant_nonlinearity(tensor,
                                                                     nonlin=self.nonlin,
                                                                     biases=biases)
                output_tensor_list[key].append(tensor_out)
        return output_tensor_list

    def reset_parameters(self):
        for key in self.biases:
            init.constant_(self.biases[key], CONSTANT_BIAS)

class Concatenation(nn.Module):

    def __init__(self):
        super(Concatenation, self).__init__()

    def forward(self, input_tensor_list):
        output_tensor_list = {0: [], 1: []}
        for key in input_tensor_list:
            # Concatenate along channel axis, handling empty lists
            if input_tensor_list[key]:
                output_tensor_list[key].append(torch.cat(input_tensor_list[key], dim=-2))
        return output_tensor_list