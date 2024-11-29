import numpy as np
import torch 

# modified import statements to access TFN-torch implementation 
import sys
import os
sys.path.append(os.path.abspath('../models/'))
import tensorfieldnetworks.layers as layers
import tensorfieldnetworks.utils as utils
from tensorfieldnetworks.utils import FLOAT_TYPE

tetris = [[(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
          [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)], # chiral_shape_2
          [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
          [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
          [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
          [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # T
          [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # zigzag
          [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]]  # L

dataset = [np.array(points_) for points_ in tetris]
num_classes = len(dataset)

# radial basis functions
rbf_low = 0.0
rbf_high = 3.5
rbf_count = 4
rbf_spacing = (rbf_high - rbf_low) / rbf_count
centers = torch.Tensor(np.linspace(rbf_low, rbf_high, rbf_count))

def get_inputs(r):
    
    # rij : [N, N, 3]
    rij = utils.difference_matrix(r)

    # dij : [N, N]
    dij = utils.distance_matrix(r)

    # rbf : [N, N, rbf_count]
    gamma = 1. / rbf_spacing
    rbf = torch.exp(-gamma * (dij.unsqueeze(-1) - centers)**2)
    
    return rij, dij, rbf

class Readout(torch.nn.Module):
    
    def __init__(self, input_dims, num_classes):
        super(Readout, self).__init__()
        
        self.lin = torch.nn.Linear(input_dims, num_classes,)
        self.input_dims = input_dims
        self.num_classes = num_classes
        
    def forward(self, inputs):
        inputs = torch.mean(inputs.squeeze(),dim=0)
#         print(inputs)
        inputs = self.lin.forward(inputs).unsqueeze(0)
        return inputs
        
        
class TetrisNetwork(torch.nn.Module):
    
    def __init__(self, rbf_dim = rbf_count, num_classes = num_classes):
        super(TetrisNetwork, self).__init__()
        self.layer_dims = [1, 4, 4, 4]
#         self.layer_dims = [1,4]
        self.num_layers = len(self.layer_dims) - 1  
        self.rbf_dim = rbf_dim
        self.embed = layers.SelfInteractionLayer(input_dim = 1, output_dim = 1, bias = False)
    
        self.layers = []
        for layer, (layer_dim_out, layer_dim_in) in enumerate(zip(self.layer_dims[1:], self.layer_dims[:-1])):
            self.layers.append(layers.Convolution(rbf_dim, layer_dim_in))
            self.layers.append(layers.Concatenation())
            self.layers.append(layers.SelfInteraction(layer_dim_in, layer_dim_out))
            self.layers.append(layers.NonLinearity(layer_dim_out))
        self.layers = torch.nn.ModuleList(self.layers)
        self.ones = torch.ones(1,4,1,1)
        self.readout = Readout(self.layer_dims[-1], num_classes)
        
    def forward(self, rbf, rij):
        embed = self.embed(self.ones.repeat([rbf.size()[0],1,1,1]))   
        input_tensor_list = {0: [embed]}
        for il, layer in enumerate(self.layers[::4]):
            input_tensor_list = self.layers[4*il](input_tensor_list, rbf, rij) #Convolution
#             if il == 1:
#                 print(input_tensor_list[0][0])
#                 print(input_tensor_list[0][1])
            input_tensor_list = self.layers[4*il+1](input_tensor_list) # Concatenation
#             if il == 0:
#                 print(input_tensor_list[1][0])
            input_tensor_list = self.layers[4*il+2](input_tensor_list) # Self interaction
#             if il == 2:
#                 print(input_tensor_list[0][0])
            input_tensor_list = self.layers[4*il+3](input_tensor_list) # Nonlinearity
#             if il == 1:
#                 print(input_tensor_list[0][0])
        return self.readout(input_tensor_list[0][0])
    
model = TetrisNetwork()

tetris_tensor = torch.Tensor(tetris)
rbf_list = []
rij_list = []
for t in tetris_tensor:
    rbf_list.append(get_inputs(t)[2])
    rij_list.append(get_inputs(t)[0])

labels = torch.LongTensor(np.arange(len(tetris_tensor))).view(-1,1)

i = 0
rij, rbf = rij_list[i], rbf_list[i]

outputs = model(rbf.unsqueeze(0), rij.unsqueeze(0))
print(outputs)

import torch.optim as optim

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(2001):  # loop over the dataset multiple times
# for epoch in range(100):  # loop over the dataset multiple times
    running_loss = 0.0
    order = np.arange(len(tetris_tensor))
#     np.random.shuffle(order)
    for i in order:
        label = labels[i]
        rij, rbf = rij_list[i].unsqueeze(0), rbf_list[i].unsqueeze(0)
        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = model(rbf, rij)
        loss = criterion(outputs, label)
   
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    if epoch%100 == 0:
        print('{:3.3f}'.format(running_loss/len(tetris_tensor)))
print('Finished Training')