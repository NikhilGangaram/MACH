import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Set the default floating-point type for PyTorch tensors
FLOAT_TYPE = torch.float32
EPSILON = 1e-8  # A small value to avoid division by zero

def get_eijk():
    """
    Generate the constant Levi-Civita tensor, which is used for handling cross products and other operations
    in vector calculus, specifically for 3D space.

    The Levi-Civita symbol is a three-dimensional, antisymmetric tensor that encodes the sign of a 
    permutation of the indices (i, j, k). It's commonly used in physics, particularly in electromagnetism 
    and fluid dynamics.

    Returns:
        torch.Tensor of shape [3, 3, 3]: The Levi-Civita tensor
    """
    # Initialize a 3x3x3 tensor filled with zeros
    eijk_ = np.zeros((3, 3, 3))
    
    # Set the values for the Levi-Civita tensor
    eijk_[0, 1, 2] = eijk_[1, 2, 0] = eijk_[2, 0, 1] = 1.  # Positive values for cyclic permutations
    eijk_[0, 2, 1] = eijk_[2, 1, 0] = eijk_[1, 0, 2] = -1.  # Negative values for reverse cyclic permutations
    
    # Return as a PyTorch tensor
    return torch.tensor(eijk_, dtype=FLOAT_TYPE)

def norm_with_epsilon(input_tensor, dim=None, keepdim=False):
    """
    Compute the Euclidean norm of a tensor with a small epsilon added to prevent division by zero.

    The regularized norm is useful for numerical stability, especially when working with vectors 
    or tensors that may have near-zero magnitudes.

    Args:
        input_tensor: torch.Tensor
            The input tensor for which the norm is computed.
        dim: Dimension(s) along which to compute the sum (optional).
        keepdim: Whether to retain the reduced dimension(s) (optional).

    Returns:
        torch.Tensor: The regularized Euclidean norm along the specified dimensions.
    """
    # Ensure the epsilon value is of the same type and device as the input tensor
    epsilon_tensor = torch.tensor(EPSILON, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Compute the squared sum along the specified dimension(s)
    squared_sum = torch.sum(input_tensor**2, dim=dim, keepdim=keepdim)
    
    # Regularize by ensuring that the norm is never zero by adding epsilon
    return torch.sqrt(torch.maximum(squared_sum, epsilon_tensor))

def ssp(x):
    """
    Shifted Soft Plus nonlinearity.

    This function applies the Soft Plus activation function, which is a smooth approximation to the ReLU 
    function, and then shifts the output to avoid producing large values for small inputs.

    The Soft Plus function is defined as: `ssp(x) = log(1 + exp(x))`. The shifted version adds a constant 
    term to the exponentiation to ensure the function is smooth across a wider range of inputs.

    Args:
        x: torch.Tensor
            The input tensor.

    Returns:
        torch.Tensor: The shifted Soft Plus function applied element-wise.
    """
    return torch.log(0.5 * torch.exp(x) + 0.5)  # Shifted Soft Plus

def rotation_equivariant_nonlinearity(x, nonlin=ssp, biases_initializer=None):
    """
    A nonlinearity that is equivariant to 3D rotations. This means that if the input is rotated, the output 
    will be rotated in the same way.

    Rotation-equivariant networks ensure that the output respects the symmetry of rotations, which is useful 
    in tasks such as computer vision or physics simulations where the input data has rotational symmetry.

    Args:
        x: torch.Tensor
            The input tensor with shape [N, C, M] where N is the batch size, C is the number of channels, 
            and M is the number of samples.
        nonlin: Callable (default: ssp)
            The nonlinear activation function to be applied (default is Shifted Soft Plus).
        biases_initializer: Callable (default: None)
            The initializer for the biases. If None, biases are initialized to zero.

    Returns:
        torch.Tensor: The output tensor with rotation-equivariant nonlinearity applied.
    """
    # Initialize biases if not provided
    if biases_initializer is None:
        biases_initializer = torch.zeros_like

    # Get the shape of the input tensor
    shape = x.shape
    channels = shape[-2]
    representation_index = shape[-1]

    # Create bias parameters for the channels
    biases = torch.nn.Parameter(torch.zeros(channels, dtype=FLOAT_TYPE))

    if representation_index == 1:
        # Apply the nonlinearity directly if there's only one representation index
        return nonlin(x)
    else:
        # Compute the norm with epsilon to avoid division by zero
        norm = norm_with_epsilon(x, dim=-1)
        
        # Apply the nonlinearity to the normalized input + biases
        nonlin_out = nonlin(norm + biases)
        
        # Compute a scaling factor
        factor = nonlin_out / norm
        
        # Expand dims for representation index and apply scaling
        return x * factor.unsqueeze(-1)

def difference_matrix(geometry):
    """
    Compute the relative vector matrix for an array of 3D Cartesian coordinates.

    This function calculates the pairwise differences between all points in a set of 3D coordinates. This is 
    useful in simulations that require pairwise interactions between particles, such as molecular dynamics.

    Args:
        geometry: torch.Tensor
            A tensor of shape [N, 3], where N is the number of points and each point is represented by 
            its 3D Cartesian coordinates.

    Returns:
        torch.Tensor: A tensor of shape [N, N, 3] containing the relative vectors between all pairs of points.
    """
    # Create two tensors representing each point in the set, one along the rows and the other along the columns
    ri = geometry.unsqueeze(1)  # Shape: [N, 1, 3]
    rj = geometry.unsqueeze(0)  # Shape: [1, N, 3]
    
    # Subtract each pair of points to compute the relative vectors
    rij = ri - rj  # Shape: [N, N, 3]
    
    return rij

def distance_matrix(geometry):
    """
    Compute the pairwise distances between points in 3D space.

    This function calculates the Euclidean distance between each pair of points in the 3D space defined by 
    the Cartesian coordinates. This is often used in clustering algorithms, distance-based loss functions, 
    or geometric computations.

    Args:
        geometry: torch.Tensor
            A tensor of shape [N, 3], where N is the number of points and each point is represented by 
            its 3D Cartesian coordinates.

    Returns:
        torch.Tensor: A tensor of shape [N, N] containing the pairwise distances between points.
    """
    # Compute the pairwise relative vectors
    rij = difference_matrix(geometry)
    
    # Compute the Euclidean distance (norm) of each relative vector
    dij = norm_with_epsilon(rij, dim=-1)  # Shape: [N, N]
    
    return dij

def random_rotation_matrix(numpy_random_state):
    """
    Generate a random 3D rotation matrix by selecting a random axis and angle.

    This function generates a random axis (a unit vector) and a random angle (between 0 and 2pi), and then 
    uses these to create a rotation matrix.

    Args:
        numpy_random_state: numpy random state object
            A random state object to generate random numbers.

    Returns:
        numpy.ndarray: A 3x3 rotation matrix that represents a random rotation.
    """
    # Generate a random axis by selecting a random 3D vector
    axis = numpy_random_state.randn(3)
    
    # Normalize the axis to ensure it's a unit vector
    axis /= np.linalg.norm(axis) + EPSILON
    
    # Generate a random rotation angle between 0 and 2*pi
    theta = 2 * np.pi * numpy_random_state.uniform(0.0, 1.0)
    
    # Compute the corresponding rotation matrix using the axis and angle
    return rotation_matrix(axis, theta)

def rotation_matrix(axis, theta):
    """
    Compute the 3x3 rotation matrix corresponding to a rotation around a given axis by a given angle.

    This function uses Rodrigues' rotation formula, which provides a way to compute the rotation matrix 
    from an axis of rotation and an angle of rotation.

    Args:
        axis: numpy.ndarray of shape (3,)
            The axis of rotation, which must be a unit vector.
        theta: float
            The angle of rotation in radians.

    Returns:
        numpy.ndarray: A 3x3 rotation matrix.
    """
    # Normalize the axis to ensure it's a unit vector
    axis = axis / np.linalg.norm(axis)
    
    # Skew-symmetric matrix of the axis vector (used for cross product operations)
    axis_skew = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
    
    # Identity matrix
    I = np.eye(3)
    
    # Apply Rodrigues' rotation formula
    R = I + np.sin(theta) * axis_skew + (1 - np.cos(theta)) * np.dot(axis_skew, axis_skew)
    
    return R

def unit_vectors(v, axis=-1):
    """
    Normalize the input tensor `v` along the specified axis to unit vectors.

    Divides each vector by its norm along the given axis, ensuring numerical stability 
    with `norm_with_epsilon` to handle near-zero magnitudes.

    Args:
        v: torch.Tensor
            Input tensor of vectors, shape [N, ..., M].
        axis: int (default: -1)
            Axis along which to compute the norm and normalize.

    Returns:
        torch.Tensor:
            Tensor of unit vectors with the same shape as `v`.
    """
    return v / norm_with_epsilon(v, dim=axis, keepdim=True)


## TFN LAYERS ## 

def R(inputs, nonlin=F.relu, hidden_dim=None, output_dim=1, weights_initializer=None, biases_initializer=None):
    """
    Radial function with two-layer MLP, outputting a radial map.

    Args:
        inputs: torch.Tensor
            The input tensor of shape [N, ..., input_dim].
        nonlin: callable (default: F.relu)
            The nonlinearity applied to the hidden layer.
        hidden_dim: int (default: None)
            The number of hidden units. Defaults to input_dim.
        output_dim: int (default: 1)
            The number of output units.
        weights_initializer: callable (default: None)
            Initializer for the weights.
        biases_initializer: callable (default: None)
            Initializer for the biases.

    Returns:
        torch.Tensor: Radial function output of shape [N, ..., output_dim].
    """
    input_dim = inputs.shape[-1]
    if hidden_dim is None:
        hidden_dim = input_dim

    if weights_initializer is None:
        weights_initializer = torch.nn.init.xavier_uniform_
    if biases_initializer is None:
        biases_initializer = torch.zeros_

    # Layer 1 weights and biases
    w1 = torch.nn.Parameter(weights_initializer(torch.empty(input_dim, hidden_dim)))
    b1 = torch.nn.Parameter(biases_initializer(torch.empty(hidden_dim)))
    
    # Layer 2 weights and biases
    w2 = torch.nn.Parameter(weights_initializer(torch.empty(hidden_dim, output_dim)))
    b2 = torch.nn.Parameter(biases_initializer(torch.empty(output_dim)))

    # Hidden layer computation
    hidden_layer = nonlin(torch.matmul(inputs, w1) + b1)
    
    # Radial output
    return torch.matmul(hidden_layer, w2) + b2

def Y_2(rij):
    """
    Compute the Y_2 function for a set of pairwise distance vectors.

    Args:
        rij: torch.Tensor
            A tensor of pairwise differences between points, shape [N, N, 3].

    Returns:
        torch.Tensor:
            The computed Y_2 output, shape [N, N, 5].
    """
    # Extract x, y, z components from rij
    x = rij[:, :, 0]
    y = rij[:, :, 1]
    z = rij[:, :, 2]

    # Compute r^2 with epsilon to avoid division by zero
    r2 = torch.sum(rij**2, dim=-1)
    r2 = torch.maximum(r2, torch.tensor(EPSILON, dtype=r2.dtype, device=r2.device))

    # Compute the Y_2 output
    return torch.stack([
        (x * y) / r2,
        (y * z) / r2,
        (-x**2 - y**2 + 2 * z**2) / (2 * torch.sqrt(torch.tensor(3.0, dtype=r2.dtype, device=r2.device)) * r2),
        (z * x) / r2,
        (x**2 - y**2) / (2 * r2)
    ], dim=-1)

def F_0(inputs, nonlin=F.relu, hidden_dim=None, output_dim=1,
        weights_initializer=None, biases_initializer=None):
    """
    Compute F_0 by applying the R function and expanding dimensions at the end.
    
    Args:
        inputs: torch.Tensor
            The input tensor.
        nonlin: callable (default: torch.nn.functional.relu)
            Nonlinearity to apply to hidden layers.
        hidden_dim: int (default: None)
            Number of hidden units in the MLP.
        output_dim: int (default: 1)
            Number of output dimensions.
        weights_initializer: callable (default: None)
            Initializer for the weights.
        biases_initializer: callable (default: None)
            Initializer for the biases.

    Returns:
        torch.Tensor: The output tensor with shape [N, N, output_dim, 1].
    """
    radial_output = R(inputs, nonlin=nonlin, hidden_dim=hidden_dim, output_dim=output_dim,
                      weights_initializer=weights_initializer, biases_initializer=biases_initializer)
    return radial_output.unsqueeze(-1)  # Add singleton dimension at the end

def F_1(inputs, rij, nonlin=F.relu, hidden_dim=None, output_dim=1,
        weights_initializer=None, biases_initializer=None):
    """
    Compute F_1 by applying the R function, masking radial output for dij=0,
    and multiplying by the unit vectors of rij.

    Args:
        inputs: torch.Tensor
            The input tensor.
        rij: torch.Tensor
            The pairwise difference tensor [N, N, 3].
        nonlin: callable (default: torch.nn.functional.relu)
            Nonlinearity to apply to hidden layers.
        hidden_dim: int (default: None)
            Number of hidden units in the MLP.
        output_dim: int (default: 1)
            Number of output dimensions.
        weights_initializer: callable (default: None)
            Initializer for the weights.
        biases_initializer: callable (default: None)
            Initializer for the biases.

    Returns:
        torch.Tensor: The output tensor with shape [N, N, output_dim, 3].
    """
    # Get radial output using R function
    radial = R(inputs, nonlin=nonlin, hidden_dim=hidden_dim, output_dim=output_dim,
               weights_initializer=weights_initializer, biases_initializer=biases_initializer)

    # Compute distance dij and create mask for dij = 0
    dij = torch.norm(rij, dim=-1)  # Shape [N, N]
    condition = dij < EPSILON
    masked_radial = torch.where(condition.unsqueeze(-1), torch.zeros_like(radial), radial)

    # Multiply by unit vectors of rij
    unit_rij = unit_vectors(rij, axis=-1)  # Unit vectors [N, N, 3]
    return masked_radial.unsqueeze(-1) * unit_rij.unsqueeze(-2)  # Shape [N, N, output_dim, 3]

def F_2(inputs, rij, nonlin=F.relu, hidden_dim=None, output_dim=1,
        weights_initializer=None, biases_initializer=None):
    """
    Compute F_2 by applying the R function, masking radial output for dij=0,
    and multiplying by Y_2 of rij.

    Args:
        inputs: torch.Tensor
            The input tensor.
        rij: torch.Tensor
            The pairwise difference tensor [N, N, 3].
        nonlin: callable (default: torch.nn.functional.relu)
            Nonlinearity to apply to hidden layers.
        hidden_dim: int (default: None)
            Number of hidden units in the MLP.
        output_dim: int (default: 1)
            Number of output dimensions.
        weights_initializer: callable (default: None)
            Initializer for the weights.
        biases_initializer: callable (default: None)
            Initializer for the biases.

    Returns:
        torch.Tensor: The output tensor with shape [N, N, output_dim, 5].
    """
    # Get radial output using R function
    radial = R(inputs, nonlin=nonlin, hidden_dim=hidden_dim, output_dim=output_dim,
               weights_initializer=weights_initializer, biases_initializer=biases_initializer)

    # Compute distance dij and create mask for dij = 0
    dij = torch.norm(rij, dim=-1)  # Shape [N, N]
    condition = dij < EPSILON
    masked_radial = torch.where(condition.unsqueeze(-1), torch.zeros_like(radial), radial)

    # Multiply by Y_2 of rij
    Y2_rij = Y_2(rij)  # Shape [N, N, 5]
    return masked_radial.unsqueeze(-1) * Y2_rij.unsqueeze(-2)  # Shape [N, N, output_dim, 5]

def filter_0(layer_input, rbf_inputs, nonlin=torch.relu, hidden_dim=None, output_dim=1,
             weights_initializer=None, biases_initializer=None):
    """
    Compute filter_0 by applying F_0 and performing an einsum operation with the layer input.
    
    Args:
        layer_input: torch.Tensor
            The input tensor with shape [N, input_dim].
        rbf_inputs: torch.Tensor
            The input tensor for radial basis functions.
        nonlin: callable (default: torch.relu)
            Nonlinearity to apply to hidden layers.
        hidden_dim: int (default: None)
            Number of hidden units in the MLP.
        output_dim: int (default: 1)
            Number of output dimensions.
        weights_initializer: callable (default: None)
            Initializer for the weights.
        biases_initializer: callable (default: None)
            Initializer for the biases.

    Returns:
        torch.Tensor: The output tensor after filtering.
    """
    # Compute F_0
    F_0_out = F_0(rbf_inputs, nonlin=nonlin, hidden_dim=hidden_dim, output_dim=output_dim,
                  weights_initializer=weights_initializer, biases_initializer=biases_initializer)
    
    # Expand filter axis "j"
    input_dim = layer_input.shape[-1]
    cg = torch.eye(input_dim, dtype=layer_input.dtype).unsqueeze(-2)
    
    # Perform einsum operation: L x 0 -> L
    return torch.einsum('ijk,abfj,bfk->afi', cg, F_0_out, layer_input)

def filter_1_output_0(layer_input, rbf_inputs, rij, nonlin=torch.relu, hidden_dim=None, output_dim=1,
                      weights_initializer=None, biases_initializer=None):
    """
    Compute filter_1_output_0 by applying F_1 and performing an einsum operation with the layer input.
    
    Args:
        layer_input: torch.Tensor
            The input tensor with shape [N, input_dim].
        rbf_inputs: torch.Tensor
            The input tensor for radial basis functions.
        rij: torch.Tensor
            The pairwise distance tensor [N, N, 3].
        nonlin: callable (default: torch.relu)
            Nonlinearity to apply to hidden layers.
        hidden_dim: int (default: None)
            Number of hidden units in the MLP.
        output_dim: int (default: 1)
            Number of output dimensions.
        weights_initializer: callable (default: None)
            Initializer for the weights.
        biases_initializer: callable (default: None)
            Initializer for the biases.

    Returns:
        torch.Tensor: The output tensor after filtering.
    """
    # Compute F_1
    F_1_out = F_1(rbf_inputs, rij, nonlin=nonlin, hidden_dim=hidden_dim, output_dim=output_dim,
                  weights_initializer=weights_initializer, biases_initializer=biases_initializer)
    
    input_dim = layer_input.shape[-1]
    
    if input_dim == 1:
        raise ValueError("0 x 1 cannot yield 0")
    elif input_dim == 3:
        # 1 x 1 -> 0
        cg = torch.eye(3, dtype=layer_input.dtype).unsqueeze(0)
        return torch.einsum('ijk,abfj,bfk->afi', cg, F_1_out, layer_input)
    else:
        raise NotImplementedError("Other Ls not implemented")

def filter_1_output_1(layer_input, rbf_inputs, rij, nonlin=torch.relu, hidden_dim=None, output_dim=1,
                      weights_initializer=None, biases_initializer=None):
    """
    Compute filter_1_output_1 by applying F_1 and performing an einsum operation with the layer input.
    
    Args:
        layer_input: torch.Tensor
            The input tensor with shape [N, input_dim].
        rbf_inputs: torch.Tensor
            The input tensor for radial basis functions.
        rij: torch.Tensor
            The pairwise distance tensor [N, N, 3].
        nonlin: callable (default: torch.relu)
            Nonlinearity to apply to hidden layers.
        hidden_dim: int (default: None)
            Number of hidden units in the MLP.
        output_dim: int (default: 1)
            Number of output dimensions.
        weights_initializer: callable (default: None)
            Initializer for the weights.
        biases_initializer: callable (default: None)
            Initializer for the biases.

    Returns:
        torch.Tensor: The output tensor after filtering.
    """
    # Compute F_1
    F_1_out = F_1(rbf_inputs, rij, nonlin=nonlin, hidden_dim=hidden_dim, output_dim=output_dim,
                  weights_initializer=weights_initializer, biases_initializer=biases_initializer)
    
    input_dim = layer_input.shape[-1]
    
    if input_dim == 1:
        # 0 x 1 -> 1
        cg = torch.eye(3, dtype=layer_input.dtype).unsqueeze(-1)
        return torch.einsum('ijk,abfj,bfk->afi', cg, F_1_out, layer_input)
    elif input_dim == 3:
        # 1 x 1 -> 1
        eijk = get_eijk()  # Assuming get_eijk() is implemented
        return torch.einsum('ijk,abfj,bfk->afi', eijk, F_1_out, layer_input)
    else:
        raise NotImplementedError("Other Ls not implemented")

def filter_2_output_2(layer_input, rbf_inputs, rij, nonlin=torch.relu, hidden_dim=None, output_dim=1,
                      weights_initializer=None, biases_initializer=None):
    """
    Compute filter_2_output_2 by applying F_2 and performing an einsum operation with the layer input.
    
    Args:
        layer_input: torch.Tensor
            The input tensor with shape [N, input_dim].
        rbf_inputs: torch.Tensor
            The input tensor for radial basis functions.
        rij: torch.Tensor
            The pairwise distance tensor [N, N, 3].
        nonlin: callable (default: torch.relu)
            Nonlinearity to apply to hidden layers.
        hidden_dim: int (default: None)
            Number of hidden units in the MLP.
        output_dim: int (default: 1)
            Number of output dimensions.
        weights_initializer: callable (default: None)
            Initializer for the weights.
        biases_initializer: callable (default: None)
            Initializer for the biases.

    Returns:
        torch.Tensor: The output tensor after filtering.
    """
    # Compute F_2
    F_2_out = F_2(rbf_inputs, rij, nonlin=nonlin, hidden_dim=hidden_dim, output_dim=output_dim,
                  weights_initializer=weights_initializer, biases_initializer=biases_initializer)
    
    input_dim = layer_input.shape[-1]
    
    if input_dim == 1:
        # 0 x 2 -> 2
        cg = torch.eye(5, dtype=layer_input.dtype).unsqueeze(-1)
        return torch.einsum('ijk,abfj,bfk->afi', cg, F_2_out, layer_input)
    else:
        raise NotImplementedError("Other Ls not implemented")

def self_interaction_layer_without_biases(inputs, output_dim, weights_initializer=None, biases_initializer=None):
    """
    Apply self-interaction layer without biases.
    
    Args:
        inputs: torch.Tensor
            Input tensor of shape [N, C, 2L+1].
        output_dim: int
            Number of output dimensions.
        weights_initializer: callable (optional)
            Initializer for weights.
        biases_initializer: callable (optional)
            Initializer for biases (not used in this case).

    Returns:
        torch.Tensor: The output tensor after applying the transformation.
    """
    if weights_initializer is None:
        weights_initializer = torch.nn.init.orthogonal_

    input_dim = inputs.shape[1]  # Assuming the second dimension is input_dim

    # Initialize weights
    w_si = torch.empty(output_dim, input_dim)
    weights_initializer(w_si)  # Initialize weights
    
    # Perform the transformation
    output = torch.einsum('afi,gf->aig', inputs, w_si)
    return output.transpose(1, 2)

def self_interaction_layer_with_biases(inputs, output_dim, weights_initializer=None, biases_initializer=None):
    """
    Apply self-interaction layer with biases.
    
    Args:
        inputs: torch.Tensor
            Input tensor of shape [N, C, 2L+1].
        output_dim: int
            Number of output dimensions.
        weights_initializer: callable (optional)
            Initializer for weights.
        biases_initializer: callable (optional)
            Initializer for biases.

    Returns:
        torch.Tensor: The output tensor after applying the transformation.
    """
    if weights_initializer is None:
        weights_initializer = torch.nn.init.orthogonal_

    if biases_initializer is None:
        biases_initializer = torch.zeros_

    input_dim = inputs.shape[1]  # Assuming the second dimension is input_dim

    # Initialize weights and biases
    w_si = torch.empty(output_dim, input_dim)
    b_si = torch.empty(output_dim)
    
    weights_initializer(w_si)  # Initialize weights
    biases_initializer(b_si)  # Initialize biases

    # Perform the transformation
    output = torch.einsum('afi,gf->aig', inputs, w_si) + b_si
    return output.transpose(1, 2)

def convolution(input_tensor_list, rbf, unit_vectors, weights_initializer=None, biases_initializer=None):
    """
    Perform convolution for each input tensor in input_tensor_list.
    
    Args:
        input_tensor_list: dict
            A dictionary of input tensors with keys 0 and 1.
        rbf: torch.Tensor
            Radial basis function inputs.
        unit_vectors: torch.Tensor
            Unit vectors for the operation.
        weights_initializer: callable (optional)
            Initializer for the weights.
        biases_initializer: callable (optional)
            Initializer for the biases.

    Returns:
        dict: Dictionary of output tensors after the convolution operation.
    """
    output_tensor_list = {0: [], 1: []}
    
    for key in input_tensor_list:
        for tensor in input_tensor_list[key]:
            output_dim = tensor.shape[1]  # Assuming the second dimension is output_dim

            # Filter 0
            tensor_out = filter_0(tensor, rbf, output_dim=output_dim,
                                  weights_initializer=weights_initializer,
                                  biases_initializer=biases_initializer)
            m = 0 if tensor_out.shape[-1] == 1 else 1
            output_tensor_list[m].append(tensor_out)

            # Filter 1 (output 0)
            tensor_out = filter_1_output_0(tensor, rbf, unit_vectors, output_dim=output_dim,
                                           weights_initializer=weights_initializer,
                                           biases_initializer=biases_initializer)
            m = 0 if tensor_out.shape[-1] == 1 else 1
            output_tensor_list[m].append(tensor_out)

            # Filter 1 (output 1)
            tensor_out = filter_1_output_1(tensor, rbf, unit_vectors, output_dim=output_dim,
                                           weights_initializer=weights_initializer,
                                           biases_initializer=biases_initializer)
            m = 0 if tensor_out.shape[-1] == 1 else 1
            output_tensor_list[m].append(tensor_out)

    return output_tensor_list

def self_interaction(input_tensor_list, output_dim, weights_initializer=None, biases_initializer=None):
    """
    Apply self-interaction layers with and without biases, depending on the key in input_tensor_list.
    
    Args:
        input_tensor_list: dict
            A dictionary of input tensors with keys 0 and 1.
        output_dim: int
            Number of output dimensions.
        weights_initializer: callable (optional)
            Initializer for weights.
        biases_initializer: callable (optional)
            Initializer for biases.

    Returns:
        dict: Dictionary of output tensors after applying self-interaction.
    """
    output_tensor_list = {0: [], 1: []}
    
    for key in input_tensor_list:
        for tensor in input_tensor_list[key]:
            if key == 0:
                tensor_out = self_interaction_layer_with_biases(tensor, output_dim,
                                                               weights_initializer=weights_initializer,
                                                               biases_initializer=biases_initializer)
            else:
                tensor_out = self_interaction_layer_without_biases(tensor, output_dim,
                                                                  weights_initializer=weights_initializer,
                                                                  biases_initializer=biases_initializer)

            m = 0 if tensor_out.shape[-1] == 1 else 1
            output_tensor_list[m].append(tensor_out)

    return output_tensor_list

def nonlinearity(input_tensor_list, nonlin=torch.nn.functional.elu, biases_initializer=None):
    """
    Apply nonlinearity to the input tensor list, with rotation-equivariant behavior.
    
    Args:
        input_tensor_list: dict
            A dictionary of input tensors with keys 0 and 1.
        nonlin: callable (optional)
            The nonlinearity function to apply (default: ELU).
        biases_initializer: callable (optional)
            Initializer for biases.

    Returns:
        dict: Dictionary of output tensors after applying the nonlinearity.
    """
    output_tensor_list = {0: [], 1: []}

    for key in input_tensor_list:
        for tensor in input_tensor_list[key]:
            if key == 0:
                tensor_out = rotation_equivariant_nonlinearity(tensor, nonlin=nonlin,
                                                               biases_initializer=biases_initializer)
            else:
                tensor_out = rotation_equivariant_nonlinearity(tensor, nonlin=nonlin,
                                                               biases_initializer=biases_initializer)

            m = 0 if tensor_out.shape[-1] == 1 else 1
            output_tensor_list[m].append(tensor_out)

    return output_tensor_list

def concatenation(input_tensor_list):
    """
    Concatenate input tensors along the channel axis.
    
    Args:
        input_tensor_list: dict
            A dictionary of input tensors with keys 0 and 1.

    Returns:
        dict: Dictionary of concatenated tensors for each key.
    """
    output_tensor_list = {0: [], 1: []}

    for key in input_tensor_list:
        # Concatenate along the channel axis (dimension -2)
        concatenated_tensor = torch.cat(input_tensor_list[key], dim=-2)
        output_tensor_list[key].append(concatenated_tensor)

    return output_tensor_list


## TFN Models ## 

class TFN_Tetris_Model(nn.Module):
    def __init__(self, layer_dims, rbf_low=0.0, rbf_high=3.5, rbf_count=4, num_classes=1):
        """
        Args:
            layer_dims (list): List of layer dimensions, e.g., [1, 4, 4, 4].
            rbf_low (float): The low boundary of the RBF range.
            rbf_high (float): The high boundary of the RBF range.
            rbf_count (int): The number of radial basis functions.
            num_classes (int): The number of output classes.
        """
        super(TFN_Tetris_Model, self).__init__()

        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1
        self.num_classes = num_classes
        
        # Set RBF parameters and calculate spacing and centers
        self.rbf_low = rbf_low
        self.rbf_high = rbf_high
        self.rbf_count = rbf_count
        self.rbf_spacing = (self.rbf_high - self.rbf_low) / self.rbf_count
        self.centers = torch.linspace(self.rbf_low, self.rbf_high, self.rbf_count)
        
        # Define the layers
        self.embed = self_interaction_layer_without_biases(torch.ones((4, 1, 1)), layer_dims[0])
        
        # Define convolution layers and fully connected layers
        self.conv_layers = nn.ModuleList()
        self.fc_layer = nn.Linear(layer_dims[-2], num_classes)
    
    def get_inputs(self, r):
        """
        Calculates the pairwise differences (rij), distances (dij), and radial basis functions (rbf)
        based on input positions (r) and the pre-initialized RBF parameters.
        """
        rij = difference_matrix(r)
        dij = distance_matrix(r)
        gamma = 1. / self.rbf_spacing
        rbf = torch.exp(-gamma * (dij.unsqueeze(-1) - self.centers)**2)
        return rij, dij, rbf

    def forward(self, r):
        """
        Forward pass through the network.
        Args:
            r (torch.Tensor): Input tensor of shape [N, 3] where N is the number of points in the shape.
        """
        # Get rij, dij, rbf from the inputs
        rij, dij, rbf = self.get_inputs(r)

        # Embed layer (output: [N, layer1_dim, 1])
        embed = self.embed
        
        input_tensor_list = {0: [embed]}
        
        for layer, layer_dim in enumerate(self.layer_dims[1:]):
            input_tensor_list = convolution(input_tensor_list, rbf, rij)
            input_tensor_list = concatenation(input_tensor_list)
            input_tensor_list = self_interaction(input_tensor_list, layer_dim)
            input_tensor_list = nonlinearity(input_tensor_list)
        
        tfn_scalars = input_tensor_list[0][0]
        tfn_output = tfn_scalars.mean(dim=0).squeeze()

        # Fully connected layer: [num_classes]
        output = self.fc_layer(tfn_output)
        return output