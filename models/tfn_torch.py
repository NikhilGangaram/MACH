import torch
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
