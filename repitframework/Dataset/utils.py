import numpy as np
from typing import List, Union, Optional, Tuple
from pathlib import Path


from ..Metrics.ResidualNaturalConvection import residual_mass

def hard_constraint_bc(
    data: np.ndarray,
    extended_vars_list: List[str],
    left_wall_temperature: float = 307.75,
    right_wall_temperature: float = 288.15
) -> List[np.ndarray]:
    """
    Encodes hard boundary conditions for 1D, 2D, or 3D data by padding the data array.

    This function assumes a standard axis order for spatial dimensions, from slowest to
    fastest changing: (Z, Y, X).
    - 1D: [vars, grid_x]
    - 2D: [vars, grid_y, grid_x]
    - 3D: [vars, grid_z, grid_y, grid_x]

    Args
    ----
    data: np.ndarray
        - The input data array with shape [vars, ...spatial_dims].
    extended_vars_list: List[str]
        - List of variable names corresponding to the first axis of data_list.
          Example: ["U_x", "U_y", "U_z", "T"].
    left_wall_temperature: float
        - Dirichlet condition for Temperature on the left wall (minimum x-boundary).
    right_wall_temperature: float
        - Dirichlet condition for Temperature on the right wall (maximum x-boundary).

    Returns
    -------
    List[np.ndarray]
        - A list of numpy arrays, each with the boundary conditions applied, in the
          same order as the input variables.

    Boundary Conditions Logic
    -------------------------
    - **Velocity (U_x, U_y, U_z):** No-slip condition (value is 0) on all boundaries.
    - **Temperature (T):**
        - **X-axis (Left/Right):** Dirichlet condition (fixed temperature).
        - **Y-axis (Top/Bottom, 2D/3D):** Adiabatic (zero-gradient Neumann).
        - **Z-axis (Front/Back, 3D):** Adiabatic (zero-gradient Neumann).
    """
    processed_vars = {}
    
    # Determine spatial dimensions directly from the input array's dimensions.
    # The number of spatial dimensions is the total number of dimensions minus one (for the 'vars' axis).
    spatial_dims = data.ndim - 1
    
    # Create a dynamic padding width that applies a pad of 1 to all spatial dimensions.
    # e.g., for 2D this becomes ((1, 1), (1, 1))
    # e.g., for 3D this becomes ((1, 1), (1, 1), (1, 1))
    pad_width = ((1, 1),) * spatial_dims

    for i, var_name in enumerate(extended_vars_list):
        matrix = data[i]
        
        # 1. Pad the matrix with a layer of zeros.
        # For velocity components (no-slip), this is the final state.
        # For temperature, we pad with 0 first and then overwrite the boundary values.
        padded_matrix = np.pad(matrix, pad_width, mode="constant", constant_values=0)

        # 2. Apply specific boundary conditions only for the Temperature variable.
        if var_name == "T":
            # --- X-axis (Left/Right Walls) - Applies to 1D, 2D, & 3D ---
            # The Ellipsis (...) automatically handles all preceding dimensions.
            # This is equivalent to [:, :, 0] in 3D or [:, 0] in 2D.
            padded_matrix[..., 0] = left_wall_temperature   # Left wall
            padded_matrix[..., -1] = right_wall_temperature  # Right wall

            # --- Y-axis (Top/Bottom Walls) - Applies to 2D & 3D ---
            if spatial_dims >= 2:
                # Adiabatic condition: value at boundary equals the value of the neighbor inside.
                # Top wall (index 0 of the second-to-last axis)
                padded_matrix[..., 0, :] = padded_matrix[..., 1, :]
                # Bottom wall (index -1 of the second-to-last axis)
                padded_matrix[..., -1, :] = padded_matrix[..., -2, :]

            # --- Z-axis (Front/Back Walls) - Applies to 3D only ---
            if spatial_dims == 3:
                # Adiabatic condition for the Z-axis (the first spatial axis).
                # Front wall (index 0)
                padded_matrix[0, ...] = padded_matrix[1, ...]
                # Back wall (index -1)
                padded_matrix[-1, ...] = padded_matrix[-2, ...]
        
        processed_vars[var_name] = padded_matrix

    # Return the processed arrays in the original order.
    return [processed_vars[name] for name in extended_vars_list]

def add_feature(input_matrix: np.ndarray) -> np.ndarray:
    """
    Extracts correlated features from a data matrix for 1D, 2D, or 3D cases.
    
    This function implements a stencil operation that captures the value of each point 
    and its immediate neighbors along each axis. The number of features returned
    is equal to (2 * num_dimensions + 1).

    - 1D Input -> 3 features: [center, left, right]
    - 2D Input -> 5 features: [center, top, bottom, left, right]
    - 3D Input -> 7 features: [center, front, back, top, bottom, left, right]

    Args
    ----
    input_matrix: np.ndarray
        The input data matrix.
        Example Shapes: [202], [202, 202], or [50, 50, 50]

    Returns
    -------
    correlated_features: np.ndarray
        The correlated features stacked along the first axis. The spatial dimensions
        of the output are reduced by 2 compared to the input matrix.
        Example Output Shapes: [3, 200], [5, 200, 200], or [7, 48, 48, 48]
    """
    # Determine the number of spatial dimensions (1, 2, or 3).
    spatial_dims = input_matrix.ndim
    
    # Create a window of size 3 for each spatial dimension.
    # e.g., (3,) for 1D, (3, 3) for 2D, (3, 3, 3) for 3D.
    window_shape = (3,) * spatial_dims
    
    # Use sliding_window_view to create a view of the data without copying it.
    # The output dimensions will be smaller than the input by (window_size - 1).
    sliding_window = np.lib.stride_tricks.sliding_window_view(input_matrix, window_shape)
    
    correlated_features = []
    
    # 1. Define the index for the center point of the window.
    # This will be (1,) for 1D, (1, 1) for 2D, etc.
    center_idx_tuple = (1,) * spatial_dims
    
    # The slice needs to cover the output grid dimensions first (using Ellipsis),
    # followed by the index within the window.
    center_slice = (Ellipsis,) + center_idx_tuple
    correlated_features.append(sliding_window[center_slice])
    
    # 2. Iterate through each dimension to get the neighbors.
    for d in range(spatial_dims):
        # Create a mutable list from the center index tuple to modify it.
        neighbor_idx = list(center_idx_tuple)
        
        # Get the neighbor "before" the center point on the current axis (index 0).
        neighbor_idx[d] = 0
        neighbor_before_slice = (Ellipsis,) + tuple(neighbor_idx)
        correlated_features.append(sliding_window[neighbor_before_slice])
        
        # Get the neighbor "after" the center point on the current axis (index 2).
        neighbor_idx[d] = 2
        neighbor_after_slice = (Ellipsis,) + tuple(neighbor_idx)
        correlated_features.append(sliding_window[neighbor_after_slice])
        
    # Stack the collected feature arrays into a single NumPy array.
    return np.stack(correlated_features, axis=0)
    
def parse_numpy(
    dataset_file: Union[str, Path],
    grid_x: int,
    grid_y: int,
    grid_z: int,
    data_dim: int = 3
) -> np.ndarray:
    """
    Loads a .npy file and reshapes it for 1D, 2D, or 3D grids.
    This function handles both scalar fields (e.g., Temperature) and vector
    fields (e.g., Velocity).

    Args
    ----
    dataset_file: Union[str, Path]
        Path to the dataset file (.npy).
    grid_shape: Tuple[int, ...]
        A tuple defining the grid dimensions. The order should be (Z, Y, X).
        - For 1D: (grid_x,)
        - For 2D: (grid_y, grid_x)
        - For 3D: (grid_z, grid_y, grid_x)

    Returns
    -------
    np.ndarray
        Parsed numpy array reshaped according to grid dimensions.
        - Scalar field shape: (*grid_shape)
        - Vector field shape: (*grid_shape, num_components)
    """
    grid_shape = (grid_z, grid_y, grid_x)
    data: np.ndarray = np.load(dataset_file)
    expected_len = np.prod(grid_shape)

    if data.shape[0] != expected_len:
        raise ValueError(
            f"Data shape mismatch: expected {expected_len} grid points, but file has {data.shape[0]} points."
        )

    # Case 1: Scalar data (e.g., pressure, temperature)
    # The loaded data is a 1D array of shape (n_points,)
    if data.ndim == 1:
        # Squeeze the array to remove any singleton dimensions (e.g., (y, x, 1) -> (y, x))
        return data.reshape(grid_shape).squeeze()
    
    # Case 2: Vector data (e.g., velocity)
    # The loaded data is a 2D array of shape (n_points, n_components)
    elif data.ndim == 2:
        num_components = data.shape[-1]
        
        # Reshape each component, squeeze to remove singleton dimensions, and stack.
        components = [data[:, i].reshape(grid_shape).squeeze() for i in range(num_components)]
        return np.stack(components, axis=-1)
        
    else:
        raise NotImplementedError(f"Unsupported data shape with {data.ndim} dimensions!")

    

def denormalize(
    data: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """
    Denormalize data using mean and std.
    """
    return data * std + mean

def normalize(
    data: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    select_dims: Tuple[int, ...] = (0,),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize data using mean and std over selected dims.
    """
    if mean is None:
        mean = np.mean(data, axis=select_dims, keepdims=True)
    if std is None:
        std = np.std(data, axis=select_dims, keepdims=True)
    normalized_data = (data - mean) / (std + np.full_like(std, 1e-18))
    return normalized_data, mean, std

def match_input_dim(
    output_dims:str, inputs: List[np.ndarray]
) -> np.ndarray:
    """
    Reshapes (and stacks) inputs/labels based on output_dims.
    last input is the last time step input, used for prediction.

    Args
    ----
    output_dims: str
        The shape you want to get the data. Example: "BD", "BCD", "BCHW"
    inputs: List[np.ndarray]
        List of input numpy arrays at specific time step with shape [num_features, *grid_shape].


    Implementation
    --------------
    - **BD:**
        - **B:** product(*grid_shape)*time_steps
        - **D:** num_features (variables*5 if feature_selection else variables)
    - **BCD:**
        - **B:** time_steps
        - **C:** num_features (variables*5 if feature_selection else variables)
        - **D:** product(*grid_shape)
    - **BCHW:**
        - **2D Case:**
            - **B:** time_steps
            - **C:** num_features (variables*5 if feature_selection else variables)
            - **H:** grid_y
            - **W:** grid_x
        - **3D Case:**
            - **B:** time_steps * num_features
            - **C:** grid_z
            - **H:** grid_y
            - **W:** grid_x
    """
    match output_dims:
        case "BD":
            inputs = [inp.reshape(inp.shape[0], -1).T for inp in inputs]
            inputs = np.concatenate(inputs, axis=0)
        case "BCD":
            inputs = [inp.reshape(inp.shape[0], -1) for inp in inputs]
            inputs = np.stack(inputs, axis=0)
        case "BCHW":
            inputs: np.ndarray = np.stack(inputs, axis=0)
            if inputs.ndim > 4:
                inputs = inputs.reshape(-1, *inputs.shape[-3:])  # Merge leading dimensions except last 3.
        case _:
            raise ValueError(f"Invalid output_dims: {output_dims}. Must be one of ['BD', 'BCD', 'BCHW'].")
    return inputs


def calculate_residual(dataset_dir: Path,
                       time: Union[int, float],
                       grid_x: int, 
                       grid_y: int, 
                       grid_z: int=1,
                       dims:int=2) -> float:
    '''
    The switching point between Ml-CFD is residual mass, hence this functionality
    must not be neglected.

    Note:
    ----
    The framework expects the velocity data to be in the form of a 2D numpy array
    with shape (grid_y, grid_x, 2) where the last dimension contains the
    x and y components of the velocity.
    '''
    data_path = dataset_dir / f"U_{time}.npy"
    vel_data = parse_numpy(
        data_path,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
        data_dim=dims
    )
    return residual_mass(vel_data)