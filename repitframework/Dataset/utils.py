import numpy as np
from typing import List, Union, Optional, Tuple
from pathlib import Path


from ..Metrics.ResidualNaturalConvection import residual_mass

def hard_constraint_bc(
    data_list: np.ndarray,
    extended_vars_list: List[str],
    left_wall_temperature: float = 307.75,
    right_wall_temperature: float = 288.15
    )-> List[np.ndarray]:
    '''
    We are just encoding the boundary conditions as an extra layer to the predicted values. 
    Also, while preparing the training data, we need to ensure that the boundary conditions are 
    imposed the same way.
    
    Args
    ----
    data_list: np.ndarray.
        - [vars, grid_y, grid_x] or [vars, grid_y, grid_x, grid_z]
    extended_vars_list: List[str]
        - List of variables. Example: ["U_x", "U_y", "T"]
    left_wall_temperature: float
        - Temperature of the left wall. Default: 288.15
    right_wall_temperature: float
        - Temperature of the right wall. Default: 307.75

    ux and uy are noSlip conditions, hence they should be zero.

    The BC for temperature: 
        - Left wall: 288.15
        - Right wall: 307.75
        - Top wall: adiabatic
        - Bottom wall: adiaabatic
    '''
    ux_index = extended_vars_list.index("U_x")
    uy_index = extended_vars_list.index("U_y")
    t_index = extended_vars_list.index("T")


    ux_matrix = data_list[ux_index]
    uy_matrix = data_list[uy_index]
    t_matrix = data_list[t_index]
    del data_list # Free up memory

    ux_matrix = np.pad(ux_matrix, ((1,1),(1,1)), mode="constant", constant_values=0)
    uy_matrix = np.pad(uy_matrix, ((1,1),(1,1)), mode="constant", constant_values=0)
    t_matrix = np.pad(t_matrix, ((1,1),(1,1)), mode="constant", constant_values=0)

    '''
    Applying the boundary conditions:
        - Left wall is the hot wall
        - Right wall is the cold wall
    '''
    t_matrix[0, :] = t_matrix[1, :]
    t_matrix[-1, :] = t_matrix[-2, :]
    t_matrix[:, 0] = left_wall_temperature
    t_matrix[:, -1] = right_wall_temperature

    temp_list = []
    temp_list.insert(ux_index, ux_matrix)
    temp_list.insert(uy_index, uy_matrix)
    temp_list.insert(t_index, t_matrix)

    return temp_list

def add_feature(input_matrix:np.ndarray) -> np.ndarray:
    '''
    This function is used to add correlated features to the data.
                |        |
                | (x-1,y)|                 
    ----------|--------|---------      
        (x,y-1)  |  (x,y) | (x,y+1)   ---> [(x,y), (x-1,y), (x+1,y), (x,y-1), (x,y+1)]
    ----------|--------|---------
                |(x-1,y) | 
                |        |
                    
    Args
    ----
    input_matrix: np.ndarray
        The input data matrix. Example Shape: [202,202]

    Returns
    -------
    correlated_features: np.ndarray
        The correlated features. Example Shape: [5, 200, 200]

    '''
    window_shape = (3, 3)
    sliding_window = np.lib.stride_tricks.sliding_window_view(input_matrix, window_shape)
    x,y = window_shape[0] // 2, window_shape[1] // 2 
    correlated_features = [
        sliding_window[:,:,x,y],
        sliding_window[:,:,x-1,y],
        sliding_window[:,:,x+1,y],
        sliding_window[:,:,x,y-1],
        sliding_window[:,:,x,y+1]
    ]
    return np.stack(correlated_features, axis=0)

def parse_numpy(
    dataset_file: Union[str, Path],
    grid_x: int,
    grid_y: int,
    grid_z: int = 1,
    data_dim: int = 2,
) -> np.ndarray:
    """
    Loads a .npy file and reshapes as scalar or vector, tailored for 2D/3D.
    Returns parsed numpy array.

    Args
    ----
    dataset_file: Union[str, Path]
        Path to the dataset file.
    grid_x: int
        Number of grid points in the x direction.
    grid_y: int
        Number of grid points in the y direction.
    grid_z: int
        Number of grid points in the z direction. Default is 1.
    data_dim: int
        Dimension of the data. Default is 2.

    Returns
    -------
    np.ndarray
        Parsed numpy array reshaped according to grid dimensions.
    """
    data = np.load(dataset_file)
    expected_len = grid_x * grid_y * grid_z
    if data.shape[0] != expected_len:
        raise ValueError(
            f"Data shape mismatch: expected {expected_len}, got {data.shape[0]}"
        )
    if len(data.shape) == 2:
        data_dim = data.shape[-1]
        if data_dim == 1:
            return data[:, 0].reshape(grid_y, grid_x)
        elif data_dim == 2:
            x_data = data[:, 0].reshape(grid_y, grid_x)
            y_data = data[:, 1].reshape(grid_y, grid_x)
            return np.stack([x_data, y_data], axis=-1)
        elif data_dim == 3:
            x_data = data[:, 0].reshape(grid_y, grid_x)
            y_data = data[:, 1].reshape(grid_y, grid_x)
            z_data = data[:, 2].reshape(grid_y, grid_x)
            return np.stack([x_data, y_data, z_data], axis=-1)
        else:
            raise NotImplementedError("Beyond 3D is not supported.")
    elif len(data.shape) == 1:
        return data.reshape(grid_y, grid_x)
    else:
        raise NotImplementedError("Unsupported data shape!")
    
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
        List of input numpy arrays at specific time step with shape [num_features, grid_y, grid_x].
    """
    match output_dims:
        case "BD":
            inputs = [inp.reshape(inp.shape[0], -1).T for inp in inputs]
            inputs = np.concatenate(inputs, axis=0)
        case "BCD":
            inputs = [inp.reshape(inp.shape[0], -1) for inp in inputs]
            inputs = np.stack(inputs, axis=0)
        case _:
            inputs = np.stack(inputs, axis=0)
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
    ux_matrix = vel_data[:,:,0]
    uy_matrix = vel_data[:,:,1]
    return residual_mass(ux_matrix, uy_matrix, order="C")