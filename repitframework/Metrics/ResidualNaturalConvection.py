import numpy as np

from ..config import OpenfoamConfig

foam_config = OpenfoamConfig()
ny = foam_config.grid_y
nx = foam_config.grid_x
grid_step = foam_config.grid_step
time_step = foam_config.write_interval

def residual_mass(
    velocity_field: np.ndarray
) -> float:
    """
    Computes the mass conservation residual for a 1D, 2D, or 3D velocity field.
    
    The function calculates the squared divergence of the velocity field using a
    second-order central difference scheme and normalizes it.

    Formula: 
    Rs_mass = sum({divergence(U)}^2) / num_points
    where divergence(U) = d(u_x)/dx + d(u_y)/dy + d(u_z)/dz

    Args:
    ----
    velocity_field: np.ndarray
        A numpy array representing the vector velocity field.
        - Shape must be (*grid_shape, num_components).
        - Assumes a grid axis order of (Z, Y, X) and a component order of (u_x, u_y, u_z).
    grid_step: float
        The uniform distance between grid points.

    Returns:
    -------
    float
        The sum of the squared mass residual.
    """
    num_spatial_dims = velocity_field.ndim - 1
    
    # Check for consistency
    num_components = velocity_field.shape[-1]
    if num_spatial_dims != num_components:
        raise ValueError(
            f"Number of spatial dimensions ({num_spatial_dims}) must match "
            f"the number of velocity components ({num_components})."
        )

    divergence = np.zeros(np.array(velocity_field.shape[:-1]) - 2)

    # Calculate divergence: d(u_x)/dx + d(u_y)/dy + d(u_z)/dz
    for i in range(num_spatial_dims):
        component_index = i
        # This maps component to axis: u_x -> x, u_y -> y, u_z -> z
        axis_index = num_spatial_dims - 1 - i

        component_field = velocity_field[..., component_index]

        # Create slicers for the central difference calculation
        # These create a view on the "core" of the domain, excluding boundaries
        core_slicer = [slice(1, -1)] * num_spatial_dims
        
        # Slice for the point ahead along the current axis
        plus_slicer = list(core_slicer)
        plus_slicer[axis_index] = slice(2, None)
        
        # Slice for the point behind along the current axis
        minus_slicer = list(core_slicer)
        minus_slicer[axis_index] = slice(None, -2)

        partial_derivative = (component_field[tuple(plus_slicer)] - component_field[tuple(minus_slicer)]) / (2 * grid_step)
        divergence += partial_derivative

    # Calculate the final residual
    residual_sq = divergence * divergence
    num_points = np.prod(velocity_field.shape[:-1])
    residual_sum = residual_sq.sum() / num_points
    
    return residual_sum

def residual_momentum(ux_matrix:np.ndarray, ux_matrix_prev:np.ndarray, uy_matrix:np.ndarray, t_matrix:np.ndarray):
    '''
    Compute the residual: momentum conservation
    Formula:
    Rs_mom = {d(ux)/dt + ux*d(ux)/dx + uy*d(ux)/dy - 1.831e-05/(348.33/alpha)*d^2(ux)/dx^2 - 9.81/293*(293-alpha)}^2.sum()/(ny*nx)
    '''
    mom_1 = ux_matrix[1:ny-1,1:nx-1] - ux_matrix_prev[1:ny-1,1:nx-1]
    mom_3 = ux_matrix[1:ny-1,1:nx-1]*(ux_matrix[2:ny,1:nx-1] - ux_matrix[0:ny-2,1:nx-1])
    mom_4 = uy_matrix[1:ny-1,1:nx-1]*(ux_matrix[1:ny-1,2:nx] - ux_matrix[1:ny-1,0:nx-2])
    mom_5_2 = ux_matrix[1:ny-1,2:nx] - 2*ux_matrix[1:ny-1,1:nx-1] + ux_matrix[1:ny-1,0:nx-2] 
    mom_5 = 1.831e-05/(348.33/t_matrix[1:ny-1,1:nx-1])*(mom_5_2) 
    mom_6 = 9.81/293*(293-t_matrix[1:ny-1,1:nx-1])

    Rs_mom = mom_1/time_step +  mom_3/(2*grid_step) + mom_4/(2*grid_step) - mom_5/(grid_step*grid_step) - mom_6
    Rs_mom_sq = Rs_mom*Rs_mom
    Rs_mom_sum = Rs_mom_sq.sum()/(ny*nx)
    return Rs_mom_sum

def residual_heat(ux_matrix:np.ndarray, uy_matrix:np.ndarray, t_matrix:np.ndarray, t_matrix_prev:np.ndarray):
    '''
    Compute the residual: heat conservation
    Formula:
    Rs_heat = {d(t)/dt + ux*d(t)/dx + uy*d(t)/dy - 0.14*(t-293)+21.7/1e6*d^2(t)/dx^2}^2.sum()/(ny*nx)
    TODO: Check the formula

    Arguments:
    ux_matrix: np.ndarray: matrix of x-velocity, shape = [200,200]
    uy_matrix: np.ndarray: matrix of y-velocity, shape = [200,200]
    t_matrix: np.ndarray: matrix of temperature, shape = [200,200]
    t_matrix_prev: np.ndarray: matrix of temperature at previous time step, shape = [200,200]

    Return:
    Rs_heat_sum: float: sum of Rs_heat
    '''
    tdiff_matrix = (0.14*(t_matrix[1:ny-1,1:nx-1] - 293)+ 21.7)/1000000
    heat_1 = t_matrix[1:ny-1,1:nx-1] - t_matrix_prev[1:ny-1,1:nx-1]
    heat_2 = (t_matrix[2:ny,1:nx-1] - t_matrix[0:ny-2,1:nx-1])*(ux_matrix[1:ny-1,1:nx-1])
    heat_3 = (t_matrix[1:ny-1,2:nx] - t_matrix[1:ny-1,0:nx-2])*(uy_matrix[1:ny-1,1:nx-1])
    heat_4 = tdiff_matrix*(t_matrix[1:ny-1,2:nx] - 2*t_matrix[1:ny-1,1:nx-1] + t_matrix[1:ny-1,0:nx-2])

    Rs_heat = heat_1/time_step + heat_2/(2*grid_step) + heat_3/(2*grid_step) - heat_4/(grid_step*grid_step)
    Rs_heat_sq = Rs_heat*Rs_heat
    Rs_heat_sum = Rs_heat_sq.sum()/(ny*nx)
    return Rs_heat_sum


if __name__ == "__main__":
    # Test residual_mass
    ux_matrix = np.random.rand(200,200)
    ux_matrix_prev = np.random.rand(200,200)
    uy_matrix = np.random.rand(200,200)
    t_matrix = np.random.rand(200,200)
    t_matrix_prev = np.random.rand(200,200)
    Rs_mass_sum = residual_mass(ux_matrix,uy_matrix)
    Rs_mom_sum = residual_momentum(ux_matrix, ux_matrix_prev, uy_matrix, t_matrix)
    Rs_heat_sum = residual_heat(ux_matrix, uy_matrix, t_matrix, t_matrix_prev)
    print(Rs_mass_sum, Rs_mom_sum, Rs_heat_sum)