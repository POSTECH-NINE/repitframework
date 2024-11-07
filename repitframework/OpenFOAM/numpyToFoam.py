import numpy as np
from pathlib import Path
import subprocess
import re
from repitframework import config

def parse_numpy(data: np.ndarray) -> str:
    """
    Convert a NumPy array to a string representation suitable for OpenFOAM field files.

    Parameters:
    - data (np.ndarray): The NumPy array to convert.

    Returns:
    - str: The string representation of the data.

    Raises:
    - ValueError: If the data shape is not supported.
    """
    if data.ndim == 1:
        # 1D array of scalars
        return  '\n'.join(map(str, data))
    elif data.ndim == 2:
        if data.shape[1] == 1:
            # 2D array with a single column, treat as scalar field
            return '\n'.join(map(str, data[:, 0]))
        else:
            # 2D array with multiple columns, treat as vector field
            lines = ['(' + ' '.join(map(str, row)) + ')' for row in data]
            return '\n'.join(lines)
    else:
        raise ValueError("Data shape not supported. Aborting conversion from numpy to OpenFOAM.")


def numpyToFoam(file_path:Path, 
                variable:str, 
                previousCFD_time, 
                solver_dir:Path,
                presentCFD_time) -> bool:
    """
    This function takes a numpy file and writes it to an OpenFOAM file.
    Example: 
    If we have a numpy file U_3.npy, we can write it to the OpenFOAM file U at time t=3.

    Args:
    file_path: Path to the numpy file.
    variable: The OpenFOAM variable name.
    latestCFD_time: The current time step.

    **************** NOTE ****************
    The latestCFD_time should be the time step for which the OpenFOAM file is already present. 
    Because, we need to copy format to the present time step for which we are trying to run the 
    simulation.
    **************************************

    solver_dir: The directory of the solver.

    Returns:
    True if the function executes successfully.
    """
    
    latestCFD_time_dir = Path(str(solver_dir) + f"/{previousCFD_time}") # time directory for current time
    next_time_dir      = Path(str(solver_dir) + f"/{presentCFD_time}") # time directory for next time (latestCFD_time + 1)
    openfoam_var_path  = Path(str(next_time_dir) + f"/{variable}")      # openfoam variable path: where we write the numpy data

    subprocess.run(["cp", "-r", latestCFD_time_dir, next_time_dir]) if not next_time_dir.exists() else None     # copy the foam time directory to t+1
    
    # numpy file processing:
    data = np.load(file_path)
    data_str = "(\n" + parse_numpy(data) + "\n)\n;" # convert numpy data to OpenFOAM format

    with open(openfoam_var_path, "r") as file:
        foam_data_temp = file.read()
        foam_data = re.sub(r'(location\s*)"([^"]*)"',rf'\1"{presentCFD_time}"',foam_data_temp) # update the location to the next time step
        foam_data = re.sub(r'\([\s\S]*?\)\n;', f'{data_str}',foam_data,count=1) # Update the data in the OpenFOAM file; count=1 to replace only the first occurrence.

    with open(openfoam_var_path, "w") as file:
        file.write(foam_data)
    return True    

    
if __name__ == "__main__":
    from utils import run_solver, update_time_foamDictionary

    previousCFD_time:int = 2
    presentCFD_time:int = 5
    time_step:int = 1
    openfoam_config = config.OpenfoamConfig()
    solver_dir:Path = openfoam_config.solver_dir

    for var in openfoam_config.data_vars:
        numpy_file_path = openfoam_config.assets_dir / openfoam_config.case_name / f"{var}_{presentCFD_time}.npy"
        numpyToFoam(numpy_file_path, var, previousCFD_time, solver_dir, presentCFD_time)

    check_time_update:bool = update_time_foamDictionary(solver_dir=solver_dir,
                                                    present_time=presentCFD_time,
                                                    end_time=presentCFD_time + 2*time_step)
    
    if check_time_update:
        run_solver()