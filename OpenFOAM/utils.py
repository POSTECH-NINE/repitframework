import Ofpp
from pathlib import Path
import os
import subprocess
import re
import logging
import numpy as np

__all__ = [ "parse_to_numpy", "run_the_solver", "read_mesh_type", "read_solver_type", "update_time_steps"]

# Setting the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler(Path(__file__).parent.parent.resolve() / "logs" / "OpenFOAM.log")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

def manage_assets(solver_dir:Path = None, assets_dir:Path = None) -> Path:
    '''
    If we are trying out with different cases, this function is to put them nicely inside the assets directory with the name
    of the case as classifiers.  
    '''    
    case_name = solver_dir.name
    assets_dir = Path.joinpath(assets_dir, case_name)
    assets_dir.mkdir(parents=True, exist_ok=True)
    return assets_dir

def parse_to_numpy(solver_dir:Path = None, assets_dir:Path = None, variables:list = ["U", "p", "T"]) -> None:
    '''
    OpenFOAM stores the data in the form of Dictionary(OpenFOAM type) files. But to train the model it will be easier to change to tensors 
    if we can convert them to numpy arrays. This function does the same. To carry out this task, we can use the Ofpp library.

    Args:
    solver_dir: str: The path to the solver directory where OpenFOAM has stored the data after running the solver.
    assets_dir: str: The path to the assets directory where we want to save the data in the numpy format. 
    '''
    # List the time directories
    try:
        logger.debug("Listing the time directories!")
        command_list_dir = ["foamListTimes", "-case", solver_dir]
        time_list = subprocess.run(command_list_dir, capture_output=True, text=True).stdout.split("\n")
        assert time_list != [""], f"No time directories found in the {solver_dir} directory"
        time_list = [i for i in time_list if i]
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in listing the time directories: {e}")
    time_directories = [Path(solver_dir,i) for i in time_list]
    
    # Parse the data to numpy
    for time_dir in time_directories:
        for var in variables:
            try:
                data = Ofpp.parse_internal_field(Path(time_dir, var))
                logger.debug(f"Data parsed to numpy:{var}_{time_dir.name} --> {data.shape}")
                np.save(Path(assets_dir, f"{var}_{time_dir.name}.npy"), data)
            except Exception as e:
                logger.error(f"Error in parsing the data to numpy: {e}")
    
    try: 
        logger.debug("Deleting the time directories!")
        command_deldirs = ["foamListTimes", "-case", solver_dir, "-rm", "-time",",".join(time_list[:-1])]
        subprocess.run(command_deldirs,capture_output=True, text=True)
        logger.debug("Time directories deleted successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in deleting the time directories: {e}")
    return None

# TODO: Don't forget to write test cases for every functions inside OpenFOAM module.
def run_the_solver(solver_dir:Path = None, assets_dir:Path = Path(__file__).parent.parent.resolve()/"Assets", mesh_type:str = "blockMesh"):
    '''
    This function aims at running the CFD solver of your interest. 
    For example:
    In OpenFOAM, what you normally do is, clone the existing solver similar to the case you want to solve,
    modify different parameters according to your requirements and then run the solver.
    To run the solver you need to do these things: 
    - Go to the solver directory
    - Create the mesh
    - Run the solver

    So, this function tries lift off these steps from your shoulder.

    Args:
    solver_path: str: The path to the solver directory. If not provided, it will ask you to provide the path.
    assets_dir: str: The path to the assets directory. 

    Returns:
    None

    Functionality: 
    Saves the data in the Assets directory. 

    '''
    mesh_ = read_mesh_type(solver_dir=solver_dir, mesh_type=mesh_type)
    solver_ = read_solver_type(solver_dir=solver_dir)

    # cd to the solver directory
    # logger.debug(f"Changing the directory to the solver directory: {solver_dir}")
    # os.chdir(solver_dir)

    # Create the mesh
    try:
        logger.debug("Creating the mesh!")
        command = [mesh_, "-case", solver_dir]
        mesh_result = subprocess.run(command, capture_output=True, text=True)
        logger.debug(f"\n Mesh Output: {mesh_result.stdout}\n")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in creating the mesh: {e}")

    # Run the solver
    try:
        print("Running the CFD solver!")
        command = [solver_, "-case", solver_dir]
        solver_result = subprocess.run(command, capture_output=True, text=True)
        logger.debug(f"\n Solver Output: {solver_result.stdout}\n")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in running the solver: {e}")

    # Save the data in the assets directory:
    assets_path = manage_assets(solver_dir=solver_dir,assets_dir=assets_dir)
    parse_to_numpy(solver_dir=solver_dir, assets_dir=assets_path)
    return None

def read_mesh_type(solver_dir:Path = None, mesh_type:str = None) -> str:
    '''
    Ensures the meshing technique used to generate the mesh. It seems there is no direct way to get the mesh type.
    So, we can ask the user to provide the mesh type.

    - Example: blockMesh, snappyHexMesh, etc.
    - mesh_type information is also saved in the OpenfoamConfig class for future reference. It might seem little non-pythonic but 
    it feels like a right thing to do. #TODO: Change in future if you find a better way to do this, refer to solver_type also.  
    '''
    if mesh_type is None:
        mesh_type = input("Please enter the mesh type: e.g. blockMesh, snappyHexMesh, etc.\n")
    return mesh_type

def read_solver_type(solver_dir:Path = None) -> str:
    '''
    Ensures the solver type used to solve the problem. foamDictionary command comes in very handy to get the solver type.

    - Example: foamDictionary system/controlDict -entry application gives the application information of the case. 
    applying regular expression to the received output, we can get the solver type.
    - solver_type information is also saved in the OpenfoamConfig class for future reference.

    Args:
    solver_dir: str: The path to the solver directory.

    Returns:
    solver_type: str: The solver type used to solve the problem.
    '''

    control_dict_path = Path.joinpath(solver_dir, "system", "controlDict")
    if not control_dict_path.exists():
        raise FileNotFoundError(f"controlDict file not found in the directory: {solver_dir}")
    
    try:
        command = ["foamDictionary",control_dict_path, "-entry", "application"]
        command_result = subprocess.run(command, capture_output=True, text=True)
        solver_type = re.search(r'application\s+(\w+);\n', command_result.stdout).group(1)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in reading the solver type: {e}")
    
    return solver_type

def update_time_steps():
    pass

if __name__ == "__main__":
    solver_dir = Path("/home/ninelab/repitframework/Solvers/natural_convection")
    assets_dir = Path("/home/ninelab/repitframework/Assets")

    parse_to_numpy(solver_dir=solver_dir, assets_dir=assets_dir)
    run_the_solver(solver_dir=solver_dir)
    # logger.info("Solver ran successfully!")