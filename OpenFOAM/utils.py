import Ofpp
from pathlib import Path
import os
import subprocess
import re

terminal_output = {}

def manage_assets(solver_dir:Path = None, assets_dir:Path = None):
    '''
    If we are trying out with different cases, this function is to put them nicely inside the assets directory with the name
    of the case as classifiers.  
    '''    
    case_name = solver_dir.name
    assets_dir = Path.joinpath(assets_dir, case_name)
    assets_dir.mkdir(parents=True, exist_ok=True)
    return assets_dir

def parse_to_numpy(solver_dir:Path = None, assets_dir:Path = None, variables:list = ["U", "p", "T"]):
    pass


# TODO: Don't forget to write test cases for every functions inside OpenFOAM module.
def run_the_solver(solver_dir:Path = None, assets_dir:Path = None, mesh_type:str = "blockMesh"):
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
    os.chdir(solver_dir)

    # Create the mesh
    try:
        command = [mesh_]
        mesh_result = subprocess.run(command, capture_output=True, text=True)
        terminal_output["mesh_output"] = mesh_result.stdout
        terminal_output["mesh_error"] = mesh_result.stderr
    except subprocess.CalledProcessError as e:
        print(f"Error in creating the mesh: {e}")

    # Run the solver
    try:
        command = [solver_]
        solver_result = subprocess.run(command, capture_output=True, text=True)
        terminal_output["solver_output"] = solver_result.stdout
        terminal_output["solver_error"] = solver_result.stderr
    except subprocess.CalledProcessError as e:
        print(f"Error in running the solver: {e}")
    # Save the data in the assets directory:
    return terminal_output

def read_mesh_type(solver_dir:Path = None, mesh_type:str = None):
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

def read_solver_type(solver_dir:Path = None):
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
        print(f"Error in reading the solver type: {e}")

    # if solver_type is None:
    #     OpenfoamConfig.solver_type = input("Please enter the solver type: e.g. simpleFoam, pisoFoam, etc.\n")
    #     return OpenfoamConfig.solver_type
    
    return solver_type

def update_time_steps():
    pass

if __name__ == "__main__":
    solver_dir = Path("/home/ninelab/repitframework/Solvers/natural_convection")
    assets_dir = Path("/home/ninelab/repitframework/Assets")

    command_to_list_time_directories = ["foamListTimes", "-case", solver_dir]
    result = subprocess.run(command_to_list_time_directories, capture_output=True, text=True).stdout.split("\n")
    result = [Path(solver_dir,i) for i in result if i]
    test_path = Path(result[0], "U")
    data = Ofpp.parse_internal_field(test_path)
    print(data.shape)