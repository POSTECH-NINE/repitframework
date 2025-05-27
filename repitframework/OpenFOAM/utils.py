from pathlib import Path
import subprocess
from datetime import datetime
from typing import List
import timeit

import Ofpp
import numpy as np
from tqdm import tqdm

from repitframework.config import OpenfoamConfig

class OpenfoamUtils:
    def __init__(self, openfoam_config:OpenfoamConfig, 
                 solver_dir:Path=None, assets_dir:Path=None):

        self.openfoam_config = openfoam_config
        self.solver_dir = solver_dir if solver_dir else self.openfoam_config.solver_dir
        self.assets_dir = assets_dir if assets_dir else self.openfoam_config.assets_dir
        self.assets_path = self._get_assets_path()
        self.mesh_type = self._get_mesh_type()
        self.solver_type = self._read_solver_type()
    
    def _get_assets_path(self) -> Path:
        '''
        If we are trying out with different cases, this method is to put them nicely inside 
        the assets directory with the name of the case as classifiers.  
        '''  
        case_name = self.solver_dir.name
        assets_path = Path.joinpath(self.assets_dir, case_name)
        assets_path.mkdir(parents=True, exist_ok=True)
        return assets_path
    
    def _get_mesh_type(self) -> str:
        '''
        We are getting the mesh type analyzing the solvers directory. For example: 
        if we have "blockMeshDict" file in the "system" directory inside the solver case directory,
        then we can say that the mesh type is "blockMesh". 

        For now we could handle only two types of mesh: blockMesh and snappyHexMesh.

        Also, we can manually set the mesh type if we know it already. 
        '''
        if self.openfoam_config.mesh_type:
            return self.openfoam_config.mesh_type
        
        system_dir = Path.joinpath(self.solver_dir, "system")
        if Path.joinpath(system_dir, "blockMeshDict").exists():
            mesh_type = "blockMesh"
        elif Path.joinpath(system_dir, "snappyHexMeshDict").exists():
            mesh_type = "snappyHexMesh"
        else:
            raise ValueError("The mesh type is not recognized. Please set the mesh type manually.")
        return mesh_type
    
    def _read_solver_type(self):
        '''
        Ensures the solver type used to solve the problem. foamDictionary command comes in very handy
        to get the solver type.

        - Example: foamDictionary system/controlDict -entry application gives the application information 
        of the case. Applying regular expression to the received output, we can get the solver type.
        - solver_type information is also saved in the OpenfoamConfig class for future reference.

        OpenFOAM command used
        ---------------------
        1. Get application type, like buoyantFoam, simpleFoam, etc::

            foamDictionary -case $(CASE_DIRECTORY) -entry application -value system/controlDict
        '''
        if self.openfoam_config.solver_type:
            return self.openfoam_config.solver_type

        try:
            command = ["foamDictionary",
                       "-case",
                       self.solver_dir, 
                       "-entry", 
                       "application", 
                       "-value", 
                       "system/controlDict"]
            command_result = subprocess.run(command, capture_output=True, text=True)
            solver_type = command_result.stdout
        except subprocess.CalledProcessError as e:
            self.openfoam_config.logger.exception(f"Error in reading the solver type: {e}")
        
        return solver_type.strip()
    
    def _run_mesh_utility(self):
        self.openfoam_config.logger.debug("Creating the mesh!")
        command_to_create_mesh = [self.mesh_type, "-case", self.solver_dir]
        polyMesh_dir = Path.joinpath(self.solver_dir, "constant/polyMesh")
        if polyMesh_dir.exists():
            return "Mesh already exists!"
        return self.run_subprocess(command_to_create_mesh)

    @staticmethod
    def generate_intervals(
                        start_time:int|float, 
                        end_time:int|float, 
                        time_step:int|float, 
                        round_to:int=2
                    ) -> list:
        '''
        np.arange gave inconsistent results. So, we are using this method to generate the time intervals.
        '''
        time_list = []
        running_time = start_time
        while running_time <= end_time:
            time_list.append(round(running_time, round_to))
            running_time = round(running_time+time_step, round_to)
        return time_list

    @staticmethod
    def run_subprocess(command_list:list):
        '''
        Run the subprocess command and return the output.
        '''
        command_result = subprocess.run(command_list, capture_output=True, text=True, check=True)
        return command_result.stdout

    @staticmethod
    def parse_to_numpy(
                    openfoam_config:OpenfoamConfig,
                    start_time:float,
                    end_time:float,
                    solver_dir:Path=None,
                    save_path:Path=None,
                    variables:list=None,
                    write_interval:float=0.01,
                    del_dirs:bool=False
                ) -> Path:
        '''
        OpenFOAM stores the data in the form of Dictionary(OpenFOAM type) files. But to train the model
        it will be easier to change to tensors if we can convert them to numpy arrays. This method
        does the same. To carry out this task, we can use the Ofpp library.

        Args
        ----
        openfoam_config: OpenfoamConfig: 
            The OpenFOAM configuration object.
        solver_dir: str: 
            The path to the solver directory where OpenFOAM has stored the data after 
            running the solver.
        assets_dir: str: 
            The path to the assets directory where we want to save the data in the numpy
            format. 
        variables: list: 
            The list of variables that we want to convert to numpy. If not provided, 
            it will be get from the OpenFOAM configuration.
        time_list: list: 
            The list of time directories that we want to parse to numpy. If not provided, 
            it will list the time directories.
        del_dirs: bool: 
            If True, it will delete the time directories after parsing the data to numpy.

        OpenFOAM command used
        --------------------- 
        1. List the time directories::

            foamListTimes -case solver_dir

        2. Deleting the time directories::

            foamListTimes -case solver_dir -rm -time "1,2,3,4,5"

        Returns
        -------
        assets_path: Path: 
            The path to the assets directory where the data is saved in numpy format.
        '''
        solver_dir = solver_dir if solver_dir else openfoam_config.solver_dir
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
        else:
            save_path = Path(str(solver_dir).replace("Solvers", "Assets"))
            save_path.mkdir(parents=True, exist_ok=True)
        
        variables = variables if variables else openfoam_config.extend_variables()
        write_interval = write_interval if write_interval else openfoam_config.write_interval
        round_to = openfoam_config.round_to
        time_list: List[int|float] = OpenfoamUtils.generate_intervals(start_time, end_time, write_interval, round_to)

        time_list = [int(time) if time.is_integer() else time for time in time_list]
        time_directories = [Path(solver_dir,str(i)) for i in time_list]

        # Parse the data to numpy
        for time_dir in time_directories:
            for var in variables:
                try:
                    data = Ofpp.parse_internal_field(Path.joinpath(time_dir, var))
                    openfoam_config.logger.debug(f"Data parsed to numpy:{var}_{float(time_dir.name)} --> {data.shape}")
                    np.save(Path(save_path, f"{var}_{float(time_dir.name)}.npy"), data) # We are saving it in float because we want to keep things consistent.
                except Exception as e:
                    openfoam_config.logger.exception(f"Check if {time_dir/var} exists else refer to this error: \n{e}")
                    return False

        if del_dirs:
            #We save the last time directory for future reference.
            command_to_delete_directories = ["foamListTimes", "-case", solver_dir,
                                             "-rm", "-time",",".join(time_list[:-1])]
            command_output = OpenfoamUtils.run_subprocess(command_to_delete_directories)
            openfoam_config.logger.debug(f"Time directories deleted successfully: {command_output}")
        
        return save_path
    
    @staticmethod
    def update_time_foamDictionary(openfoam_config:OpenfoamConfig,
                                   solver_dir:Path=None,
                                   start_time:float=None,
                                   end_time:float=None,
                                   write_interval:float=None) -> bool:
        '''
        This method updates the time in the controlDict file. This is useful when you want to 
        run the solver for a specific time. 

        Args
        ----
        openfoam_config: OpenfoamConfig: 
            The OpenFOAM configuration object.
        solver_dir: str:
            The path to the solver directory.
        start_time: int|float: 
            The time step from which we want to start the simulation.
        end_time: int|float: 
            The end time of the simulation.
        write_interval: int|float: 
            interval between two consecutive timestamps.

        OpenFOAM command used
        ---------------------
        - Update the time in the controlDict file::

            foamDictionary -case $(CASE_DIRECTORY) -set startTime=0,endTime=10,writeInterval=0.01 system/controlDict
        '''
        solver_dir = solver_dir if solver_dir else openfoam_config.solver_dir
        write_interval = write_interval if write_interval else openfoam_config.write_interval
        round_to = len(str(write_interval).split(".")[-1])
        start_time = round(start_time,round_to) if start_time else openfoam_config.start_time
        
        end_time = round(end_time,round_to) if end_time else openfoam_config.end_time

        commands_to_update_time = ["foamDictionary",
                                    "-case",
                                    solver_dir,
                                    "-set",
                                    f"startTime={start_time},endTime={end_time},writeInterval={write_interval}",
                                    "system/controlDict"]
        
        command_output = OpenfoamUtils.run_subprocess(commands_to_update_time)
        openfoam_config.logger.debug(f"Time updated successfully: {command_output}")

        return True
    
    @staticmethod
    def max_time_directory(solver_dir:Path, round_to:int=2) -> float:
        '''
        Get the maximum time directory from the solver directory. This is useful when we want to 
        know the maximum time directory in the solver directory. 

        Args
        ----
        solver_dir: Path: 
            The path to the solver directory.
        round_to: int:
            The number of decimal places to round to.
            (depends on the write_interval)

        Returns
        -------
        float: 
            The maximum time directory in the solver directory.
        '''
        command_to_list_time_directories = ["foamListTimes", "-case", solver_dir]
        command_result = OpenfoamUtils.run_subprocess(command_to_list_time_directories)
        time_list = command_result.split("\n")

        '''
        if i.isnumeric() or i.replace(".", "").isnumeric() is to check if the time directory is a number.
        Because, if the time directories in the solver directory are not continuous, then OpenFOAM gives 
        an warning. So, the subprocess will capture this warning also. Hence, to avoid this we are implementing
        this work-around. 
        '''
        time_list = [round(float(time),round_to) for time in time_list if time.isnumeric() or time.replace(".", "").isnumeric()]
        max_time = max(time_list) if time_list else float(0)
        return int(max_time) if max_time.is_integer() else max_time

    def run_solver(
        self, 
        start_time:int|float=None,
        end_time:int|float=None,
        write_interval:int|float=None,
        save_to_numpy:bool=True,
        del_dirs:bool=False
    ) -> bool:
        '''
        This method aims at running the CFD solver of your interest. 

        Args
        ----
        start_time: int|float:
            The time step from which we want to start the simulation.
        end_time: int|float:
            The end time of the simulation.
        write_interval: int|float:
            interval between two consecutive timestamps.
        save_to_numpy: bool:
            If True, it will save the data in the numpy format.
        del_dirs: bool:
            If True, it will delete the time directories after parsing the data to numpy.
            NOTE: But keeps the last time directory for future reference. 

        Returns
        -------
        bool: 
            Just in case. 

        Functionality
        -------------
        - Updates the time in the controlDict file.
        - Creates the mesh.
        - Runs the solver.
        - Saves the data in the Assets directory. 

        Example
        -------
        In OpenFOAM, what you normally do is, clone the existing solver similar to the case you want
        to solve, modify different parameters according to your requirements and then run the solver.
        To run the solver you need to do these things: 
        - Go to the solver directory
        - Create the mesh
        - Run the solver
        
        So, this function tries to lift off these steps from your shoulder.

        OpenFOAM commands used
        ----------------------
        - Create the mesh::

            blockMesh -case solver_dir
        - Run the solver::
            
            buoyantFoam -case solver_dir
        '''
        write_interval = write_interval if write_interval else self.openfoam_config.write_interval
        round_to = self.openfoam_config.round_to
        start_time = round(start_time,round_to) if start_time else self.openfoam_config.start_time
        end_time = round(end_time,round_to) if end_time else self.openfoam_config.end_time
        # max_time = OpenfoamUtils.max_time_directory(self.solver_dir, round_to=round_to)
        '''
        Because if we already have a time directory greater than the time we want to start
        the simulation, then OpenFOAM doesn't start the simulation. So, to capture this 
        we throw an error. 
        '''
        # if start_time < max_time:
        #     raise ValueError(f"Max timestamp is {max_time}, illogical to start simulation from {start_time}")
        
        # Update the time in the controlDict file
        self.update_time_foamDictionary(
            self.openfoam_config, start_time=start_time,
            end_time=end_time, write_interval=write_interval
        )
        self.openfoam_config.logger.debug(
            f"Time updated successfully: \
            start_time={start_time}|end_time={end_time}|write_interval={write_interval}"
        )
        self.openfoam_config.logger.debug(f"Solver directory: {self.solver_dir}")

        # Create the mesh
        mesh_result = self._run_mesh_utility()
        self.openfoam_config.logger.debug(f"\n Mesh Output: {mesh_result}\n")

        # Run the solver
        command_to_run_solver = [self.solver_type, "-case", self.solver_dir]
        with tqdm(total=1, desc="Running Solver", unit="step") as pbar:
            self.openfoam_config.logger.debug(f"Solver running from {start_time} to {end_time} started at {datetime.now()}")
            solver_result = self.run_subprocess(command_to_run_solver)
            self.openfoam_config.logger.debug(f"Solver running from {start_time} to {end_time} ended at {datetime.now()}")
            pbar.update(1)
        self.openfoam_config.logger.debug(f"\n Solver Output: {solver_result}\n")

        if save_to_numpy:
            self.parse_to_numpy(
                openfoam_config=self.openfoam_config, 
                start_time=start_time, 
                end_time=end_time,
                write_interval=write_interval,
                del_dirs=del_dirs
            )

        return True

if __name__ == "__main__":
    openfoam_config = OpenfoamConfig()
    openfoam_utils = OpenfoamUtils(openfoam_config)
    start_time = timeit.default_timer()
    openfoam_utils.run_solver(start_time=10.0, end_time=20.0, write_interval=0.01,save_to_numpy=False, del_dirs=False)
    # openfoam_utils.parse_to_numpy(
    #     openfoam_config, 
    #     start_time=100.0, 
    #     end_time=110.0, 
    #     solver_dir=f"/home/shilaj/repitframework/repitframework/Solvers/natural_convection_case4",
    #     save_path=f"/home/shilaj/repitframework/repitframework/Assets/natural_convection_case4_backup",
    # )
    # end_time = timeit.default_timer()
    # print(f"Time taken: {end_time-start_time} seconds")