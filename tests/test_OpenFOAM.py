import unittest
import OpenFOAM
import config as cfg
from pathlib import Path
import subprocess
import re

class TestOpenFOAM(unittest.TestCase):
    def setUp(self):
        self.solver_dir = cfg.OpenfoamConfig().solver_dir
        self.assets_dir = cfg.OpenfoamConfig().assets_dir

        assert Path.exists(self.solver_dir)
        assert Path.exists(self.assets_dir)

        self.assertIsInstance(self.solver_dir, Path)
        self.assertIsInstance(self.assets_dir, Path)
        
    def test_manage_assets(self):
        asset_individual_dir = OpenFOAM.manage_assets(solver_dir=self.solver_dir,\
                                                assets_dir=self.assets_dir)
        assert Path.exists(asset_individual_dir)
        self.assertIsInstance(asset_individual_dir, Path)
        
    def test_parse_to_numpy(self):

        '''
        Checking if the OpenFOAM command: foamListTimes -case solver_dir works or not.
        '''
        command = ["foamListTimes", "-case", self.solver_dir]
        command_run = subprocess.run(command, 
                                   capture_output=True, 
                                   text=True)
        time_list = command_run.stdout.split("\n")
        
        self.assertEqual(command_run.returncode,
                        0,
                        f"Command: {command} failed to execute!")
        self.assertNotEqual(time_list,
                            [""], 
                            f"No time directories found in the {self.solver_dir} directory")
        self.assertIsNotNone(cfg.OpenfoamConfig().data_vars)

    def test_numpy_to_OpenFOAM(self):
        pass

    def test_run_the_solver(self):
        pass

    def test_read_mesh_type(self):
        mesh_ = cfg.OpenfoamConfig().mesh_type
        mesh_type = OpenFOAM.read_mesh_type(self.solver_dir, mesh_)

        self.assertIsNotNone(mesh_type)
        self.assertIsInstance(mesh_type, str)

    def test_read_solver_type(self):
        control_dict_path = Path.joinpath(self.solver_dir, "system", "controlDict")

        assert Path.exists(control_dict_path), "controlDict file not found in the directory"

        command = ["foamDictionary", control_dict_path, "-entry", "application", "-value"]
        command_result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(command_result.returncode, 0, "foamDictionary command failed to execute!")

        solver_type = command_result.stdout
        self.assertIsNotNone(solver_type, "Solver type not found in the controlDict file: CHECK REGEX PATTERN")

    def test_update_time_foamDictionary(self):
        controlDict_path = Path.joinpath(self.solver_dir, "system", "controlDict")
        assert Path.exists(controlDict_path), "controlDict file not found in the directory"

        start_time_command = ["foamDictionary", controlDict_path, "-entry", "startTime", "-value"]
        end_time_command = ["foamDictionary", controlDict_path, "-entry", "endTime", "-value"]
        present_time = float(subprocess.run(start_time_command, capture_output=True, text=True).stdout)
        end_time = float(subprocess.run(end_time_command, capture_output=True, text=True).stdout)

        command = ["foamDictionary", controlDict_path, "-set", f"startTime={present_time},endTime={end_time}"]
        command_result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(command_result.returncode, 0, "Error in updating the time: CHECK COMMAND")


if __name__ == "__main__":
    unittest.main()