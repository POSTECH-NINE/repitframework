from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging
import json
from typing import Literal

from torch import cuda
import torch

@dataclass
class BaseConfig:

	# Directories:
	root_dir:Path = Path(__file__).parent.resolve()
	dataloader_dir:Path = Path(root_dir, "DataLoader")
	logs_dir:Path = Path(root_dir, "logs")
	metrics_dir:Path = Path(root_dir, "Metrics")
	model_selector_dir:Path = Path(root_dir, "Models")
	openfoam_dir:Path = Path(root_dir, "OpenFOAM")
	solver_dir:Path = Path(root_dir, "Solvers","natural_convection_case1")

	plots_dir:Path = Path(root_dir, "plots")
	assets_root_dir:Path = Path(root_dir, "Assets")
	assets_dir:Path = Path.joinpath(assets_root_dir, solver_dir.name)
	model_dump_dir:Path = Path(root_dir, f"ModelDump/{solver_dir.name}")

	# Logging Level
	logger_level = logging.DEBUG

	# Data parameters: Remember, if you calculating the residual, first vector should be U and last scalar should be T. 
	data_vars:dict[str] = field(default_factory=lambda: {"scalars":["T"],"vectors":["U"]})
	data_dim:int = 2 # It is to denote either data is 1D or 2D or 3D.
	grid_x: int = 200 # Number of grid points in x-direction
	grid_y: int = 200 # Number of grid points in y-direction
	grid_z: int = 1 # Number of grid points in z-direction
	grid_shape: tuple = tuple(item for item in (grid_z, grid_y, grid_x) if item > 1) # Shape of the grid based on data_dim
	grid_step: int|float = 1/grid_x # Assuming uniform grid spacing
	write_interval: int|float = 0.01 # Time step size
	round_to: int = len(str(write_interval).split(".")[-1]) # Number of decimal places to round to

	def extend_variables(self)->list[str]:
		"""
		Because we have a dictionary that separates variables into scalars and vectors,
		we need to combine them into a single list. We need to further differentiate vectors 
		based on the number of dimensions (1D or 2D or 3D).

		Returns
		-------
		list: 
			The list of variables: ["U_x", "U_y", "T"]
		"""
		vars_list = []
		for key, value in self.data_vars.items():
			if key == "vectors":
				for var in value:
					if self.data_dim == 1:
						vars_list.append(f"{var}_x")
					elif self.data_dim == 2:
						vars_list.extend([f"{var}_x", f"{var}_y"])
					elif self.data_dim == 3:
						vars_list.extend([f"{var}_x", f"{var}_y", f"{var}_z"])
			elif key == "scalars":
				vars_list.extend(value)
			else: 
				raise ValueError(f"Invalid key {key} in vars_dict. Must be either 'vectors' or 'scalars'.")
		return vars_list
	
	def get_variables(self, vars_dict:dict=None):
		'''
		There are cases where we don't exactly need dimension separated variables, 
		just the variables themselves just like how they are represented in OpenFOAM. 
		Hence, this method is to return the variables as they are, as a list.

		Args
		----
		vars_dict: dict:
			The dictionary containing the variables separated into vectors and scalars.
		
		Returns
		-------
		list: 
			The list of variables: ["U", "T"]
		'''
		vars_dict = self.data_vars if vars_dict is None else vars_dict
		return [var for _, value in vars_dict.items() for var in value]
	
	def setup_logger(self, name:str,log_file: Path) -> logging.Logger:
		"""
		Sets up and returns a logger instance.

		Args
		----
		name: str: 
			The name of the logger.
		log_file: Path: 
			The file path where logs will be saved.

		Returns
		-------
		logging.Logger: 
			Configured logger.
		"""
		level = self.logger_level
		logger = logging.getLogger(name)

		today_date = datetime.now().strftime("%Y-%m-%d")
		adding_solver_name = self.solver_dir.name + "/" + today_date
		today_date_dir = Path.joinpath(self.logs_dir, adding_solver_name)
		today_date_dir.mkdir(parents=True, exist_ok=True)
		log_file = Path(today_date_dir, log_file)
		
		# Prevent adding multiple handlers if the logger is already configured
		if not logger.handlers:
			logger.setLevel(level)
			formatter = logging.Formatter('%(asctime)s:%(pathname)s:%(levelname)s:%(message)s', datefmt='%H:%M:%S')

			file_handler = logging.FileHandler(log_file)
			file_handler.setFormatter(formatter)

			logger.addHandler(file_handler)

		return logger

	def __post_init__(self):
		log_file: Path = Path("BaseConfig.log")
		self.logger = self.setup_logger("BaseLogger",log_file)
		
		for dir in [
			self.dataloader_dir, 
			self.logs_dir, 
			self.metrics_dir, 
			self.model_selector_dir, 
			self.openfoam_dir, 
			self.solver_dir, 
			self.assets_root_dir, 
			self.assets_dir, 
			self.model_dump_dir,
			self.plots_dir,
		]:
			dir.mkdir(parents=True, exist_ok=True)

@dataclass
class OpenfoamConfig(BaseConfig):
	log_file = Path("OpenFOAM.log")
	mesh_type:str = "blockMesh"
	solver_type:str = "buoyantFoam"
	start_time = 10.0
	end_time = 10.02
	num_processors: int = 1 # Number of processors for parallel execution. If 1, it will run in serial.

	def __post_init__(self):
		self.case_name = self.solver_dir.name
		self.start_time = round(self.start_time, self.round_to)
		self.end_time = round(self.end_time, self.round_to)
		self.logger = self.setup_logger("OpenFOAMLogger",self.log_file)

@dataclass
class TrainingConfig(BaseConfig):
	batch_size: int = 10000
	epochs: int = 5000
	learning_rate: float = 1e-3
	residual_threshold: float = 5.0 # Adapted from the paper: Section 4.1; page 8
	device: str = "cuda" if cuda.is_available() else "cpu"
	loss = torch.nn.MSELoss()
	activation = torch.nn.ReLU()

	training_start_time = 10.0
	training_end_time = 10.03
	prediction_start_time = 10.03
	prediction_end_time = 110.0
	bc_type:str = "enforced" # either "enforced" or "ground_truth"

	# Dataset parameters:
	do_feature_selection: bool = True
	do_normalize: bool = True
	output_dims: Literal["BD", "BCD", "BCHW"] = "BD"

	# Model parameters: To understand them better, Go to: ./model_selector.py
	model_type: str = "fvmn" # or "fvfno2d"
	optimizer_type:str = "adam"
	scheduler_type:str = None # If None, no scheduler will be used.
	model_kwargs: dict = field(default_factory=lambda: {
		"hidden_layers": 3,
		"hidden_size": 398,
		"dropout": 0.2})
	layers_to_freeze: int = 1 # Number of layers to freeze in the model.

	log_file: Path = Path("Training.log")


	def log_metrics(self, key:str, value:int|float, metrics_type:str="prediction"):
		logging_path = Path(self.model_dump_dir) / f"{metrics_type}_metrics.ndjson"
		logging_path.parent.mkdir(parents=True, exist_ok=True)
		
		# build a single JSON record for this metric
		record = {"key": key, "value": value}
		
		# append one JSON object per line
		with open(logging_path, "a") as f:
			f.write(json.dumps(record) + "\n")
	
	def __post_init__(self):
		self.logger = self.setup_logger("TrainingLogger",self.log_file)
		variables = self.extend_variables()
		input_channels = len(variables)*((2*self.data_dim)+1) if self.do_feature_selection else len(variables)
		
		new_kwargs = {
			"vars": variables,
			"activation": self.activation,
			"input_channels": input_channels,
			"output_channels": 1, # If the network is Node assigned, the output channels should be set 1.
			"feature_selection": self.do_feature_selection
			}
		self.model_kwargs.update(new_kwargs)

		self.optim_kwargs = {
        "lr": self.learning_rate, 
        "betas": (0.9, 0.999), 
        "eps": 1e-8, 
        "weight_decay": 1e-5, 
        "amsgrad": False,
        "momentum": 0.0,
        "nesterov": False,
        "alpha": 0.99, 
        "centered": False,
        "dampening": 0.0,
        "initial_accumulator_value": 0.0

    }

@dataclass
class NaturalConvectionConfig(TrainingConfig):

	def _assign_temperature_profiles(self,):
		# TODO: for ease of use, it is hard-coded but think about making it dynamic.
		if  "natural_convection_case1" in self.solver_dir.name:
			if "Ampofo" in self.solver_dir.name:
				self.left_wall_temperature = 323.15
				self.right_wall_temperature = 283.15
			else:
				self.left_wall_temperature = 307.75
				self.right_wall_temperature = 288.15
		elif "natural_convection_case2" in self.solver_dir.name:
			self.left_wall_temperature = 317.75
			self.right_wall_temperature = 278.15
		elif "natural_convection_case3" in self.solver_dir.name:
			self.left_wall_temperature = 327.75
			self.right_wall_temperature = 268.15
		else:
			UserWarning(f"Unknown case name: {self.solver_dir.name}. Please check the case name.")
	
	def __post_init__(self):
		super().__post_init__()
		self._assign_temperature_profiles()
		self.logger = self.setup_logger("NaturalConvectionLogger", self.log_file)

if __name__ == "__main__":
	training_config = NaturalConvectionConfig()
	print(training_config.left_wall_temperature, training_config.right_wall_temperature)
	print(training_config.model_kwargs)