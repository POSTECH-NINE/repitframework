from pathlib import Path
import warnings
import imageio
from typing import Dict
import json

import matplotlib.pyplot as plt
import numpy as np

from repitframework.config import BaseConfig

warning_string = '''\n
Data dimension mismatch: 
	Variable shape: {}
	Expected data dimension: {}
Please update the data_dim parameter in the config file if you want to visualize \
all the dimensions of the data.\n'''

AXIS_LIST = ["x","y","z"]

def flip_and_reshape(data:np.ndarray,nx:int,ny:int) -> np.ndarray:
	'''
	This function is used to flip the data and reshape it to the desired shape.
	Order="C" is used because OpenFOAM stores the data in row-major order.
	We are flipping the data because numpy writes the data in the first row first which is different from OpenFOAM.
	'''
	return np.flipud(data.reshape(ny,nx,order="F"))

def process_variable(data_dict:Dict[str, np.ndarray], 
					 var:str, data_dim:int, nx:int, ny:int) -> Dict[str, np.ndarray]:
	'''
	This function is used to process the variable data.

	Args
	----
	data_dict: Dict[str, np.ndarray]
		The dictionary containing the variable data.
		Example: {"T": np.ndarray[40000,], "U": np.ndarray[40000,2]}
	var: str
		The variable name. e.g. "T", "U"
	data_dim: int
		The data dimension. e.g. 1 or 2
	nx: int
		The number of grid points in the x-direction.
	ny: int
		The number of grid points in the y-direction.

	Returns
	-------
	data_dict: Dict[str, np.ndarray]
		The dictionary containing the processed variable data.
		Example: {"T": np.ndarray[200,200], "U_x": np.ndarray[200,200], "U_y": np.ndarray[200,200]}
	'''
	shape_of_variable = data_dict[var].shape
	last_dim = shape_of_variable[-1] if len(shape_of_variable) >= 2 else None

	match len(shape_of_variable):
		case 2:
			if last_dim != data_dim: 
				warnings.warn(warning_string.format(shape_of_variable, data_dim))

			if last_dim == 1:
				# Single-dimensional data
				data_dict[var] = flip_and_reshape(data_dict[var],nx,ny)
			else:
				# Multi-dimensional data
				for i in range(min(data_dim, len(AXIS_LIST))):
					data_dict[f"{var}_{AXIS_LIST[i]}"] = flip_and_reshape(data_dict[var][:,i],nx,ny)
				del data_dict[var]
		case 1:
			# Single-dimensional data
			data_dict[var] = flip_and_reshape(data_dict[var],nx,ny)
		
		case _:
			raise ValueError(f"Data dimension mismatch. Expected 1 or 2 but got {data_dict[var].shape}")
	return data_dict

def visualize_output(base_config:BaseConfig,
					timestamp:int|float,
					np_data_dir:Path = None, 
					data_vars:list=None, 
					save_name:str="output",
					mode:str="image",
					is_ground_truth:bool=True):
	'''
	This function is used to visualize the output of the simulation.

	Args
	----
	base_config: 
		Base configuration object, so that we can give less arguments to the function.
	timestamp: int or float
		The time at which the output is to be visualized.
	np_data_dir: Path
		The path to the directory where the output numpy files are stored. 
		1. files should be in the format: U_{timestamp}.npy, T_{timestamp}.npy, etc.
		2. It must be a pathlib.Path object.
		3. The last directory of the path should be case name for the framework to work properly. e.g. 
			/home/openfoam/repitframework/repitframework/Assets/natural_convection
	data_vars: list
		The list of variables to be visualized. Default is ["U", "T"]
	save_name: str
		The name of the visualization file. Default is "output".
	mode: str
		1. "rgb_array": It will return the RGB array of the output.
		2. "image": It will save the image of the output in the specified directory. 
	is_ground_truth: bool
		1. If True, it will load the ground truth data.
		2. If False, it will load the predicted data.

	Returns
	-------
	1. If mode is "rgb_array": it will return the list (timestamps) of RGB arrays -> list[rgb_arrays]
	2. If mode is "image": it will  just save the images in the specified directory -> list[image_files]

	Functionality
	-------------
	1. numpy files are loaded in C-order i.e row-major order because OpenFOAM stores the data in row-major order.
	2. np.flipuid is used to flip the rows from top to bottom because numpy writes first data in first row which is different from OpenFOAM.
	'''
	np_data_dir = Path(np_data_dir) if np_data_dir else base_config.assets_path
	data_vars = data_vars if data_vars else base_config.extend_variables()
	save_path = base_config.root_dir / "plots" / np_data_dir.name 
	save_path.mkdir(exist_ok=True, parents=True)
	data_dim = base_config.data_dim
	ny = base_config.grid_y
	nx = base_config.grid_x

	data_dict:Dict[str, np.ndarray] = {} 
	'''
	Remember the format of the data_dict is:
	{"T": np.ndarray, "U_x": np.ndarray, "U_y": np.ndarray}
	'''
	for var in data_vars:
		numpy_file_name = f"{var}_{timestamp}.npy" if is_ground_truth else f"{var}_{timestamp}_predicted.npy"
		data_dict[var] = np.load(np_data_dir / numpy_file_name)
		data_dict = process_variable(data_dict, var, data_dim, nx, ny)
		
	num_subplots = len(data_dict)
	fig, ax = plt.subplots(1, num_subplots, figsize=(5*num_subplots, 5))
	for i,(key,value) in enumerate(data_dict.items()):
		value = ax[i].imshow(value)
		fig.colorbar(value, ax=ax[i])
		ax[i].set_title(key)
	fig.tight_layout()
	fig.suptitle("At time={}s".format(timestamp))
	if mode == "image":
		plt.savefig(save_path / f"{save_name}_{timestamp}.png")
		plt.close()
		return True
	elif mode == "rgb_array": # Convert plot to image
		fig.canvas.draw()
		rgb_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
		plt.close()
		return rgb_array
	else:
		raise ValueError("Invalid mode. Must be either 'image' or 'rgb_array'.")

def make_animation(base_config:BaseConfig, 
					timestamps:list[int|float],
					is_ground_truth:bool,
					np_data_dir:Path=None, 
					data_vars:list=None, 
					save_name:Path=None,
					plot_pred_gaps:bool=False,
					set_fps:int=1)->bool:
	'''
	This function is used to make an animation of the output of the simulation.
	'''
	np_data_dir = Path(np_data_dir) if np_data_dir else base_config.assets_path
	save_dir = base_config.root_dir / "plots" / np_data_dir.name
	save_dir.mkdir(exist_ok=True, parents=True)
	save_path = save_dir / f"{save_name}.gif" if save_name else save_dir / "output.gif"

	images_list = []
	pred_time_list = [] if is_ground_truth else timestamps
	if plot_pred_gaps:
		min_time = min(timestamps)
		max_time = max(timestamps)
		interval_time = round(timestamps[1] - timestamps[0], base_config.round_to)
		timestamps = np.round(np.arange(min_time, max_time, interval_time), base_config.round_to)
		
	for timestamp in timestamps:
		if timestamp in pred_time_list:
			images_list.append(visualize_output(base_config=base_config,
												timestamp=timestamp,
												np_data_dir=np_data_dir,
												data_vars=data_vars,
												mode="rgb_array",
												is_ground_truth=False))
		else:
			images_list.append(visualize_output(base_config=base_config,
												timestamp=timestamp,
												np_data_dir=np_data_dir,
												data_vars=data_vars,
												mode="rgb_array",
												is_ground_truth=True))
	imageio.mimsave(save_path, images_list, fps=set_fps, loop=0)
	return True

if __name__ == "__main__":

	base_config = BaseConfig()
	full_time_list = np.round(np.arange(10.01, 20.0, 0.01),2)
	with open("/home/shilaj/repitframework/repitframework/ModelDump/natural_convection/prediction_metrics.json","r") as f:
		metrics = json.load(f)
	time_list = metrics["Running Time"]
	make_animation(base_config=base_config,
					timestamps=time_list,
					is_ground_truth=False,
					set_fps=50,
					plot_pred_gaps=False,
					save_name="prediction_simulation",
					np_data_dir="/home/shilaj/repitframework/repitframework/Assets/natural_convection",)
	# visualize_output(base_config=base_config,
	# 				timestamp=10.51,
	# 				is_ground_truth=True,
	# 				save_name="true",
	# 				np_data_dir="/home/shilaj/repitframework/repitframework/Assets/natural_convection_backup",)
