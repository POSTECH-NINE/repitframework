from pathlib import Path
import warnings
import imageio
from typing import Dict
import json
from collections import defaultdict
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from repitframework.config import BaseConfig, TrainingConfig

import seaborn as sns

# 1) Set Seaborn theme for “paper” context (smaller labels, ticks)
sns.set_theme(
	context="talk",                # Large, readable fonts for presentations/papers
	style="whitegrid",             # Clean grid background for clarity
	palette="colorblind",          # Vibrant, colorblind-friendly palette
	font="DejaVu Serif",           # Professional serif font (widely available)
	font_scale=1.4                 # Large font scaling for all elements
)

# Matplotlib rcParams for bold, clear, and consistent visuals
plt.rcParams.update({
	"axes.linewidth": 2,           # Bold axes lines
	"axes.edgecolor": "black",
	"axes.labelweight": "bold",
	"axes.titlesize": 20,          # Large, bold titles
	"axes.labelsize": 18,          # Large axis labels
	"xtick.labelsize": 16,
	"ytick.labelsize": 16,
	"xtick.direction": "in",       # Ticks inside for scientific style
	"ytick.direction": "in",
	"xtick.major.size": 7,
	"ytick.major.size": 7,
	"xtick.minor.size": 4,
	"ytick.minor.size": 4,
	"xtick.major.width": 2,
	"ytick.major.width": 2,
	"xtick.minor.width": 1,
	"ytick.minor.width": 1,
	"grid.linestyle": "--",        # Dashed grid for readability
	"grid.alpha": 0.5,
	"legend.frameon": True,        # Boxed legend for clarity
	"legend.framealpha": 0.95,
	"legend.fancybox": True,
	"legend.fontsize": 15,
	"lines.linewidth": 3,          # Bold lines for all plots
	"lines.markersize": 8,
	"figure.dpi": 300,             # High-res for screen and print
	"savefig.dpi": 300,
	"savefig.format": "png",       # Use "pdf" for vector output if needed
	"figure.facecolor": "white",
	"axes.facecolor": "white",
	"pdf.fonttype": 42,            # Editable fonts in PDF
	"ps.fonttype": 42
})

warning_string = '''\n
Data dimension mismatch: 
	Variable shape: {}
	Expected data dimension: {}
Please update the data_dim parameter in the config file if you want to visualize \
all the dimensions of the data.\n'''

AXIS_LIST = ["x","y","z"]

def flip_and_reshape(
		data:np.ndarray,
		nx:int,ny:int
	) -> np.ndarray:
	'''
	This function is used to flip the data and reshape it to the desired shape.
	Order="C" is used because OpenFOAM stores the data in row-major order.
	We are flipping the data because numpy writes the data in the first row first which is different from OpenFOAM.
	'''
	return data.reshape(ny,nx)

def process_variable(
		data_dict:Dict[str, np.ndarray], 
		var:str, 
		data_dim:int, 
		nx:int, 
		ny:int
	) -> Dict[str, np.ndarray]:
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

def load_metrics(metrics_path:Path ) -> dict[str, list[float]]:
	data = defaultdict(list)
	# Read each JSON object (one per line) and bucket by key
	with open(metrics_path, "r") as f:
		if str(metrics_path).endswith(".ndjson"):
			for line in f:
				record = json.loads(line)
				# record is like {"key": "loss", "value": 0.123}
				data[record["key"]].append(record["value"])
		else:
			data = json.load(f)
	return data

def visualize_output(
		base_config:BaseConfig,
		timestamp:int|float,
		np_data_dir:Path = None, 
		data_vars:list=None, 
		save_name:str="output",
		mode:str="image",
		is_ground_truth:bool=True
	):
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

	# Ensure ax is always a list-like object
	ax = np.atleast_1d(ax)

	for i,(key,value) in enumerate(data_dict.items()):
		value = ax[i].imshow(value, origin="lower", cmap="inferno")
		# fig.colorbar(value, ax=ax[i]) # TODO: uncomment these two lines.
		# ax[i].set_title(key)
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

def make_animation(
		base_config:BaseConfig, 
		timestamps:list[int|float],
		is_ground_truth:bool,
		np_data_dir:Path=None, 
		data_vars:list=None, 
		save_name:Path=None,
		plot_pred_gaps:bool=False,
		set_fps:int=50
	)->bool:
	'''
	This function is used to make an animation of the output of the simulation.

	Args
	----
	base_config: BaseConfig
		The base configuration object.
	timestamps: list
		The list of timestamps for which the output is to be visualized.
	is_ground_truth: bool
		1. If True, it will visualize the ground truth data.
		2. If False, it will visualize the predicted data.
	np_data_dir: Path
		The path to the directory where the output numpy files are stored.
	data_vars: list
		The list of variables to be visualized. Default is ["U", "T"]
	save_name: str
		The name of the visualization file. Default is "output".
	plot_pred_gaps: bool
		1. If True, it will plot the prediction gaps.
		2. If False, it will not plot the prediction gaps.
	set_fps: int
		The frames per second for the animation.
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
			images_list.append(
				visualize_output(
					base_config=base_config,
					timestamp=timestamp,
					np_data_dir=np_data_dir,
					data_vars=data_vars,
					mode="rgb_array",
					is_ground_truth=False
				)
			)

		else:
			images_list.append(
				visualize_output(
					base_config=base_config,
					timestamp=timestamp,
					np_data_dir=np_data_dir,
					data_vars=data_vars,
					mode="rgb_array",
					is_ground_truth=True
				)
			)
	imageio.mimsave(save_path, images_list, fps=set_fps, loop=0)
	return True

def get_probes_data(
		pred_time_list:list[int|float],
		full_time_list:list[int|float],
		ground_truth_dir:Path=None,
		prediction_dir:Path=None,
		plot_prediction_only:bool=False
	):
	'''
	This function is used to perform quantitative analysis on the output of the simulation.

	Args
	----
	pred_time_list: list
		The list of timestamps for which the prediction is made.
	full_time_list: list
		The list of all the timestamps.
	ground_truth_dir: Path
		The path to the directory where the ground truth data is stored.
	prediction_dir: Path
		The path to the directory where the predicted data is stored.
	plot_prediction_only: bool
		1. If True, it will plot the prediction only.
		2. If False, it will plot the prediction and ground truth together.
	
	Note
	----
	Combination of pred_time_list, full_time_list, and plot_prediction_only will allow you: 
	1. To plot the prediction only. [pred_time_list required, full_time_list not required, plot_prediction_only=True]
	2. To plot the prediction and ground truth together. [pred_time_list required, full_time_list required, plot_prediction_only=False]
	3. To plot the ground truth only. [pred_time_list not required, full_time_list required, plot_prediction_only=False]
	'''
	backup_dir = Path(ground_truth_dir)
	assert backup_dir.exists(), f"Backup directory {backup_dir} does not exist."
	assert prediction_dir.exists(), f"Prediction directory {prediction_dir} does not exist."


	probes_data = {"T": {"ground_truth": defaultdict(list), "predicted": defaultdict(list)},
				   "U_x": {"ground_truth": defaultdict(list), "predicted": defaultdict(list)},
				   "U_y": {"ground_truth": defaultdict(list), "predicted": defaultdict(list)}}

	# Defining coordinates for probes: 
	probes_labels = {"t1":(39699), "t2":(39499), "t3":(39299), "b1":(299), "b2":(499), "b3":(699)}

	if pred_time_list:
		min_time = min(pred_time_list)
		max_time = max(pred_time_list)
		interval_time = round(0.01, 2) # TODO: hardcoded the round to value. 
		timestamps = np.round(np.arange(min_time, max_time, interval_time), 2)
	else:
		timestamps = full_time_list

	if plot_prediction_only: timestamps = pred_time_list
	for timestamp in timestamps:
		t_data_ground_truth = np.load(backup_dir / f"T_{timestamp}.npy")
		U_data_ground_truth = np.load(backup_dir / f"U_{timestamp}.npy")

		ux_data_ground_truth = U_data_ground_truth[:,0]
		uy_data_ground_truth = U_data_ground_truth[:,1]

		if timestamp in pred_time_list:
			t_data_predicted = np.load(prediction_dir / f"T_{timestamp}_predicted.npy")
			U_data_predicted = np.load(prediction_dir / f"U_{timestamp}_predicted.npy")
		else:
			t_data_predicted = np.load(prediction_dir / f"T_{timestamp}.npy")
			U_data_predicted = np.load(prediction_dir / f"U_{timestamp}.npy")

		ux_data_predicted = U_data_predicted[:,0]
		uy_data_predicted = U_data_predicted[:,1]

		for probe_location in probes_labels.keys():
			probes_data["T"]["ground_truth"][probe_location].append(t_data_ground_truth[probes_labels[probe_location]].item())
			probes_data["T"]["predicted"][probe_location].append(t_data_predicted[probes_labels[probe_location]].item())

			probes_data["U_x"]["ground_truth"][probe_location].append(ux_data_ground_truth[probes_labels[probe_location]].item())
			probes_data["U_x"]["predicted"][probe_location].append(ux_data_predicted[probes_labels[probe_location]].item())

			probes_data["U_y"]["ground_truth"][probe_location].append(uy_data_ground_truth[probes_labels[probe_location]].item())
			probes_data["U_y"]["predicted"][probe_location].append(uy_data_predicted[probes_labels[probe_location]].item())
		
	return probes_data

def quantitative_analysis(
		pred_time_list:list[int|float],
		full_time_list:list[int|float],
		ground_truth_dir:Path=None,
		prediction_dir:Path=None,
		save_name:str= "velocity-x (m/s)",
		plot_prediction_only:bool=False
	):
	'''
	This function is used to perform quantitative analysis on the output of the simulation.

	Args
	----
	pred_time_list: list
		The list of timestamps for which the prediction is made.
	full_time_list: list
		The list of all the timestamps.
	ground_truth_dir: Path
		The path to the directory where the ground truth data is stored.
	prediction_dir: Path
		The path to the directory where the predicted data is stored.
	save_name: str
		The name of the variable to be saved as.
	plot_prediction_only: bool
		1. If True, it will plot the prediction only.
		2. If False, it will plot the prediction and ground truth together.
	
	Note
	----
	Combination of pred_time_list, full_time_list, and plot_prediction_only will allow you: 
	1. To plot the prediction only. [pred_time_list required, full_time_list not required, plot_prediction_only=True]
	2. To plot the prediction and ground truth together. [pred_time_list required, full_time_list required, plot_prediction_only=False]
	3. To plot the ground truth only. [pred_time_list not required, full_time_list required, plot_prediction_only=False]
	'''
	
	if pred_time_list:
		plot_MAE(
			pred_time_list=pred_time_list,
			ground_truth_dir=ground_truth_dir,
			prediction_dir=prediction_dir,
			var_name=save_name
		)
	else:
		prediction_dir = ground_truth_dir

	probes_data = get_probes_data(
		pred_time_list=pred_time_list,
		full_time_list=full_time_list,
		ground_truth_dir=ground_truth_dir,
		prediction_dir=prediction_dir,
		plot_prediction_only=plot_prediction_only
	)

	with open(prediction_dir / "probes_data.json", "w") as f:
		json.dump(probes_data, f, indent=4)
	
	fig, ax = plt.subplots(2, 1, figsize=(15,10))
	plt.rcParams['axes.titlesize'] = 22           # Title font size
	plt.rcParams['axes.labelsize'] = 20           # x and y label font size
	plt.rcParams['xtick.labelsize'] = 18          # x tick label size
	plt.rcParams['ytick.labelsize'] = 18          # y tick label size
	is_temp = False
	match save_name:
		case "velocity-x"|"U_x":
			ground_truth_data = probes_data["U_x"]["ground_truth"]
			predicted_data = probes_data["U_x"]["predicted"]
		case "velocity-y"|"U_y":
			ground_truth_data = probes_data["U_y"]["ground_truth"]
			predicted_data = probes_data["U_y"]["predicted"]
		case "temperature"|"T":
			ground_truth_data = probes_data["T"]["ground_truth"]
			predicted_data = probes_data["T"]["predicted"]
			is_temp = True
	ax[0].plot(ground_truth_data["t1"], label="T1", linestyle="-", color="red")
	ax[0].plot(ground_truth_data["t2"], label="T2", linestyle="-", color="green")
	ax[0].plot(ground_truth_data["t3"], label="T3", linestyle="-", color="blue")
	ax[0].plot(predicted_data["t1"], linestyle="--", color="red")
	ax[0].plot(predicted_data["t2"], linestyle="--", color="green")
	ax[0].plot(predicted_data["t3"], linestyle="--", color="blue")
	ax[0].legend()
	ax[0].grid()
	ax[0].set_title("Top Wall")
	ax[0].set_xlabel("Timesteps")
	ax[0].set_ylabel(save_name)
	# if is_temp: 
	# 	ax[0].set_ylim(292,304)
	# else: 
	# 	ax[0].set_ylim(-0.1,0.2)
	ax[0].margins(x=0)
	
	ax[1].plot(ground_truth_data["b1"], label="B1", linestyle="-", color="red")
	ax[1].plot(ground_truth_data["b2"], label="B2", linestyle="-", color="green")
	ax[1].plot(ground_truth_data["b3"], label="B3", linestyle="-", color="blue")
	ax[1].plot(predicted_data["b1"], linestyle="--", color="red")
	ax[1].plot(predicted_data["b2"], linestyle="--", color="green")
	ax[1].plot(predicted_data["b3"], linestyle="--", color="blue")
	ax[1].legend()
	ax[1].grid()
	ax[1].set_title("Bottom Wall")
	ax[1].set_xlabel("Timesteps")
	# if is_temp: 
	# 	ax[1].set_ylim(290.5,293.5)
	# else:
	# 	ax[1].set_ylim(-0.1,0.04)
	ax[1].set_ylabel(save_name)
	ax[1].margins(x=0)

	fig.tight_layout()
	#Save the plot
	plots_path = Path(str(prediction_dir).replace("Assets", "plots"))
	plots_path.mkdir(parents=True, exist_ok=True)
	fig.savefig(plots_path / f"{save_name}_analysis.png")

def plot_MAE(
		pred_time_list:list[int|float],
		ground_truth_dir:Path=None,
		prediction_dir:Path=None,
		var_name:str="velocity-x"
	):
	vars_dict = {
		"velocity-x": "U",
		"velocity-y": "U",
		"temperature": "T",
		"U_x": "U",
		"U_y": "U"}
	
	max_MAE_list = []
	ground_truth_values_at_MAE = []
	MeanAE_list = []
	for timestamp in pred_time_list:
		ground_truth = np.load(ground_truth_dir / f"{vars_dict[var_name]}_{timestamp}.npy")
		predicted_output = np.load(prediction_dir / f"{vars_dict[var_name]}_{timestamp}_predicted.npy")

		if var_name == "velocity-x" or var_name == "U_x":
			ground_truth = ground_truth[:,0]
			predicted_output = predicted_output[:,0]
		elif var_name == "velocity-y" or var_name == "U_y":
			ground_truth = ground_truth[:,1]
			predicted_output = predicted_output[:,1]
		elif var_name == "temperature" or var_name == "T":
			predicted_output = predicted_output.flatten()
		else:
			raise ValueError(f"Invalid variable name {var_name}. Must be either 'velocity-x', 'velocity-y', or 'temperature'.")
		absolute_error = np.abs(ground_truth - predicted_output)
		meanAE_error = np.mean(absolute_error)
		max_absolute_error = np.max(absolute_error)
		max_error_index = np.argmax(absolute_error)
		ground_truth_values_at_MAE.append(ground_truth[max_error_index])
		max_MAE_list.append(max_absolute_error)
		MeanAE_list.append(meanAE_error)
	
	x_min = min(pred_time_list)
	x_max = max(pred_time_list)
	y_min = min(max_MAE_list)
	y_max = max(max_MAE_list)

	max_MAE = max(max_MAE_list)
	max_MAE_time_step = pred_time_list[np.argmax(max_MAE_list)]
	gt_at_max_MAE = ground_truth_values_at_MAE[np.argmax(max_MAE_list)]
	# Plot the MAE values
	plt.figure(figsize=(8, 5))
	plt.plot(pred_time_list, max_MAE_list, label="MAE Over Time", color="blue", linewidth=2)

	# Highlight the max AE point
	plt.scatter(max_MAE_time_step, max_MAE, color="red", label=f"Max AE: {max_MAE:.3f} at t={max_MAE_time_step:.2f}")

	# Set axis limits
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)

	# Annotate the max AE point
	plt.annotate(f"Max AE: {max_MAE:.3f}\nTime: {max_MAE_time_step:.2f}\nrelErr: {max_MAE*100/gt_at_max_MAE:.3f}%",
				xy=(max_MAE_time_step, max_MAE), xycoords='data',
				xytext=(-50, 30), textcoords='offset points',
				arrowprops=dict(arrowstyle="->", color='red'), fontsize=10)

	# Labels and title
	plt.xlabel("Timestamps")
	plt.ylabel("Max Absolute Error (MAE)")
	plt.title(var_name)
	plt.legend()
	plt.grid(True)

	#Save the plot
	plots_path = Path(str(prediction_dir).replace("Assets", "plots"))
	plots_path.mkdir(parents=True, exist_ok=True)
	plt.savefig(plots_path / f"{var_name}_MAE.png", bbox_inches='tight')
	plt.close()
	
	# Plot the MSE values
	plt.figure(figsize=(8, 5))
	plt.plot(pred_time_list, MeanAE_list, label="MeanAE Over Time", color="blue", linewidth=2)
	# Highlight the max AE point
	max_MeanAE_time_step = pred_time_list[np.argmax(MeanAE_list)]
	plt.scatter(max_MeanAE_time_step, max(MeanAE_list), color="red", label=f"MeanAE: {max(MeanAE_list):.3f} at t={max_MeanAE_time_step:.2f}")
	
	# Labels and title
	plt.xlabel("Timestamps")
	plt.ylabel("Mean Absolute Error")
	plt.title(var_name)
	plt.legend()
	plt.grid(True)
	plt.savefig(plots_path / f"{var_name}_MeanAE.png", bbox_inches='tight'); plt.close()
	

def plot_residual_change(
		running_times:list, 
		relative_residual:list,
		residual_limit:float=5,
		save_name:str="relative_residual",
		save_path:str="/home/shilaj/repitframework/repitframework/plots/natural_convection"
	):

	true_residual = np.ones_like(relative_residual)

	plt.figure(figsize=(10,4))
	plt.plot(running_times, true_residual, ":k", label="Reference value")
	plt.plot(running_times, relative_residual, "-b", label="RePIT-Framework", linewidth=2)
	plt.ylim(0.1,100)
	# plt.xlim(10, 20)
	# plt.xticks(x_ticks)
	plt.xlabel("Timestamps")
	plt.yscale("log", base=10)
	plt.ylabel("Scaled residual")
	plt.legend()
	plt.title(f"Relative residual mass limit: {residual_limit}")
	plt.tight_layout()
	plt.savefig(f"{save_path}/{save_name}.png")

def plot_L2_error(
	pred_time_list: list[float],
	ground_truth_dir: Path,
	prediction_dir: Path,
	var_name: str = "temperature"
):
	"""
	Plots the relative L2 error over time for the given variable.
	"""
	vars_dict = {
		"velocity-x": "U",
		"temperature": "T",
		"U_x": "U"
	}
	l2_errors = []
	for timestamp in pred_time_list:
		gt = np.load(ground_truth_dir / f"{vars_dict[var_name]}_{timestamp}.npy")
		pred = np.load(prediction_dir / f"{vars_dict[var_name]}_{timestamp}_predicted.npy")
		if var_name in ["velocity-x", "U_x"]:
			gt = gt[:, 0]
			pred = pred[:, 0]
		elif var_name == "temperature":
			gt = gt.flatten()
			pred = pred.flatten()
		# Relative L2 error
		l2 = np.linalg.norm(gt - pred) / (np.linalg.norm(gt) + 1e-12)
		l2_errors.append(l2)
	# Plot
	plt.figure(figsize=(8, 5))
	plt.plot(pred_time_list, l2_errors, label="Relative L2 Error", color="purple", linewidth=2)
	plt.xlabel("Timestamps")
	plt.ylabel("Relative L2 Error")
	plt.title(f"L2 Error ({var_name})")
	plt.legend()
	plt.grid(True)
	plots_path = Path(str(prediction_dir).replace("Assets", "plots"))
	plots_path.mkdir(parents=True, exist_ok=True)
	plt.savefig(plots_path / f"{var_name}_L2_error.png", bbox_inches='tight')
	plt.close()

# def still_comparisons(
# 		prediction_dir:Path|str,
# 		ground_truth_dir:Path|str,
# 		time_list:list[float]=[20,60,110],
# 		temp_profiles:list[float]=[288.15, 307.75]):
# 	var_labels = ["Temperature", "Velocity"]
# 	data_dict = {
# 		"Ground truth": [],
# 		"Prediction": []
# 	}
# 	prediction_dir = Path(prediction_dir)
# 	ground_truth_dir = Path(ground_truth_dir)
# 	# Check if the directories exist
# 	if not prediction_dir.exists():
# 		raise FileNotFoundError(f"Prediction directory {prediction_dir} does not exist.")
# 	if not ground_truth_dir.exists():
# 		raise FileNotFoundError(f"Ground truth directory {ground_truth_dir} does not exist.")
# 	# Load the data
# 	for time in time_list:
# 		time = float(time)
# 		ground_truth_temp = np.load(ground_truth_dir / f"T_{time}.npy").reshape(200,200)
# 		prediction_path_T = prediction_dir / f"T_{time}_predicted.npy"
# 		prediction_path_U = prediction_dir / f"U_{time}_predicted.npy"
# 		if not prediction_path_T.exists():
# 			prediction_path_T = prediction_dir / f"T_{time}.npy"
# 			prediction_path_U = prediction_dir / f"U_{time}.npy"
# 		predicted_temp = np.load(prediction_path_T).reshape(200,200)
# 		ground_truth_vel = np.load(ground_truth_dir / f"U_{time}.npy")[:,0].reshape(200,200)
# 		predicted_vel = np.load(prediction_path_U)[:,0].reshape(200,200)

# 		data_dict["Ground truth"].append([ground_truth_temp, ground_truth_vel])
# 		data_dict["Prediction"].append([predicted_temp, predicted_vel])

# 	cols = len(data_dict)
# 	rows = len(time_list)*len(var_labels)
# 	time_labels = [f"{time} s" for time in time_list]
# 	fig, axs = plt.subplots(rows, cols ,figsize=(10, 10), constrained_layout=True)

# 	vmin_temp = temp_profiles[0]
# 	vmax_temp = temp_profiles[1]

# 	# vmin_vel = -0.2
# 	# vmax_vel = 0.3
# 	# Loop through rows and columns to fill in data
# 	for row in range(rows):  # 2 variables x 2 time steps
# 		for col, (title, data_list) in enumerate(data_dict.items()):
# 			if row in list(range(len(time_list))): img = axs[row, col].imshow(data_list[row//len(var_labels)][0], cmap="coolwarm", origin="lower") 
# 			else: img = axs[row, col].imshow(data_list[row//len(var_labels)][1], cmap="turbo",origin="lower")
# 			if title == "Ground truth": 
# 				cbar = plt.colorbar(img, ax=axs[row, col], fraction=0.046, pad=0.04)
# 			axs[row, col].set_xticks([])
# 			axs[row, col].set_yticks([])
# 			if row == 0:
# 				axs[row, col].set_title(title, fontsize=12)

# 	# Add row labels for Temperature/Velocity
# 	for row_idx, var_label in enumerate(var_labels):
# 		fig.text(0.02, 0.75 - row_idx * 0.5, var_label, va='center', ha='center', 
# 				fontsize=12, fontweight='bold', rotation=90, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

# 	# Add row labels for Time (12s, 20s) in both Temperature & Velocity sections
# 	for row_idx, time_label in enumerate(time_labels):
# 		fig.text(0.02, 0.88 - row_idx * 0.25, time_label, va='center', ha='center', 
# 				fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
# 		fig.text(0.02, 0.38 - row_idx * 0.25, time_label, va='center', ha='center', 
# 				fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
		
# 	#Save the plot
# 	plots_path = Path(str(prediction_dir).replace("Assets", "plots"))
# 	plots_path.mkdir(parents=True, exist_ok=True)
# 	plt.savefig(plots_path / "still_comparisons.png", bbox_inches='tight')

def still_comparisons(
		prediction_dir:Path|str,
		ground_truth_dir:Path|str,
		time_list:list[float]=[20,60,110],
		temp_profiles:list[float]=[288.15, 307.75]):

	var_labels = ["Temperature", "Velocity"]
	prediction_dir = Path(prediction_dir)
	ground_truth_dir = Path(ground_truth_dir)

	if not prediction_dir.exists():
		raise FileNotFoundError(f"Prediction directory {prediction_dir} does not exist.")
	if not ground_truth_dir.exists():
		raise FileNotFoundError(f"Ground truth directory {ground_truth_dir} does not exist.")

	data_dict = {"Ground truth": [], "Prediction": []}

	for time in time_list:
		time = float(time)
		gt_temp = np.load(ground_truth_dir / f"T_{time}.npy").reshape(200, 200)
		gt_vel = np.load(ground_truth_dir / f"U_{time}.npy")[:, 0].reshape(200, 200)

		pred_temp_path = prediction_dir / f"T_{time}_predicted.npy"
		pred_vel_path = prediction_dir / f"U_{time}_predicted.npy"

		if not pred_temp_path.exists():
			pred_temp_path = prediction_dir / f"T_{time}.npy"
			pred_vel_path = prediction_dir / f"U_{time}.npy"

		pred_temp = np.load(pred_temp_path).reshape(200, 200)
		pred_vel = np.load(pred_vel_path)[:, 0].reshape(200, 200)

		data_dict["Ground truth"].append((gt_temp, gt_vel))
		data_dict["Prediction"].append((pred_temp, pred_vel))

	n_rows = len(time_list)
	n_cols = 4  # 2 conditions (Ground truth/Prediction) * 2 variables (Temp/Vel)

	fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows), constrained_layout=True)

	for row, time in enumerate(time_list):
		for col, key in enumerate(["Ground truth", "Prediction"]):
			temp_img = axs[row, col*2].imshow(data_dict[key][row][0], cmap="turbo", origin="lower")
			vel_img = axs[row, col*2+1].imshow(data_dict[key][row][1], cmap="viridis", origin="lower")

			# Set titles
			if row == 0:
				axs[row, col*2].set_title(f"{key} - Temperature", fontsize=12)
				axs[row, col*2+1].set_title(f"{key} - Velocity", fontsize=12)

			# Colorbars
			fig.colorbar(temp_img, ax=axs[row, col*2], fraction=0.046, pad=0.04)
			fig.colorbar(vel_img, ax=axs[row, col*2+1], fraction=0.046, pad=0.04)

			# Remove ticks
			for idx in [col*2, col*2+1]:
				axs[row, idx].set_xticks([])
				axs[row, idx].set_yticks([])

			# Row labels (time)
			axs[row, 0].set_ylabel(f"{time} s", fontsize=12, rotation=90, labelpad=15)

	# Save the plot
	plots_path = Path(str(prediction_dir).replace("Assets", "plots"))
	plots_path.mkdir(parents=True, exist_ok=True)
	plt.savefig(plots_path / "still_comparisons.png", bbox_inches='tight')
	plt.close()

def plot_streamlines_comparison(
	data_path_1,
	data_path_2,
	ground_truth_path,
	t,
	save_path=None
):
	"""
	Plot streamlines for two prediction configurations side by side (or just one if data_path_2 is None).
	Args:
		data_path_1: str, directory containing U_{t}_predicted.npy for first prediction
		data_path_2: str or None, directory for second prediction (can be None)
		ground_truth_path: str, directory containing ground truth U_{t}.npy
		t: float or int, time step (e.g., 20.0)
		case1_label: str, label for first prediction
		case2_label: str, label for second prediction
	"""
	# --- Load ground truth ---
	U_true = np.load(os.path.join(ground_truth_path, f"U_{t}.npy"))
	Ux_true = U_true[:, 0].reshape(200, 200)
	Uy_true = U_true[:, 1].reshape(200, 200)

	# --- Helper to load prediction (with fallback) ---
	def load_pred_U(data_path, t):
		pred_file = os.path.join(data_path, f"U_{t}_predicted.npy")
		if not os.path.exists(pred_file):
			pred_file = os.path.join(data_path, f"U_{t}.npy")
		return np.load(pred_file)

	# --- Load prediction 1 ---
	U_pred1 = load_pred_U(data_path_1, t)
	Ux_pred1 = U_pred1[:, 0].reshape(200, 200)
	Uy_pred1 = U_pred1[:, 1].reshape(200, 200)

	# --- Load prediction 2 if provided ---
	has_second = data_path_2 is not None and data_path_2 != ""
	if has_second:
		U_pred2 = load_pred_U(data_path_2, t)
		Ux_pred2 = U_pred2[:, 0].reshape(200, 200)
		Uy_pred2 = U_pred2[:, 1].reshape(200, 200)

	# --- Prepare grid ---
	X = np.linspace(0, 199, 200, dtype=int)
	Y = np.linspace(0, 199, 200, dtype=int)
	X, Y = np.meshgrid(X, Y)

	# --- Colors ---
	gt_color = "#20B2AA"      # Teal for ground truth
	pred1_color = "#D81B60"   # Magenta for pred1
	pred2_color = "#2B34A7"   # Green for pred2

	# --- Plot ---
	if has_second:
		fig, axs = plt.subplots(1, 2, figsize=(16, 7))
		# Prediction 1
		axs[0].streamplot(X, Y, Ux_true, Uy_true, color=gt_color, linewidth=1.5, density=2, arrowsize=1.2)
		axs[0].streamplot(X, Y, Ux_pred1, Uy_pred1, color=pred1_color, linewidth=1.5, density=2, arrowsize=1.2)
		axs[0].set_title(f"Streamlines @ t={t}")
		axs[0].legend([
			plt.Line2D([0], [0], color=gt_color, lw=2, label="True"),
			plt.Line2D([0], [0], color=pred1_color, lw=2, label="Pred.")
		], ["True", "Pred."], loc="upper right")
		axs[0].set_aspect('equal')
		axs[0].set_xticks([]); axs[0].set_yticks([])

		# Prediction 2
		axs[1].streamplot(X, Y, Ux_true, Uy_true, color=gt_color, linewidth=1.5, density=2, arrowsize=1.2)
		axs[1].streamplot(X, Y, Ux_pred2, Uy_pred2, color=pred2_color, linewidth=1.5, density=2, arrowsize=1.2)
		axs[1].set_title(f"Streamlines @ t={t}")
		axs[1].legend([
			plt.Line2D([0], [0], color=gt_color, lw=2, label="True"),
			plt.Line2D([0], [0], color=pred2_color, lw=2, label="Pred.")
		], ["True", "Pred."], loc="upper right")
		axs[1].set_aspect('equal')
		axs[1].set_xticks([]); axs[1].set_yticks([])

		plt.tight_layout()
	else:
		plt.figure(figsize=(8, 7))
		plt.streamplot(X, Y, Ux_true, Uy_true, color=gt_color, linewidth=1.5, density=2, arrowsize=1.2)
		plt.streamplot(X, Y, Ux_pred1, Uy_pred1, color=pred1_color, linewidth=1.5, density=2, arrowsize=1.2)
		plt.title(f"Streamlines @ t={t}")
		plt.legend([
			plt.Line2D([0], [0], color=gt_color, lw=2, label="True"),
			plt.Line2D([0], [0], color=pred1_color, lw=2, label="Pred.")
		], ["True", "Pred."], loc="upper right")
		plt.gca().set_aspect('equal')
		plt.xticks([]); plt.yticks([])
		plt.tight_layout()
	if save_path:
		plt.savefig(f"{save_path}/streamlines_comparison_{t}.png", bbox_inches='tight', dpi=300)

def plot_spectral_analysis(
	prediction_dir: str,
	ground_truth_dir: str,
	timestep: float,
	save_path: str = None
):
	"""
	Plot spectral (energy) analysis for velocity and temperature fields.
	Args:
		Ux_true, Ux_pseudo, Uy_true, Uy_pseudo, T_true, T_pseudo: 2D arrays (shape: [ny, nx])
		save_path: if provided, saves the figure to this path
		title_prefix: optional string to prepend to plot titles
	"""
	U_true = np.load(f"{ground_truth_dir}/U_{timestep}.npy")
	Ux_true = U_true[:, 0].reshape(200, 200)
	Uy_true = U_true[:, 1].reshape(200, 200)
	T_true = np.load(f"{ground_truth_dir}/T_{timestep}.npy").reshape(200, 200)

	if os.path.exists(f"{prediction_dir}/U_{timestep}_predicted.npy"):
		U_pseudo = np.load(f"{prediction_dir}/U_{timestep}_predicted.npy")
		T_pseudo = np.load(f"{prediction_dir}/T_{timestep}_predicted.npy").reshape(200, 200)
	else:
		U_pseudo = np.load(f"{prediction_dir}/U_{timestep}.npy")
		T_pseudo = np.load(f"{prediction_dir}/T_{timestep}.npy").reshape(200, 200)
	Ux_pseudo = U_pseudo[:, 0].reshape(200, 200)
	Uy_pseudo = U_pseudo[:, 1].reshape(200, 200)
	# --- Ensure data is 2D arrays ---
	if Ux_true.ndim != 2 or Ux_pseudo.ndim != 2 or Uy_true.ndim != 2 or Uy_pseudo.ndim != 2 or T_true.ndim != 2 or T_pseudo.ndim != 2:
		raise ValueError("Input arrays must be 2D (shape: [200, 200])")

	# --- Compute FFTs ---
	fft_ux_true = np.fft.fft2(Ux_true)
	fft_ux_pseudo = np.fft.fft2(Ux_pseudo)
	fft_uy_true = np.fft.fft2(Uy_true)
	fft_uy_pseudo = np.fft.fft2(Uy_pseudo)
	fft_T_true = np.fft.fft2(T_true)
	fft_T_pseudo = np.fft.fft2(T_pseudo)

	# --- Energy ---
	E_vel_true = np.abs(fft_ux_true)**2 + np.abs(fft_uy_true)**2
	E_vel_pseudo = np.abs(fft_ux_pseudo)**2 + np.abs(fft_uy_pseudo)**2
	E_T_true = np.abs(fft_T_true)**2
	E_T_pseudo = np.abs(fft_T_pseudo)**2

	# --- Shift ---
	E_vel_true = np.fft.fftshift(E_vel_true)
	E_vel_pseudo = np.fft.fftshift(E_vel_pseudo)
	E_T_true = np.fft.fftshift(E_T_true)
	E_T_pseudo = np.fft.fftshift(E_T_pseudo)

	# --- Radial Spectrum ---
	def radial_spectrum(E: np.ndarray) -> np.ndarray:
		nx, ny = E.shape
		cx, cy = nx // 2, ny // 2
		y, x = np.indices((nx, ny))
		r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
		tbin = np.bincount(r.ravel(), E.ravel())
		nr = np.bincount(r.ravel())
		return tbin / (nr + 1e-10)

	E_vel_true_radial = radial_spectrum(E_vel_true)
	E_vel_pseudo_radial = radial_spectrum(E_vel_pseudo)
	E_T_true_radial = radial_spectrum(E_T_true)
	E_T_pseudo_radial = radial_spectrum(E_T_pseudo)

	k = np.arange(len(E_T_true_radial))

	# --- Plotting ---
	fig, axs = plt.subplots(2, 2, figsize=(14, 10))
	fig.suptitle(f"Spectral Energy Analysis: RePIT vs CFD", fontsize=18, fontweight='bold')

	# Velocity Spectrum (Difference)
	axs[0, 0].plot(k, E_vel_pseudo_radial - E_vel_true_radial, label='RePIT - CFD', color='#0072B2')
	axs[0, 0].set_title("Δ Velocity Energy Spectrum")
	axs[0, 0].set_xlabel("Wavenumber (k)")
	axs[0, 0].set_ylabel("Δ Energy")
	axs[0, 0].set_yscale("log")
	axs[0, 0].grid(True, alpha=0.4)
	axs[0, 0].legend()
	axs[0, 0].annotate("Positive: RePIT > CFD\nNegative: RePIT < CFD", xy=(0.7, 0.1), xycoords='axes fraction', fontsize=10, color='gray')

	# Temperature Spectrum (Difference)
	axs[0, 1].plot(k, E_T_pseudo_radial - E_T_true_radial, label='RePIT - CFD', color='#D55E00')
	axs[0, 1].set_title("Δ Temperature Energy Spectrum")
	axs[0, 1].set_xlabel("Wavenumber (k)")
	axs[0, 1].set_ylabel("Δ Energy")
	axs[0, 1].set_yscale("log")
	axs[0, 1].grid(True, alpha=0.4)
	axs[0, 1].legend()
	axs[0, 1].annotate("Positive: RePIT > CFD\nNegative: RePIT < CFD", xy=(0.7, 0.1), xycoords='axes fraction', fontsize=10, color='gray')

	# Velocity Spectrum (Ratio)
	axs[1, 0].plot(k, E_vel_pseudo_radial / (E_vel_true_radial + 1e-12), label='RePIT / CFD', color='#009E73')
	axs[1, 0].axhline(1, color='gray', linestyle='--', linewidth=1)
	axs[1, 0].set_title("Velocity Energy Ratio (RePIT/CFD)")
	axs[1, 0].set_xlabel("Wavenumber (k)")
	axs[1, 0].set_ylabel("Energy Ratio")
	axs[1, 0].set_yscale("log")
	axs[1, 0].grid(True, alpha=0.4)
	axs[1, 0].legend()

	# Temperature Spectrum (Ratio)
	axs[1, 1].plot(k, E_T_pseudo_radial / (E_T_true_radial + 1e-12), label='RePIT / CFD', color='#CC79A7')
	axs[1, 1].axhline(1, color='gray', linestyle='--', linewidth=1)
	axs[1, 1].set_title("Temperature Energy Ratio (RePIT/CFD)")
	axs[1, 1].set_xlabel("Wavenumber (k)")
	axs[1, 1].set_ylabel("Energy Ratio")
	axs[1, 1].set_yscale("log")
	axs[1, 1].grid(True, alpha=0.4)
	axs[1, 1].legend()

	plt.tight_layout()

	if save_path is not None:
		save_path = os.path.join(save_path, f"spectral_analysis_{timestep}.png")
		plt.savefig(save_path, bbox_inches='tight')

def save_loss(training_config:TrainingConfig,
			  save_initial_losses:bool=False,
			  merge_initial_losses:bool=False,
			  save_path:Path="./"):
	training_metrics_path = training_config.model_dir / "training_metrics.ndjson"
	metrics = load_metrics(training_metrics_path)
	
	plots_dir = str(training_config.model_dir).replace("ModelDump", "plots")
	plots_dir = Path(plots_dir) / "loss"
	plots_dir.mkdir(parents=True, exist_ok=True)
	train_loss:list = metrics["Training Loss"]
	val_loss:list = metrics["Validation Loss"]
	
	initial_loss_path = training_config.model_dir / "initial_losses.json"

	if save_initial_losses:
		initial_epochs = val_loss.index(min(val_loss)) + 1
		train_loss = train_loss[:initial_epochs]
		val_loss = val_loss[:initial_epochs]
		with open(initial_loss_path, "w") as f:
			json.dump({"Training Loss": train_loss, "Validation Loss": val_loss}, f, indent=4)

	if initial_loss_path.exists() and merge_initial_losses:
		with open(initial_loss_path, "r") as f:
			initial_losses = json.load(f)
		train_loss = initial_losses["Training Loss"] + train_loss
		val_loss = initial_losses["Validation Loss"] + val_loss
	
	# Annotate the max AE point
	epochs = np.linspace(1, len(train_loss), len(train_loss))
	# plt.annotate(f"Min Val Loss: {min(val_loss)}\nEpochs: {val_loss.index(min(val_loss))}",
	# 			xy=(epochs[1:], val_loss[1:]), xycoords='data',
	# 			xytext=(-50, 30), textcoords='offset points',
	# 			arrowprops=dict(arrowstyle="->", color='red'), fontsize=10)
	
	plt.figure(figsize=(10,5))
	plt.plot(epochs[1:], train_loss[1:], label="Training Loss")
	plt.plot(epochs[1:], val_loss[1:], label="Validation Loss")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.yscale("log")
	plt.title("Training and Validation Loss")
	plt.legend()
	plt.tight_layout()
	plt.savefig(plots_dir / f"training_loss_{datetime.now().strftime("%b-%d_%H:%M")}.png")
	plt.close()

def plot_everything(
		plot_start_time:float=10.0,
		plot_end_time:float=110.0,
		residual_limit:float=5.0
):
	base_config = BaseConfig()
	training_config = TrainingConfig()
	full_time_list = np.round(np.arange(plot_start_time, plot_end_time, 0.01),2)
	solver_dir = base_config.solver_dir
	model_dump_dir = str(solver_dir).replace("Solvers", "ModelDump")
	ground_truth_dir = str(solver_dir).replace("Solvers", "Assets")+ "_backup"
	prediction_dir = str(solver_dir).replace("Solvers", "Assets")
	plots_dir = str(solver_dir).replace("Solvers", "plots")
	# with open(model_dump_dir + "/prediction_metrics.json","r") as f:
	# 	metrics = json.load(f)
	metrics = load_metrics(model_dump_dir + "/prediction_metrics.ndjson")
	time_list = metrics["Running Time"] 

	# extended_time_list = [10.57,10.58,10.59,10.6,10.61,10.62]
	# time_list.extend(extended_time_list)
	# dest_dir = "/home/shilaj/repitframework/repitframework/Assets/natural_convection_case1"
	# source_dir = "/home/shilaj/repitframework/repitframework/Assets/natural_convection_case1_backup"
	# source_files = [source_dir + f"/{var}_{timestep}.npy" for var in ["U", "T"] for timestep in extended_time_list]
	# dest_files = [dest_dir + f"/{var}_{timestep}_predicted.npy" for var in ["U", "T"] for timestep in extended_time_list]
	# for source_file, dest_file in zip(source_files, dest_files):
	# 	os.system(f"cp {source_file} {dest_file}")
	
	# make_animation(base_config=base_config,
	# 				timestamps=full_time_list,
	# 				is_ground_truth=True,
	# 				set_fps=1000,
	# 				plot_pred_gaps=True,
	# 				save_name="ground_truth_simulation_case3",
	# 				np_data_dir="/home/shilaj/repitframework/repitframework/Assets/natural_convection_case3",)
	save_name_list = ["velocity-x", "velocity-y", "temperature"]
	quantitative_analysis(
		pred_time_list=time_list,
		full_time_list=full_time_list,
		ground_truth_dir=Path(ground_truth_dir),
		prediction_dir=Path(prediction_dir),
		save_name=save_name_list[0],
		plot_prediction_only=False
	)
	quantitative_analysis(
		pred_time_list=time_list,
		full_time_list=full_time_list,
		ground_truth_dir=Path(ground_truth_dir),
		prediction_dir=Path(prediction_dir),
		save_name=save_name_list[-1],
		plot_prediction_only=False
	)

	plot_L2_error(
		pred_time_list=time_list,
		ground_truth_dir=Path(ground_truth_dir),
		prediction_dir=Path(prediction_dir),
		var_name=save_name_list[0]
	)
	plot_L2_error(
		pred_time_list=time_list,
		ground_truth_dir=Path(ground_truth_dir),
		prediction_dir=Path(prediction_dir),
		var_name=save_name_list[-1]
	)
	still_comparisons(
		prediction_dir=prediction_dir,
		ground_truth_dir=ground_truth_dir,
		time_list=[int((plot_start_time+plot_end_time)/2), plot_end_time],
		temp_profiles=[training_config.left_wall_temperature, training_config.right_wall_temperature]
	)
	plot_residual_change(
		running_times=time_list,
		relative_residual=metrics["Relative Residual Mass"],
		residual_limit=residual_limit, ## <------------------------------------ change this
		save_name="relative_residual",
		save_path=plots_dir
	)

	plot_streamlines_comparison(
		data_path_1=prediction_dir,
		data_path_2=None,  # Set to None if you don't have a second prediction
		ground_truth_path=ground_truth_dir,
		t=plot_end_time,
		save_path=plots_dir
	)

	plot_spectral_analysis(
		prediction_dir=prediction_dir,
		ground_truth_dir=ground_truth_dir,
		timestep=plot_end_time,
		save_path=plots_dir
	)

def transfer_to_required_directory(dir_name:str, case:str, timesteps:int=10000):
	destination_dir = f"/home/shilaj/repitframework/repitframework/Assets/natural_convection_{case}_study/{timesteps}timesteps/{dir_name}"

	if not os.path.exists(destination_dir):
		os.makedirs(destination_dir)

	today_date = datetime.now().strftime("%Y-%m-%d")
	solver_name = Path(f"natural_convection_{case}")
	logs_path = Path("/home/shilaj/repitframework/repitframework/logs", solver_name, today_date, "Training.log")
	pred_metrics_path = Path("/home/shilaj/repitframework/repitframework/ModelDump", solver_name, "prediction_metrics.ndjson")
	training_metrics_path = Path("/home/shilaj/repitframework/repitframework/ModelDump", solver_name, "training_metrics.ndjson")
	plots_path = Path("/home/shilaj/repitframework/repitframework/plots", solver_name)
	assets_path = Path("/home/shilaj/repitframework/repitframework/Assets", solver_name)
	
	files_to_move = [
		logs_path,
		pred_metrics_path,
		training_metrics_path,
		assets_path]
	
	for file in os.scandir(plots_path):
		if file.name.endswith(".png") or file.name.endswith(".pdf"):
			files_to_move.append(Path(file.path))

	for file in files_to_move:
		if os.path.exists(file):
			destination_file = Path(destination_dir, file.name)
			os.system(f"mv {file} {destination_file}")
			print(f"Copied {file} to {destination_file}")
		else:
			print(f"File {file} does not exist, skipping.")
			
if __name__ == "__main__":

	training_config = TrainingConfig()

	plot_everything(plot_start_time=10.0, plot_end_time=20.0, residual_limit=training_config.residual_threshold)
	transfer_to_required_directory("2epochs5res", "case1", 1000)

