from pathlib import Path
import warnings
import json
from collections import defaultdict
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from typing import Dict, List
from datetime import datetime

from repitframework.config import TrainingConfig

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
	"grid.alpha": 0.0,
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

AXIS_LIST = ["x", "y", "z"]
warning_string = '''\nData dimension mismatch:\n    Variable shape: {}\n    Expected data dimension: {}\nPlease update the data_dim parameter in the config file if you want to visualize \
all the dimensions of the data.\n'''

def flip_and_reshape(data: np.ndarray, nx: int, ny: int) -> np.ndarray:
	return data.reshape(ny, nx)

def process_variable(data_dict: Dict[str, np.ndarray], var: str, data_dim: int, nx: int, ny: int) -> Dict[str, np.ndarray]:
	shape_of_variable = data_dict[var].shape
	last_dim = shape_of_variable[-1] if len(shape_of_variable) >= 2 else None
	if len(shape_of_variable) == 2:
		if last_dim != data_dim:
			warnings.warn(warning_string.format(shape_of_variable, data_dim))
		if last_dim == 1:
			data_dict[var] = flip_and_reshape(data_dict[var], nx, ny)
		else:
			for i in range(min(data_dim, len(AXIS_LIST))):
				data_dict[f"{var}_{AXIS_LIST[i]}"] = flip_and_reshape(data_dict[var][:, i], nx, ny)
			del data_dict[var]
	elif len(shape_of_variable) == 1:
		data_dict[var] = flip_and_reshape(data_dict[var], nx, ny)
	else:
		raise ValueError(f"Data dimension mismatch. Expected 1 or 2 but got {shape_of_variable}")
	return data_dict

def load_metrics(metrics_path: Path) -> Dict[str, List[float]]:
	data = defaultdict(list)
	with open(metrics_path, "r") as f:
		if str(metrics_path).endswith(".ndjson"):
			for line in f:
				record = json.loads(line)
				data[record["key"]].append(record["value"])
		else:
			data = json.load(f)
	return data

def binned_stats(x, y, bins):
	sort_idx = np.argsort(x)
	x_sorted, y_sorted = x[sort_idx], y[sort_idx]
	bins_edges = np.linspace(x_sorted.min(), x_sorted.max(), bins + 1)
	bin_centers = 0.5 * (bins_edges[:-1] + bins_edges[1:])
	inds = np.digitize(x_sorted, bins_edges) - 1
	y_means, y_stds, centers = [], [], []
	for i in range(bins):
		mask = inds == i
		if np.any(mask):
			y_bin = y_sorted[mask]
			y_means.append(np.mean(y_bin))
			y_stds.append(np.std(y_bin))
			centers.append(bin_centers[i])
	return np.array(centers), np.array(y_means), np.array(y_stds)

def visualize_output(
	training_config:TrainingConfig,
	timestamp: int | float,
	np_data_dir: Path,
	data_vars: List[str],
	save_name: str = "output",
	mode: str = "image",
	is_ground_truth: bool = True,
	exclude_vars: List[str] = None,
	save_path: Path = None
):
	exclude_vars = exclude_vars or ["U_y"]
	data_vars = data_vars or training_config.extend_variables()
	data_dim = training_config.data_dim
	ny, nx = training_config.grid_y, training_config.grid_x
	data_dict: Dict[str, np.ndarray] = {}
	for var in data_vars:
		numpy_file_name = f"{var}_{timestamp}.npy" if is_ground_truth else f"{var}_{timestamp}_predicted.npy"
		data_dict[var] = np.load(np_data_dir / numpy_file_name)
		data_dict = process_variable(data_dict, var, data_dim, nx, ny)
	for var in exclude_vars:
		data_dict.pop(var, None)
	num_subplots = len(data_dict)
	fig, ax = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 5))
	ax = np.atleast_1d(ax)
	for i, (key, value) in enumerate(data_dict.items()):
		im = ax[i].imshow(value, origin="lower", cmap="inferno")
	fig.tight_layout()
	fig.suptitle(f"At time={timestamp}s")
	if save_path is None:
		save_path = np_data_dir
	if mode == "image":
		plt.savefig(Path(save_path) / f"{save_name}_{timestamp}.png")
		plt.close()
		return True
	elif mode == "rgb_array":
		fig.canvas.draw()
		rgb_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
			fig.canvas.get_width_height()[::-1] + (4,))
		plt.close()
		return rgb_array
	else:
		plt.close()
		raise ValueError("Invalid mode. Must be either 'image' or 'rgb_array'.")

def extend_timesteps_to_full(pred_time_list: List[int | float],time_step: int | float = 0.01) -> List[int | float]:
	min_time = min(pred_time_list)
	max_time = max(pred_time_list)
	full_time_list = []
	running_time = round(min_time,2)
	while running_time <= max_time:
		full_time_list.append(running_time)
		running_time = round(running_time + time_step,2)
	return full_time_list

def plot_MAE(
	pred_time_list: List[int | float],
	ground_truth_dir: Path,
	prediction_dir: Path,
	var_name: str = "velocity-x",
	save_path: Path = None,
	include_sim_results: bool = True
):
	vars_dict = {
		"velocity-x": "U",
		"velocity-y": "U",
		"temperature": "T",
		"U_x": "U",
		"U_y": "U"
	}
	maxAE_list = []
	meanAE_list = []

	temp_dict ={
		"max_maxAE": 0.0,
		"max_meanAE": 0.0,
		"max_maxAE_timestep": 0.0,
		"max_maxAE_gt": 0.0,
		"max_meanAE_gt": 0.0,
		"max_meanAE_timestep": 0.0
	}

	if include_sim_results:
		pred_time_list = extend_timesteps_to_full(pred_time_list, time_step=0.01)
	
	for timestamp in pred_time_list:
		ground_truth = np.load(ground_truth_dir / f"{vars_dict[var_name]}_{timestamp}.npy")
		predicted_output_path = prediction_dir / f"{vars_dict[var_name]}_{timestamp}_predicted.npy"
		if not predicted_output_path.exists():
			predicted_output_path = prediction_dir / f"{vars_dict[var_name]}_{timestamp}.npy"
		predicted_output = np.load(predicted_output_path)
		if var_name in ("velocity-x", "U_x"):
			ground_truth = ground_truth[:, 0]
			predicted_output = predicted_output[:, 0]
		elif var_name in ("velocity-y", "U_y"):
			ground_truth = ground_truth[:, 1]
			predicted_output = predicted_output[:, 1]
		elif var_name in ("temperature", "T"):
			predicted_output = predicted_output.flatten()
		else:
			raise ValueError(f"Invalid variable name {var_name}.")
		absolute_error = np.abs(ground_truth - predicted_output)

		meanAE_error = np.mean(absolute_error)
		maxAE_error = np.max(absolute_error)

		maxAE_list.append(maxAE_error)
		meanAE_list.append(meanAE_error)

		maxAE_index = np.argmax(absolute_error)
		if maxAE_error > temp_dict["max_maxAE"]:
			temp_dict["max_maxAE"] = maxAE_error
			temp_dict["max_maxAE_gt"] = ground_truth[maxAE_index]
			temp_dict["max_maxAE_timestep"] = timestamp
		if meanAE_error > temp_dict["max_meanAE"]:
			temp_dict["max_meanAE"] = meanAE_error
			temp_dict["max_meanAE_gt"] = np.mean(ground_truth)
			temp_dict["max_meanAE_timestep"] = timestamp

		
	x_min, x_max = min(pred_time_list), max(pred_time_list)
	y_min_maxAE, y_max_maxAE = min(maxAE_list), max(maxAE_list)
	y_min_meanAE, y_max_meanAE = min(meanAE_list), max(meanAE_list)
	
	if save_path is None:
		save_path = prediction_dir

	with open(Path(prediction_dir) / f"errors.ndjson", "a") as f:
		f.write(json.dumps({
			"key": f"{var_name}_max_AE",
			"value": maxAE_list
		}) + "\n")
		f.write(json.dumps({"key": f"{var_name}_mean_AE","value": meanAE_list})+ "\n")

	plt.figure(figsize=(8, 5))
	plt.plot(pred_time_list, maxAE_list, label="MaxAE Over Time", color="blue", linewidth=2)
	plt.scatter(temp_dict["max_maxAE_timestep"], temp_dict["max_maxAE"], color="red", 
			 label=f"Max MaxAE: {temp_dict['max_maxAE']:.3f} at t={temp_dict['max_maxAE_timestep']:.2f}")
	plt.xlim(x_min, x_max)
	plt.ylim(y_min_maxAE, y_max_maxAE)
	plt.annotate(f'''relErr: {temp_dict["max_maxAE"]*100/temp_dict["max_maxAE_gt"]:.3f}%''',
				xy=(temp_dict["max_maxAE_timestep"], temp_dict["max_maxAE"]), xycoords='data',
				xytext=(-50, 30), textcoords='offset points',
				arrowprops=dict(arrowstyle="->", color='red'), fontsize=10)
	plt.xlabel("Timestamps")
	plt.ylabel("Max Absolute Error (MaxAE)")
	plt.title(var_name)
	plt.legend()
	plt.grid(True)
	plt.savefig(Path(save_path) / f"{var_name}_MaxAE.png", bbox_inches='tight')
	plt.close()


	plt.figure(figsize=(8, 5))
	plt.plot(pred_time_list, meanAE_list, label="MeanAE Over Time", color="blue", linewidth=2)
	plt.scatter(temp_dict["max_meanAE_timestep"], temp_dict["max_meanAE"], color="red", label=f"MeanAE: {temp_dict['max_meanAE']:.3f} at t={temp_dict["max_meanAE_timestep"]:.2f}")
	plt.xlim(x_min, x_max)
	plt.ylim(y_min_meanAE, y_max_meanAE)
	plt.annotate(f'''relErr: {temp_dict["max_meanAE"]*100/temp_dict["max_meanAE_gt"]:.3f}%''',
					xy=(temp_dict["max_meanAE_timestep"], temp_dict["max_meanAE"]), xycoords='data',
					xytext=(-50, 30), textcoords='offset points',
					arrowprops=dict(arrowstyle="->", color='red'), fontsize=10)
	plt.xlabel("Timestamps")
	plt.ylabel("Mean Absolute Error")
	plt.title(var_name)
	plt.legend()
	plt.grid(True)
	plt.savefig(Path(save_path) / f"{var_name}_MeanAE.png", bbox_inches='tight')
	plt.close()

def plot_L2_error(
	pred_time_list: List[float],
	ground_truth_dir: Path,
	prediction_dir: Path,
	var_name: str = "temperature",
	save_path: Path = None,
	include_sim_results: bool = True
):
	vars_dict = {
		"velocity-x": "U",
		"temperature": "T",
		"U_x": "U",
		"T": "T"
	}
	l2_errors = []

	if include_sim_results:
		pred_time_list = extend_timesteps_to_full(pred_time_list, time_step=0.01)
	for timestamp in pred_time_list:
		gt = np.load(ground_truth_dir / f"{vars_dict[var_name]}_{timestamp}.npy")
		predicted_output_path = prediction_dir / f"{vars_dict[var_name]}_{timestamp}_predicted.npy"
		if not predicted_output_path.exists():
			predicted_output_path = prediction_dir / f"{vars_dict[var_name]}_{timestamp}.npy"
		pred = np.load(predicted_output_path)
		if var_name in ["velocity-x", "U_x"]:
			gt = gt[:, 0]
			pred = pred[:, 0]
		elif var_name == "temperature":
			gt = gt.flatten()
			pred = pred.flatten()
		l2 = np.linalg.norm(gt - pred) / (np.linalg.norm(gt) + 1e-12)
		l2_errors.append(l2)
	if save_path is None:
		save_path = prediction_dir

	with open(Path(prediction_dir) / f"errors.ndjson", "a") as f:
		f.write(json.dumps({
			"key": f"{var_name}_l2",
			"value": l2_errors
		}) + "\n")
	plt.figure(figsize=(8, 5))
	plt.plot(pred_time_list, l2_errors, label="Relative L2 Error", color="purple", linewidth=2)
	max_l2error_timestep = pred_time_list[np.argmax(l2_errors)]
	plt.scatter(max_l2error_timestep, max(l2_errors), color="red", label=f"MaxL2: {max(l2_errors):.3f} at t={max_l2error_timestep:.2f}")
	plt.xlabel("Timestamps")
	plt.ylabel("Relative L2 Error")
	plt.title(f"L2 Error ({var_name})")
	plt.legend()
	plt.grid(True)
	plt.savefig(Path(save_path) / f"{var_name}_L2_error.png", bbox_inches='tight')
	plt.close()

def still_comparisons(
	prediction_dir: Path | str,
	ground_truth_dir: Path | str,
	time_list: list[float] = [20, 60, 110],
	temp_profiles: list[float] = [288.15, 307.75],
	save_path: Path = None
):
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
	n_cols = 4
	fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), constrained_layout=True)
	for row, time in enumerate(time_list):
		for col, key in enumerate(["Ground truth", "Prediction"]):
			temp_img = axs[row, col * 2].imshow(data_dict[key][row][0], cmap="turbo", origin="lower")
			vel_img = axs[row, col * 2 + 1].imshow(data_dict[key][row][1], cmap="viridis", origin="lower")
			if row == 0:
				axs[row, col * 2].set_title(f"{key} - Temperature", fontsize=12)
				axs[row, col * 2 + 1].set_title(f"{key} - Velocity", fontsize=12)
			fig.colorbar(temp_img, ax=axs[row, col * 2], fraction=0.046, pad=0.04)
			fig.colorbar(vel_img, ax=axs[row, col * 2 + 1], fraction=0.046, pad=0.04)
			for idx in [col * 2, col * 2 + 1]:
				axs[row, idx].set_xticks([])
				axs[row, idx].set_yticks([])
			axs[row, 0].set_ylabel(f"{time} s", fontsize=12, rotation=90, labelpad=15)
	if save_path is None:
		save_path = prediction_dir
	plt.savefig(Path(save_path) / "still_comparisons.png", bbox_inches='tight')
	plt.close()


def save_loss(training_config:TrainingConfig,
			  save_initial_losses:bool=False,
			  merge_initial_losses:bool=False):
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

def plot_streamlines_comparison(
	data_path_1,
	data_path_2,
	ground_truth_path,
	t,
	save_path=None
):
	U_true = np.load(os.path.join(ground_truth_path, f"U_{t}.npy"))
	Ux_true = U_true[:, 0].reshape(200, 200)
	Uy_true = U_true[:, 1].reshape(200, 200)
	def load_pred_U(data_path, t):
		pred_file = os.path.join(data_path, f"U_{t}_predicted.npy")
		if not os.path.exists(pred_file):
			pred_file = os.path.join(data_path, f"U_{t}.npy")
		return np.load(pred_file)
	U_pred1 = load_pred_U(data_path_1, t)
	Ux_pred1 = U_pred1[:, 0].reshape(200, 200)
	Uy_pred1 = U_pred1[:, 1].reshape(200, 200)
	has_second = data_path_2 is not None and data_path_2 != ""
	if has_second:
		U_pred2 = load_pred_U(data_path_2, t)
		Ux_pred2 = U_pred2[:, 0].reshape(200, 200)
		Uy_pred2 = U_pred2[:, 1].reshape(200, 200)
	X = np.linspace(0, 199, 200, dtype=int)
	Y = np.linspace(0, 199, 200, dtype=int)
	X, Y = np.meshgrid(X, Y)
	gt_color = "#20B2AA"
	pred1_color = "#D81B60"
	pred2_color = "#2B34A7"
	if has_second:
		fig, axs = plt.subplots(1, 2, figsize=(16, 7))
		axs[0].streamplot(X, Y, Ux_true, Uy_true, color=gt_color, linewidth=1.5, density=2, arrowsize=1.2)
		axs[0].streamplot(X, Y, Ux_pred1, Uy_pred1, color=pred1_color, linewidth=1.5, density=2, arrowsize=1.2)
		axs[0].set_title(f"Streamlines @ t={t}")
		axs[0].legend([
			plt.Line2D([0], [0], color=gt_color, lw=2, label="True"),
			plt.Line2D([0], [0], color=pred1_color, lw=2, label="Pred.")
		], ["True", "Pred."], loc="upper right")
		axs[0].set_aspect('equal')
		axs[0].set_xticks([]); axs[0].set_yticks([])
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
	if save_path is None:
		save_path = data_path_1
	plt.savefig(f"{save_path}/streamlines_comparison_{t}.png", bbox_inches='tight', dpi=300)
	plt.close()

def plot_spectral_analysis(
	prediction_dir: str,
	ground_truth_dir: str,
	timestep: float,
	save_path: str = None
):
	
	color_choices = ['#009E73', "#0071B2FF", '#CC79A7', "#D55C00CE", "#DB0101FF"]
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
	if Ux_true.ndim != 2 or Ux_pseudo.ndim != 2 or Uy_true.ndim != 2 or Uy_pseudo.ndim != 2 or T_true.ndim != 2 or T_pseudo.ndim != 2:
		raise ValueError("Input arrays must be 2D (shape: [200, 200])")
	fft_ux_true = np.fft.fft2(Ux_true)
	fft_ux_pseudo = np.fft.fft2(Ux_pseudo)
	fft_uy_true = np.fft.fft2(Uy_true)
	fft_uy_pseudo = np.fft.fft2(Uy_pseudo)
	fft_T_true = np.fft.fft2(T_true)
	fft_T_pseudo = np.fft.fft2(T_pseudo)

	E_vel_true = np.abs(fft_ux_true) ** 2 + np.abs(fft_uy_true) ** 2
	E_vel_pseudo = np.abs(fft_ux_pseudo) ** 2 + np.abs(fft_uy_pseudo) ** 2
	# E_vel_true = np.abs(fft_ux_true) **2
	# E_vel_pseudo = np.abs(fft_ux_pseudo) ** 2

	E_T_true = np.abs(fft_T_true) ** 2
	E_T_pseudo = np.abs(fft_T_pseudo) ** 2
	E_vel_true = np.fft.fftshift(E_vel_true)
	E_vel_pseudo = np.fft.fftshift(E_vel_pseudo)
	E_T_true = np.fft.fftshift(E_T_true)
	E_T_pseudo = np.fft.fftshift(E_T_pseudo)
	def radial_spectrum(E: np.ndarray) -> np.ndarray:
		nx, ny = E.shape
		cx, cy = nx // 2, ny // 2
		y, x = np.indices((nx, ny))
		r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
		tbin = np.bincount(r.ravel(), E.ravel())
		nr = np.bincount(r.ravel())
		return tbin / (nr + 1e-10)
	E_vel_true_radial = radial_spectrum(E_vel_true)
	E_vel_pseudo_radial = radial_spectrum(E_vel_pseudo)
	E_T_true_radial = radial_spectrum(E_T_true)
	E_T_pseudo_radial = radial_spectrum(E_T_pseudo)

	k = np.arange(len(E_T_true_radial))
	fig, axs = plt.subplots(1, 2, figsize=(10, 5))
	axs[0].plot(k, E_vel_true_radial, label='CFD', color=color_choices[-2], linewidth=3)
	axs[0].plot(k, E_vel_pseudo_radial, label='RePIT', color=color_choices[1], linestyle='--', linewidth=2)
	axs[0].set_title("Velocity")
	axs[0].set_xlabel("Wavenumber (k)")
	axs[0].set_ylabel("Energy")
	axs[0].set_yscale("log")
	axs[0].grid(True, alpha=0.4)
	axs[0].fill_between(k, E_vel_pseudo_radial, E_vel_true_radial, color=color_choices[0], alpha=0.5, label="Error")
	axs[0].legend()

	axs[1].plot(k, E_T_true_radial, label='CFD', color=color_choices[-2], linewidth=3)
	axs[1].plot(k, E_T_pseudo_radial, label='RePIT', color=color_choices[1], linestyle='--', linewidth=2)
	axs[1].set_title("Temperature")
	axs[1].set_xlabel("Wavenumber (k)")
	axs[1].set_ylabel("Energy")
	axs[1].set_yscale("log")
	axs[1].grid(True, alpha=0.4)
	axs[1].fill_between(k, E_T_pseudo_radial, E_T_true_radial, color=color_choices[0], alpha=0.5, label="Error")
	axs[1].legend()

	plt.tight_layout()
	if save_path is None:
		save_path = prediction_dir
	save_path = os.path.join(save_path, f"spectral_analysis_{timestep}.png")
	plt.savefig(save_path, bbox_inches='tight')
	plt.close()

def plot_residual_change(
	running_times: list,
	relative_residual: list,
	residual_limit: float = 5,
	save_name: str = "relative_residual",
	save_path: str = None,
	bins: int = 100
):
	total_bins = len(running_times)
	count_limit = int(total_bins / bins) if total_bins >= bins else 1
	temp_time_list = running_times[::count_limit]
	temp_relative_residual = relative_residual[::count_limit]
	true_residual = np.ones_like(temp_relative_residual)
	plt.figure(figsize=(10, 4))
	plt.plot(temp_time_list, true_residual, ":k", label="Reference value")
	plt.plot(temp_time_list, temp_relative_residual, "-g", label="RePIT-Framework", linewidth=4)
	plt.ylim(0.1, 100)
	plt.xlabel("Timestamps")
	plt.yscale("log", base=10)
	plt.ylabel("Scaled residual")
	plt.legend()
	plt.title(f"Relative residual mass limit: {residual_limit}")
	plt.tight_layout()
	if save_path is None:
		save_path = "."
	plt.savefig(f"{save_path}/{save_name}.png")
	plt.close()

def get_probes_data(
	pred_time_list: list[int | float],
	full_time_list: list[int | float],
	ground_truth_dir: Path = None,
	prediction_dir: Path = None,
	plot_prediction_only: bool = False
):
	probes_data = {"T": {"ground_truth": defaultdict(list), "predicted": defaultdict(list)},
				   "U_x": {"ground_truth": defaultdict(list), "predicted": defaultdict(list)},
				   "U_y": {"ground_truth": defaultdict(list), "predicted": defaultdict(list)}}
	probes_labels = {"t1": 39699, "t2": 39499, "t3": 39299, "b1": 299, "b2": 499, "b3": 699}

	if pred_time_list:
		min_time = min(pred_time_list)
		max_time = max(pred_time_list)
		interval_time = round(0.01, 2)
		timestamps = np.round(np.arange(min_time, max_time + interval_time, interval_time), 2)
	else:
		timestamps = full_time_list

	if plot_prediction_only:
		timestamps = pred_time_list
	for timestamp in timestamps:
		t_data_ground_truth = np.load(ground_truth_dir / f"T_{timestamp}.npy")
		U_data_ground_truth = np.load(ground_truth_dir / f"U_{timestamp}.npy")
		ux_data_ground_truth = U_data_ground_truth[:, 0]
		uy_data_ground_truth = U_data_ground_truth[:, 1]
		if timestamp in pred_time_list:
			t_data_predicted = np.load(prediction_dir / f"T_{timestamp}_predicted.npy")
			U_data_predicted = np.load(prediction_dir / f"U_{timestamp}_predicted.npy")
		else:
			t_data_predicted = np.load(prediction_dir / f"T_{timestamp}.npy")
			U_data_predicted = np.load(prediction_dir / f"U_{timestamp}.npy")
		ux_data_predicted = U_data_predicted[:, 0]
		uy_data_predicted = U_data_predicted[:, 1]
		for probe_location in probes_labels.keys():
			probes_data["T"]["ground_truth"][probe_location].append(t_data_ground_truth[probes_labels[probe_location]].item())
			probes_data["T"]["predicted"][probe_location].append(t_data_predicted[probes_labels[probe_location]].item())
			probes_data["U_x"]["ground_truth"][probe_location].append(ux_data_ground_truth[probes_labels[probe_location]].item())
			probes_data["U_x"]["predicted"][probe_location].append(ux_data_predicted[probes_labels[probe_location]].item())
			probes_data["U_y"]["ground_truth"][probe_location].append(uy_data_ground_truth[probes_labels[probe_location]].item())
			probes_data["U_y"]["predicted"][probe_location].append(uy_data_predicted[probes_labels[probe_location]].item())
	return probes_data

def quantitative_analysis(
	pred_time_list: list[int | float],
	full_time_list: list[int | float],
	ground_truth_dir: Path = None,
	prediction_dir: Path = None,
	save_name: str = "velocity-x",
	plot_prediction_only: bool = False,
	save_path: Path = None
):
	# Save path logic
	if save_path is None:
		save_path = prediction_dir

	probes_data = get_probes_data(
		pred_time_list=pred_time_list,
		full_time_list=full_time_list,
		ground_truth_dir=ground_truth_dir,
		prediction_dir=prediction_dir,
		plot_prediction_only=plot_prediction_only
	)

	if max(full_time_list) > 20:
		linewidth = 2
	else:
		linewidth = 3

	# Save probes data for reference
	with open(Path(save_path) / "probes_data.json", "w") as f:
		json.dump(probes_data, f, indent=4)

	fig, ax = plt.subplots(2, 1, figsize=(15, 10))
	is_temp = False
	match save_name:
		case "velocity-x" | "U_x":
			ground_truth_data = probes_data["U_x"]["ground_truth"]
			predicted_data = probes_data["U_x"]["predicted"]
		case "velocity-y" | "U_y":
			ground_truth_data = probes_data["U_y"]["ground_truth"]
			predicted_data = probes_data["U_y"]["predicted"]
		case "temperature" | "T":
			ground_truth_data = probes_data["T"]["ground_truth"]
			predicted_data = probes_data["T"]["predicted"]
			is_temp = True
		case _:
			raise ValueError("Invalid save_name for probe analysis.")

	# Top Wall
	ax[0].plot(ground_truth_data["t1"], label="T1", linestyle="-", color="red", linewidth=linewidth)
	ax[0].plot(ground_truth_data["t2"], label="T2", linestyle="-", color="green", linewidth=linewidth)
	ax[0].plot(ground_truth_data["t3"], label="T3", linestyle="-", color="blue", linewidth=linewidth)
	ax[0].plot(predicted_data["t1"], linestyle="--", color="red", linewidth=linewidth)
	ax[0].plot(predicted_data["t2"], linestyle="--", color="green", linewidth=linewidth)
	ax[0].plot(predicted_data["t3"], linestyle="--", color="blue", linewidth=linewidth)
	ax[0].legend()
	ax[0].grid()
	ax[0].set_title("Top Wall")
	ax[0].set_xlabel("Timesteps")
	ax[0].set_ylabel(save_name)
	ax[0].margins(x=0)
	ax[0].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

	# Bottom Wall
	ax[1].plot(ground_truth_data["b1"], label="B1", linestyle="-", color="red", linewidth=linewidth)
	ax[1].plot(ground_truth_data["b2"], label="B2", linestyle="-", color="green", linewidth=linewidth)
	ax[1].plot(ground_truth_data["b3"], label="B3", linestyle="-", color="blue", linewidth=linewidth)
	ax[1].plot(predicted_data["b1"], linestyle="--", color="red", linewidth=linewidth)
	ax[1].plot(predicted_data["b2"], linestyle="--", color="green", linewidth=linewidth)
	ax[1].plot(predicted_data["b3"], linestyle="--", color="blue", linewidth=linewidth)
	ax[1].legend()
	ax[1].grid()
	ax[1].set_title("Bottom Wall")
	ax[1].set_xlabel("Timesteps")
	ax[1].set_ylabel(save_name)
	ax[1].margins(x=0)
	ax[1].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

	fig.tight_layout()
	plt.savefig(Path(save_path) / f"{save_name}_probe_analysis.png")
	plt.close()

def plot_everything(
	plot_start_time: float,
	plot_end_time: float,
	residual_limit: float,
	training_config: TrainingConfig,
	metrics_dir: Path,
	ground_truth_dir: Path,
	prediction_dir: Path,
	plots_dir: Path = None
):
	
	cond1 = isinstance(metrics_dir, str)
	cond2 = isinstance(ground_truth_dir, str)
	cond3 = isinstance(prediction_dir, str)
	if cond1 or cond2 or cond3:
		metrics_dir = Path(metrics_dir)
		ground_truth_dir = Path(ground_truth_dir)
		prediction_dir = Path(prediction_dir)


	full_time_list = np.round(np.arange(plot_start_time, plot_end_time, 0.01), 2)
	metrics = load_metrics(metrics_dir / "prediction_metrics.ndjson")
	time_list = metrics["Running Time"]

	# Decide where to save
	if plots_dir is None:
		plots_dir = metrics_dir / "plots"
		plots_dir.mkdir(parents=True, exist_ok=True)

	# --- MAE and L2 error plots ---
	save_name_list = ["velocity-x", "velocity-y", "temperature"]
	for save_name in [save_name_list[0], save_name_list[-1]]:
		plot_MAE(
			pred_time_list=time_list,
			ground_truth_dir=ground_truth_dir,
			prediction_dir=prediction_dir,
			var_name=save_name,
			save_path=plots_dir
		)
		plot_L2_error(
			pred_time_list=time_list,
			ground_truth_dir=ground_truth_dir,
			prediction_dir=prediction_dir,
			var_name=save_name,
			save_path=plots_dir
		)

		quantitative_analysis(
			pred_time_list=time_list,
			full_time_list=full_time_list,
			ground_truth_dir=ground_truth_dir,
			prediction_dir=prediction_dir,
			save_name=save_name,
			plot_prediction_only=False,
			save_path=plots_dir
		) 

	# --- Still comparisons ---
	still_comparisons(
		prediction_dir=prediction_dir,
		ground_truth_dir=ground_truth_dir,
		time_list=[int((plot_start_time + plot_end_time) / 2), plot_end_time],
		temp_profiles=[training_config.left_wall_temperature, training_config.right_wall_temperature],
		save_path=plots_dir
	)

	# --- Residual plot ---
	plot_residual_change(
		running_times=time_list,
		relative_residual=metrics["Relative Residual Mass"],
		residual_limit=residual_limit,
		save_name="relative_residual",
		save_path=plots_dir
	)

	# --- Streamlines (latest time) ---
	plot_streamlines_comparison(
		data_path_1=prediction_dir,
		data_path_2=None,
		ground_truth_path=ground_truth_dir,
		t=plot_end_time,
		save_path=plots_dir
	)

	# --- Spectral analysis (latest time) ---
	plot_spectral_analysis(
		prediction_dir=prediction_dir,
		ground_truth_dir=ground_truth_dir,
		timestep=plot_end_time,
		save_path=plots_dir
	)


def transfer_to_required_directory(dir_name:str, 
								   case:str, 
								   timesteps:int=10000,
								   base_dir: str = "/home/shilaj/repitframework/repitframework"):
	destination_dir = base_dir + f"/Assets/natural_convection_{case}_study/{timesteps}timesteps/{dir_name}"

	if not os.path.exists(destination_dir):
		os.makedirs(destination_dir)

	today_date = datetime.now().strftime("%Y-%m-%d")
	solver_name = Path(f"natural_convection_{case}")
	logs_path = Path("/home/shilaj/repitframework/repitframework/logs", solver_name, today_date, "Training.log")
	pred_metrics_path = Path("/home/shilaj/repitframework/repitframework/ModelDump", solver_name, "prediction_metrics.ndjson")
	training_metrics_path = Path("/home/shilaj/repitframework/repitframework/ModelDump", solver_name, "training_metrics.ndjson")
	initial_losses_path = Path("/home/shilaj/repitframework/repitframework/ModelDump", solver_name, "initial_losses.json")
	plots_path = Path("/home/shilaj/repitframework/repitframework/plots", solver_name)
	assets_path = Path("/home/shilaj/repitframework/repitframework/Assets", solver_name)
	
	files_to_move = [
		logs_path,
		pred_metrics_path,
		training_metrics_path,
		assets_path,
		initial_losses_path]
	
	for file in os.scandir(plots_path):
		if file.name.endswith(".png") or file.name.endswith(".pdf"):
			files_to_move.append(Path(file.path))

	for file in files_to_move:
		if os.path.exists(file):
			destination_file = Path(destination_dir, file.name)
			os.system(f"mv {file} {destination_file}") # because while copying we need to specify -r
			print(f"Copied {file} to {destination_file}")
		else:
			print(f"File {file} does not exist, skipping.")

if __name__ == "__main__":
	training_config = TrainingConfig()
	metrics_path = training_config.model_dir
	ground_truth_path = Path(str(training_config.assets_path)+"_backup")
	prediction_dir = training_config.assets_path
	plots_dir = training_config.plots_dir / training_config.solver_dir.name
	plot_everything(
		plot_start_time=10.0, 
		plot_end_time=110.0, 
		residual_limit=training_config.residual_threshold,
		training_config=training_config,
		metrics_dir=metrics_path,
		ground_truth_dir=ground_truth_path,
		prediction_dir=prediction_dir,
		plots_dir=plots_dir
		)
	transfer_to_required_directory("10epochs100res", "case3", 10000)