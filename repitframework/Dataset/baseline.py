from pathlib import Path
from typing import Tuple, Union, List, Literal, Optional
import shutil
import json


import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

from repitframework.Dataset.utils import (
	normalize, 
	parse_numpy, 
	match_input_dim, 
	calculate_residual
	)

class BaseDataset(Dataset):
	"""
	Base class for dataset in th RePIT framework.
	Args:
		start_time: Start time of the data.
		end_time: End time of the data.
		time_step: Time step of the data.
		dataset_dir: Directory where the numpy files are stored.
		first_training: If True, mean and std will be calculated and saved.
		vars_list: List of variables. Example: ["T", "U"]
		extended_vars_list: List of variables extended to dims. Example: ["T", "U_x", "U_y"]
		dims: Number of spatial dimensions.
		round_to: Decimal places to round time steps.
		grid_x: Number of grid points in x direction.
		grid_y: Number of grid points in y direction.
		grid_z: Number of grid points in z direction.
		grid_step: Grid step size.
		output_dims: Shape for data output, e.g., "BD", "BCD", "BCHW".
		do_normalize: Whether to normalize data.
	"""

	DATA_MISSING_MSG = (
		"\nData is missing in the directory: {}:\n"
		"You must have data from {} to {} for variables: {}. "
		"Example: {}_{}.npy\n"
	)
	INSTANCE_PRINT_MSG = (
		"\nFVMNDataset(start_time={}, end_time={}, dataset_dir={}, "
		"inputs shape={}, labels shape={})"
	)
	SELECT_DIMS = {
		"BD": (0,),
		"BCD": (0, 2),
		"BCHW": (0, 2, 3),
	}

	def __init__(
		self,
		start_time: Union[int, float],
		end_time: Union[int, float],
		time_step: Union[int, float],
		dataset_dir: Union[str, Path],
		first_training: bool = False,
		vars_list: Optional[List[str]] = ["T","U"],
		extended_vars_list: Optional[List[str]] = ["T", "U_x", "U_y"],
		dims: int = 2,
		round_to: int = 2,
		grid_x: int = 200,
		grid_y: int = 200,
		grid_z: int = 1,
		grid_step: float = 0.005,
		output_dims: Literal["BD", "BCD", "BCHW"] = "BD",
		do_normalize: bool = True,
	) -> None:
		
		super().__init__()
		self.start_time = start_time
		self.end_time = end_time
		self.time_step = time_step
		self.dataset_dir = Path(dataset_dir)
		self.first_training = first_training
		self.vars_list = vars_list or ["T", "U"]
		self.extended_vars_list = extended_vars_list or ["T", "U_x", "U_y"]
		self.dims = dims
		self.round_to = round_to
		self.output_dims = output_dims
		self.grid_x = grid_x
		self.grid_y = grid_y
		self.grid_z = grid_z
		self.grid_step = grid_step
		self.do_normalize = do_normalize

		self.time_list = self._generate_intervals()

		# Error handling for dataset directory and file presence
		if not self.dataset_dir.exists():
			raise FileNotFoundError(
				f"Dataset directory {self.dataset_dir} does not exist."
			)
		if not self._is_data_present():
			raise FileNotFoundError(
				self.DATA_MISSING_MSG.format(
					self.dataset_dir,
					self.start_time,
					self.end_time,
					self.vars_list,
					self.vars_list[0],
					self.start_time,
				)
			)

		if self.dims == 3: 
			self.SELECT_DIMS["BCHW"]= (1,2,3)
		# Load data for indexing
		self.inputs, self.labels = self._inputs_labels()

	def _is_data_present(self) -> bool:
		"""
		Checks that all required files exist for the time range and variables.
		Returns True if all present, False otherwise.
		"""
		for var in self.vars_list:
			for time in self.time_list:
				file_dir = self.dataset_dir / f"{var}_{round(time, self.round_to)}.npy"
				if not file_dir.exists():
					return False
		return True

	def _generate_intervals(self) -> List[float]:
		"""
		Generates rounded time intervals from start_time to end_time.
		Returns a list of float time values.
		"""
		time_list = []
		running_time = self.start_time
		while running_time <= self.end_time + 1e-8:  # Include last step
			time_list.append(round(running_time, self.round_to))
			running_time = round(running_time + self.time_step, self.round_to)
		return time_list


	def _prepare_input(self, time: float) -> np.ndarray:
		"""
		Loads and stacks variables for a given time step.
		Returns [var, grid_y, grid_x] or stacked shape.

		Args:
			time: Time step to load data for.

		Returns:
			np.ndarray: Stacked data for all variables at the given time. [var, grid_y, grid_x]

		Functionality:
		1. Load numpy files for each variable at the specified time.
		2. Parse the numpy data according to grid dimensions.
		3. If data has multiple dimensions, separate them accordingly.
		4. Return the stacked data as a numpy array.
		"""
		full_data_paths = [
			self.dataset_dir / f"{var}_{time}.npy" for var in self.vars_list
		]
		numpy_data = [
			parse_numpy(
				data_path,
				grid_x=self.grid_x,
				grid_y=self.grid_y,
				grid_z=self.grid_z,
				data_dim=self.dims,
			)
			for data_path in full_data_paths
		]

		temp = []
		for data in numpy_data:
			if len(data.shape) > self.dims:  # Multi-dimensional data
				for i in range(self.dims):
					temp.append(data[..., i])
			else:
				temp.append(data)
		return np.stack(temp, axis=0)  # Output: [n_vars, grid_y, grid_x]

	def _prepare_label(self, data_t: np.ndarray, data_t_next: np.ndarray) -> np.ndarray:
		"""
		Returns difference between input at t+dt and t for all variables.
		"""
		# data_t = self._prepare_input(time)
		# next_time = round(time + self.time_step, self.round_to)
		# data_t_next = self._prepare_input(next_time)
		return data_t_next - data_t

	def _normalization_routine(
		self,
		metrics_save_path: Union[str, Path],
		inputs: np.ndarray,
		labels: np.ndarray,
		read_write_flag: str = "w",
		select_dims: Optional[Tuple[int]] = None
	) -> Tuple[Tensor, Tensor]:
		"""
		Routine for normalization and saving/reading normalization statistics.
		"""
		metrics_save_path = Path(metrics_save_path)
		if select_dims is None:
			select_dims = self.SELECT_DIMS[self.output_dims]
		if read_write_flag == "r":
			if not metrics_save_path.exists():
				raise FileNotFoundError(f"Normalization metrics file missing: {metrics_save_path}")
			with open(metrics_save_path, "r") as f:
				metrics = json.load(f)
			input_mean = np.array(metrics["input_mean"])
			input_std = np.array(metrics["input_std"])
			label_mean = np.array(metrics["label_mean"])
			label_std = np.array(metrics["label_std"])

			norm_inputs, _, _ = normalize(
				inputs, mean=input_mean, std=input_std, select_dims=select_dims
			)
			norm_labels, _, _ = normalize(
				labels, mean=label_mean, std=label_std, select_dims=select_dims
			)
			return Tensor(norm_inputs), Tensor(norm_labels)

		norm_inputs, input_mean, input_std = normalize(
			inputs, select_dims=select_dims
		)
		norm_labels, label_mean, label_std = normalize(
			labels, select_dims=select_dims
		)
		with open(metrics_save_path, "w") as f:
			json.dump(
				{
					"input_mean": input_mean.tolist(),
					"input_std": input_std.tolist(),
					"label_mean": label_mean.tolist(),
					"label_std": label_std.tolist(),
					"true_residual_mass": calculate_residual(
						self.dataset_dir,
						self.end_time,
						self.grid_x,
						self.grid_y,
						self.grid_z,
						self.dims
					)
				},
				f,
				indent=4,
			)
		return Tensor(norm_inputs), Tensor(norm_labels)

	def _inputs_labels(self) -> Tuple[Tensor, Tensor]:
		"""
		Loads all input and label data for the dataset, shapes and normalizes.
		"""
		all_timesteps_data = [self._prepare_input(time) for time in self.time_list]

		inputs = []
		labels = []

		# Exclude final time since label is diff (t+1)-t
		for i in range(len(self.time_list) - 1):
			inputs.append(all_timesteps_data[i])
			labels.append(self._prepare_label(all_timesteps_data[i], all_timesteps_data[i + 1]))

		del all_timesteps_data  # Free memory
		
		inputs = match_input_dim(self.output_dims, inputs)
		labels = match_input_dim(self.output_dims, labels)
		metrics_save_path = self.dataset_dir / "norm_denorm_metrics.json"
		if self.first_training and self.do_normalize:
			normed_inputs, normed_labels = self._normalization_routine(metrics_save_path, inputs, labels, "w")
			return normed_inputs,normed_labels
		elif self.do_normalize:
			normed_inputs, normed_labels = self._normalization_routine(metrics_save_path, inputs, labels, "r")
			return normed_inputs, normed_labels
		else:
			# true residual mass is needed to determine the switching point, hence it must be saved.
			with open(metrics_save_path, "w") as f:
				json.dump({
					"true_residual_mass": calculate_residual(
						self.dataset_dir,
						self.end_time,
						self.grid_x,
						self.grid_y,
						self.grid_z,
						self.dims
					)
				})
			return Tensor(inputs), Tensor(labels)
	def __len__(self) -> int:
		return self.inputs.shape[0]

	def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
		return self.inputs[idx], self.labels[idx]

	def __repr__(self) -> str:
		return self.INSTANCE_PRINT_MSG.format(
			self.start_time,
			self.end_time,
			self.dataset_dir,
			self.inputs.shape,
			self.labels.shape,
		)

	def __iter__(self):
		for i in range(len(self)):
			yield self[i]



def create_fake_npy_files(data_dir, start_time, end_time, time_step, grid_x, grid_y):
	"""
	Generate fake scalar (T) and vector (U) .npy files for each timestep.
	"""
	time_list = []
	t = start_time
	while t <= end_time + 1e-8:
		time_list.append(round(t, 2))
		t = round(t + time_step, 2)

	for time in time_list:
		# Scalar field: T
		T = np.random.uniform(290, 310, size=(grid_x * grid_y))
		np.save(data_dir / f"T_{time}.npy", T)

		# Vector field: U (2 components)
		U = np.random.uniform(-0.5, 0.5, size=(grid_x * grid_y, 2))
		np.save(data_dir / f"U_{time}.npy", U)

def test_BaseDataset():
	tmp_dir = Path("test_tmp_data")
	if tmp_dir.exists():
		shutil.rmtree(tmp_dir)
	tmp_dir.mkdir(parents=True, exist_ok=True)

	# Hyperparameters
	start_time = 0.0
	end_time = 0.03
	time_step = 0.01
	grid_x, grid_y = 200, 200

	# Create synthetic npy files
	create_fake_npy_files(tmp_dir, start_time, end_time, time_step, grid_x, grid_y)

	# Instantiate the dataset
	dataset = BaseDataset(
		start_time=start_time,
		end_time=end_time,
		time_step=time_step,
		dataset_dir=tmp_dir,
		first_training=True,         # This will write new normalization stats
		vars_list=["T", "U"],
		dims=2,
		round_to=2,
		grid_x=grid_x,
		grid_y=grid_y,
		output_dims="BD",
		do_normalize=True
	)

	sample_input, sample_label = dataset[0]
	print(f"Input shape: {sample_input.shape}")
	print(f"Label shape: {sample_label.shape}")

	# Test DataLoader
	loader = DataLoader(dataset, batch_size=4)
	batch = next(iter(loader))
	print("Batch input shape:", batch[0].shape)
	print("Batch label shape:", batch[1].shape)

	*_, prediction_input = dataset._inputs_labels()
	print(f"Prediction input shape: {prediction_input.shape}")

	# Clean up
	shutil.rmtree(tmp_dir)

if __name__ == "__main__":
	test_BaseDataset()