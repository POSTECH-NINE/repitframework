# predictor.py

"""
Defines the BaseHybridPredictor for managing hybrid ML-CFD predictions in
natural convection simulations.

Key Features:
- Manages an autoregressive prediction loop that runs until a specified
  residual mass threshold is exceeded.
- Handles data normalization, boundary condition enforcement, and feature engineering.
- Provides methods for loading initial data, saving predictions, and logging metrics.

Important:
-  A method is provided to give the residual mass calculation function, which can be used in the `predict` method.
-  If feature selection is enabled, verify that the features are always in dimension 1.
-  In the "_normalization_metrics" method, the normalization metrics are loaded from a JSON file; hard-coded name: "norm_denorm_metrics.json".
"""


from typing import Dict, List, Union
import json
from pathlib import Path


import numpy as np
import torch

from .config import NaturalConvectionConfig
from .Dataset import normalize, denormalize, parse_numpy, add_feature, hard_constraint_bc, match_input_dim
from .Metrics.ResidualNaturalConvection import residual_mass
from .OpenFOAM.numpyToFoam import numpyToFoamDirect



# --- Constants ---
_METRICS_FILENAME = "norm_denorm_metrics.json"


class BaseHybridPredictor:
	"""
	Manages a dimension-agnostic hybrid prediction workflow, combining an ML model with physics calculations.
	"""

	def __init__(self, training_config: NaturalConvectionConfig):
		self.config = training_config
		self.variables = self.config.extend_variables()
		self.device = self.config.device
		self.relative_residual_mass = 0.0
		self.true_residual_mass = 1.0

		# Dynamically find indices for all velocity components (U_x, U_y, U_z...)
		self.velocity_indices = [
			i for i, var in enumerate(self.variables) if var.startswith("U_")
		]
		if not self.velocity_indices:
			raise ValueError(
				"No velocity variables (e.g., 'U_x') found in the variables list. "
				"Cannot calculate residual mass."
			)

	def _get_velocity_field(self, state_array: np.ndarray) -> np.ndarray:
		"""
		Extracts and assembles the velocity field from a full state array.

		Args:
			state_array: The full data array with shape [num_variables, *grid_shape].

		Returns:
			The velocity field with shape [*grid_shape, num_components].
		"""
		# Select all velocity component slices using the pre-found indices
		velocity_components = state_array[self.velocity_indices]
		# Stack components along a new, last axis
		return np.stack(velocity_components, axis=-1)

	def _get_normalization_metrics(self) -> Dict[str, np.ndarray]:
		"""Loads normalization metrics from the assets directory."""
		metrics_path = self.config.assets_dir / _METRICS_FILENAME
		with open(metrics_path, "r") as f:
			metrics = json.load(f)
		return metrics

	def _get_initial_ground_truth(self, time_step: float) -> np.ndarray:
		"""
		Loads and assembles the complete ground truth state for a given time step.
		"""
		initial_state = []
		for var_name in self.variables:
			base_var = var_name.split("_")[0]
			data = self.get_ground_truth_data(time_step, var=base_var)

			if data.ndim > len(self.config.grid_shape):  # Vector field
				component = var_name.split("_")[-1]
				component_idx = ["x", "y", "z"].index(component)
				initial_state.append(data[..., component_idx])
			else:  # Scalar field
				initial_state.append(data)

		return np.stack(initial_state, axis=0)

	def get_ground_truth_data(self, time_step: float, var: str = "U") -> np.ndarray:
		"""
		Fetches and parses a single ground truth variable from a .npy file.
		"""
		full_data_path = self.config.assets_dir / f"{var}_{time_step}.npy"
		return parse_numpy(
			dataset_file=full_data_path,
			grid_x=self.config.grid_x,
			grid_y=self.config.grid_y,
			grid_z=self.config.grid_z,
		)

	def _save_and_process_predictions(
		self, pred_data_flat: np.ndarray, time_step: float
	) -> np.ndarray:
		"""
		Saves predicted data to files and processes it for the next step.
		"""
		pred_data_gridded = [
			pred_data_flat[:, i].reshape(self.config.grid_shape).squeeze()
			for i in range(pred_data_flat.shape[1])
		]

		data_dict = {}

		for notation, var_list in self.config.data_vars.items():
			if notation == "vectors":
				for vec_name in var_list:
					# Find all components for the current vector (e.g., "U_x", "U_y")
					component_indices = {
						"x": self.variables.index(f"{vec_name}_x"),
						"y": self.variables.index(f"{vec_name}_y")
					}
					if self.config.data_dim == 3:
						component_indices["z"] = self.variables.index(f"{vec_name}_z")
					
					# Prepare components for saving
					components_to_save = [
						pred_data_gridded[idx].flatten() for idx in component_indices.values()
					]

					vector_data = np.stack(components_to_save, axis=1)
					np.save(
						self.config.assets_dir / f"{vec_name}_{time_step}_predicted.npy",
						vector_data,
					)
					data_dict[vec_name] = vector_data
			elif notation == "scalars":
				for scalar_name in var_list:
					s_idx = self.variables.index(scalar_name)
					np.save(
						self.config.assets_dir / f"{scalar_name}_{time_step}_predicted.npy",
						pred_data_gridded[s_idx].flatten(),
					)
					data_dict[scalar_name] = pred_data_gridded[s_idx].flatten()

		# # Call numpytofoam here
		# latestCFD_time = round(time_step - self.config.write_interval, self.config.round_to)
		# latestCFD_time = int(latestCFD_time) if latestCFD_time.is_integer() else latestCFD_time
		# numpyToFoamDirect(
		# 	training_config=self.config,
		# 	latestML_time=time_step,
		# 	data_dict=data_dict,
		# 	latestCFD_time=latestCFD_time,
		# 	solver_dir=self.config.solver_dir
		# )
		return np.stack(pred_data_gridded, axis=0) # Shape: [num_variables, *grid_shape]

	def _preprocess_for_model(self, data: np.ndarray) -> np.ndarray:
		"""
		Prepares data for model input by applying BCs, adding features,
		and reshaping to match the model's expected input format.
		"""
		processed_data = data
		if self.config.do_feature_selection:
			data_bc = hard_constraint_bc(
				data,
				self.variables,
				self.config.left_wall_temperature,
				self.config.right_wall_temperature,
			)
			data_features = [add_feature(d) for d in data_bc]
			processed_data = np.concatenate(data_features, axis=0)

		return match_input_dim(
			output_dims=self.config.output_dims, inputs=[processed_data]
		)

	def _advance_simulation_step(
		self, time_step: float, pred_output: np.ndarray = None
	) -> np.ndarray:
		"""
		Processes the latest prediction and prepares the input for the next step.
		"""
		if pred_output is None:  # First prediction step
			state_data = self._get_initial_ground_truth(time_step)
		else:  # Subsequent prediction steps
			state_data = self._save_and_process_predictions(pred_output, time_step)

		# Calculate residual from the current state's unified velocity field
		velocity_field = self._get_velocity_field(state_data)
		current_residual = residual_mass(velocity_field)
		self.relative_residual_mass = current_residual / self.true_residual_mass

		# Log metrics for completed steps
		if pred_output is not None:
			self.config.logger.debug(f"Relative Residual Mass: {self.relative_residual_mass:.4f}")
			self.config.log_metrics("Running Time", time_step, "prediction")
			self.config.log_metrics("Relative Residual Mass", self.relative_residual_mass, "prediction")

		return self._preprocess_for_model(state_data)

	def _run_prediction_step(
		self,
		running_time: float,
		prev_step_output: np.ndarray,
		metrics: dict,
		model: torch.nn.Module,
	) -> np.ndarray:
		"""
		Executes one full autoregressive step of the prediction loop.
		"""
		model_input = self._advance_simulation_step(running_time, prev_step_output)
		
		if self.config.do_normalize:
			norm_input, *_ = normalize(
				model_input,
				mean=np.array(metrics["input_mean"]),
				std=np.array(metrics["input_std"])
			)
		else:
			norm_input = model_input

		network_input = torch.from_numpy(norm_input).to(self.device)
		predicted_output = model(network_input)
		
		# If multiple outputs are returned, concatenate them
		if isinstance(predicted_output, Dict):
			predicted_output = torch.cat([output.cpu() for output in predicted_output.values()], dim=1)
		else:
			predicted_output = predicted_output.cpu()

		if self.config.do_normalize:
			network_output = denormalize(
				predicted_output.numpy(),
				np.array(metrics["label_mean"]),
				np.array(metrics["label_std"])
			)
		else:
			network_output = predicted_output.numpy()

		if self.config.do_feature_selection:
			skip_step = (2*self.config.data_dim)+1
			autoregressed_input = model_input[:, ::skip_step] + network_output
		else:
			autoregressed_input = model_input + network_output

		return autoregressed_input

	def predict(self, prediction_start_time: float, model: torch.nn.Module) -> float:
		"""
		Runs the main prediction loop until a condition is met.
		"""
		model.eval()
		running_time = prediction_start_time
		prediction_result = None
		metrics = self._get_normalization_metrics()
		self.true_residual_mass = metrics.get("true_residual_mass", 1.0)
		self.relative_residual_mass = 0.0 # Initialize to enter loop

		with torch.inference_mode():
			while (
				self.relative_residual_mass <= self.config.residual_threshold
				and running_time <= self.config.prediction_end_time
			):
				prediction_result = self._run_prediction_step(
					running_time=running_time,
					prev_step_output=prediction_result,
					metrics=metrics,
					model=model,
				)
				running_time = round(
					running_time + self.config.write_interval, self.config.round_to
				)

		final_time = round(
			running_time - self.config.write_interval, self.config.round_to
		)
		return final_time