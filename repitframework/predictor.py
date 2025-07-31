
from typing import Dict, List, Union
import json
from pathlib import Path


import numpy as np
import torch

from .config import NaturalConvectionConfig
from .Dataset import normalize, denormalize, parse_numpy, add_feature, hard_constraint_bc, match_input_dim
from .Metrics.ResidualNaturalConvection import residual_mass



class BaseHybridPredictor:
	def __init__(self, training_config:NaturalConvectionConfig):
		self.training_config = training_config
		self.variables = self.training_config.extend_variables()
		self.ux_index = self.variables.index("U_x")
		self.uy_index = self.variables.index("U_y")

		assert self.ux_index, "U_x and U_y must be in the variables list. Otherwise, residue calculation will not work.Hence, no swithching point."
	
	def _get_normalization_metrics(self, dataset_dir:Union[str, Path]) -> Dict[str,np.ndarray]:
		"While creatting the dataset instance, the normalization metrics are saved in a JSON file (if do_normalize is TRUE)."
		metrics_path = Path(dataset_dir) / "norm_denorm_metrics.json"
		with open(metrics_path, "r") as f:
			metrics = json.load(f)
		
		return metrics
	
	def get_ground_truth_data(
			self, 
			time_step:int|float
		) -> List[np.ndarray]:
		'''
		For the first prediction, we need ground truth data to give input to the model.
		Args
		---- 
		time_step: float: 
			The time step for which we are predicting. e.g., 5.03

		Returns
		-------
		>>> Shape: [num_variables, grid_y, grid_x]

		Functionality
		-------------
		1. Get data for the time step from ground truth data.
		2. Parse the numpy data for the variables.
		3. Separate the dimensions of the data if present.
		'''
		data_path = self.training_config.assets_dir
		variables = self.training_config.get_variables()
		full_data_path = [data_path / f"{var}_{time_step}.npy" for var in variables]
		numpy_data = [
			parse_numpy(
				dataset_file=path,
				grid_x=self.training_config.grid_x,
				grid_y=self.training_config.grid_y,
				grid_z=self.training_config.grid_z,
				data_dim=self.training_config.data_dim
			)
			for path in full_data_path
		]
		temp = list()
		for data in numpy_data:
			if len(data.shape) > 2:
				for i in range(self.training_config.data_dim):
					temp.append(data[:,:, i])
			else:
				temp.append(data)
		return np.stack(temp, axis=0)  # Shape: [num_variables, grid_y, grid_x]
	
	def save_prediction_results(self,
								pred_data:np.ndarray,
								time_step:Union[int,float]) -> np.ndarray:
		
		'''
		Save the predicted data to the assets directory.
		Args
		----
		pred_data: np.ndarray:
			The predicted data from the model.
		time_step: Union[int, float]:
			The time step for which the data is predicted.

		Returns
		-------
		The predicted data stacking the variables in the order of self.variables in the dimension 0.
		>>> [num_variables, grid_y, grid_x]
		'''
		# Regardless of the shape, throughout the framework, we always put features in dimension 1.
		pred_data = [pred_data[:, i].reshape(self.training_config.grid_y, self.training_config.grid_x) for i in range(pred_data.shape[1])]

		for notation, var in self.training_config.data_vars.items():
			if notation == "vectors":
				for i, vector in enumerate(var):
					var_Xindex = self.variables.index(vector + "_x")
					var_Yindex = self.variables.index(vector + "_y")
					Xdata:np.ndarray = pred_data[var_Xindex]
					Ydata:np.ndarray = pred_data[var_Yindex]
					if vector == "U":
						self.ux_matrix = Xdata
						self.uy_matrix = Ydata
					vector_data = np.concatenate([Xdata.reshape(-1,1), Ydata.reshape(-1,1)], axis=1)
					np.save(self.training_config.assets_dir / f"{vector}_{time_step}_predicted.npy", vector_data)
			elif notation == "scalars":
				for i, v in enumerate(var):
					var_index = self.variables.index(v)
					np.save(self.training_config.assets_dir / f"{v}_{time_step}_predicted.npy", pred_data[var_index].reshape(-1))
			else:
				raise ValueError(f"Invalid notation {notation} in data_vars. Must be either 'vectors' or 'scalars'.")

		return np.stack(pred_data, axis=0)  # Shape: [num_variables, grid_y, grid_x]
	

	def apply_boundary_conditions(self, pred_data:np.ndarray) -> np.ndarray:
		'''
		Apply the boundary conditions to the data.

		Args
		----
		data: torch.Tensor:
			The data for which the boundary conditions are to be applied.
		time_step: int|float:
			The time step for which the boundary conditions are to be applied.
		data_path: Path:
			The path where the data is stored.
		'''
		if self.training_config.do_feature_selection:
			pred_data = np.concatenate([add_feature(data) for data in hard_constraint_bc(
				pred_data,
				extended_vars_list=self.variables,
				left_wall_temperature=self.training_config.left_wall_temperature,
				right_wall_temperature=self.training_config.right_wall_temperature
				)], axis=0)
		temp = match_input_dim(
			output_dims=self.training_config.output_dims,
			inputs= [pred_data]
		)
		return temp
	
	def prepare_input_for_prediction(self, time_step:int|float,  
									 prediction_input:np.ndarray=None) -> np.ndarray:
		'''
		If feature selection is enabled, boundary values need to be enforced.

		Args
		----
		time_step: int|float:
			If we are predicting for t then time_step = t-dt.
		data: torch.Tensor: 
			The output from the model after denormalizing and adding with the input [batch_size, num_features]
		data_path: Path: 
			if we predict for time step 5.03 then we need the original data for the time step 5.03 to get the boundary data.

		Functionality
		-------------
		1. Because, we are using the same training_config.get_variables() to get the variables.
		   We leverage this to get the index of U_x, U_y, T.
		2. If it is not the first prediction, we are setting U_x and T values in that iteration as previous values 
		   and as the process progresses, we update the previous values with the predicted values.
		3. We save the predicted values here. In the prediction loop, we get the output for time(running_time) + dt.
		   So, it makes sense that we can update the running time, and while preparing input for the next prediction, 
		   we can add boundary values to the prev. predicted values and that would represent the predicted values for 
		   currently running_time in prediction loop.

		Reasoning
		---------
		But why did we assign the present/previous ux_matrix, uy_matrix, t_matrix in this function? 
		Because, we would have input and output data both in the predict method. Wouldn't it make sense to assign the values there?

		Sadly NO.
		Because, the input for the network is feature extracted. Example shape: [40000,15]
		And the output from the network is boundary excluded data. Example shape: [39204,3]
		Hence, we must do the post-processing before calculating the residue. So, for me, 
		it made a lot of sense to assign the values here. If you have a better idea, please let me know.
		'''
		if prediction_input is None:
			# If it is the first prediction, we need to get the ground truth data.
			ground_truth = self.get_ground_truth_data(time_step)
			self.relative_residual_mass = residual_mass(ground_truth[self.ux_index], ground_truth[self.uy_index])/self.true_residual_mass

			return self.apply_boundary_conditions(ground_truth)
		
		pred_data = self.save_prediction_results(prediction_input, time_step)

		# Save residual mass for the time step
		predicted_residual_mass = residual_mass(self.ux_matrix, self.uy_matrix, order="C")
		self.relative_residual_mass = predicted_residual_mass / self.true_residual_mass

		self.training_config.logger.debug(f"Relative Residual Mass: {self.relative_residual_mass:.4f}")
		self.training_config.log_metrics(key="Running Time", value=time_step, metrics_type="prediction")
		self.training_config.log_metrics(key="Relative Residual Mass", value=self.relative_residual_mass, metrics_type="prediction")

		return self.apply_boundary_conditions(pred_data)
	
	def prediction_loop(self, running_time:float,
					 prediction_input:np.ndarray,
					 metrics:dict,
					 model:torch.nn.Module) -> np.ndarray:
		
		prediction_input:np.ndarray = self.prepare_input_for_prediction(
					time_step=running_time,
					prediction_input=prediction_input
				)
		
		if self.training_config.do_normalize:
			network_input, *_ = normalize(prediction_input,mean=np.array(metrics["input_mean"]),std=np.array(metrics["input_std"]))
		else:
			network_input = prediction_input
		# Convert to torch tensor and move to device
		network_input:torch.Tensor = torch.from_numpy(network_input).to(self.training_config.device)
		predicted_output:torch.Tensor = model(network_input)

		# Denormalize the output
		if isinstance(predicted_output, Dict):
			predicted_output = torch.cat([output.cpu() for output in predicted_output.values()], dim=1)

		if self.training_config.do_normalize:
			network_output = denormalize(
				predicted_output.numpy(),
				np.array(metrics["label_mean"]),
				np.array(metrics["label_std"])
			)
		else:
			network_output = predicted_output.numpy()

		if self.training_config.do_feature_selection:
			# Always on dimension 1, we have the features.
			auto_regressed_input = prediction_input[:,::5] + network_output
		else:
			auto_regressed_input = prediction_input + network_output

		return auto_regressed_input

	def predict(self,prediction_start_time:float, model:torch.nn.Module)-> float:

		model.eval()
		# Initialize prediction start parameters:
		running_time = prediction_start_time
		prediction_input = None
		self.relative_residual_mass = self.training_config.residual_threshold # This will be updated during the prediction loop.

		metrics = self._get_normalization_metrics(self.training_config.assets_dir)
		self.true_residual_mass = metrics["true_residual_mass"]

		with torch.inference_mode():
			while (self.relative_residual_mass <= self.training_config.residual_threshold) and (running_time <= self.training_config.prediction_end_time):
				prediction_input = self.prediction_loop(
					running_time=running_time,
					prediction_input=prediction_input,
					metrics=metrics,
					model=model
				)
				# Update the running time
				running_time = round(running_time + self.training_config.write_interval, self.training_config.round_to)

			# Because prepare_input_for_prediction function calculates the residual values.
			# Hence, even if the residue value exceeds the threshold, the running time will be updated.
			# So, we need to step down the running time by the write interval outside the loop.
		return round(running_time - self.training_config.write_interval, self.training_config.round_to)
	
	def predict_batched(self, model: torch.nn.Module, inputs):
		model.eval()
		# If model returns a dict, create per-key storage
		all_outputs = None
		with torch.no_grad():
			for i in range(0, inputs.shape[0], self.training_config.batch_size):
				batch = inputs[i:i+self.training_config.batch_size].to(self.training_config.device)
				outputs = model(batch)
				if isinstance(outputs, dict):
					if all_outputs is None:
						all_outputs = {k: [] for k in outputs}
					for k, v in outputs.items():
						all_outputs[k].append(v.cpu())
				else:
					if all_outputs is None:
						all_outputs = []
					all_outputs.append(outputs.cpu())
		# Concatenate appropriately
		if isinstance(all_outputs, dict):
			return {k: torch.cat(v, dim=0) for k, v in all_outputs.items()}
		else:
			return torch.cat(all_outputs)