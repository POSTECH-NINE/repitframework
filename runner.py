"""RePIT Framework: Hybrid CFD-ML Simulation Framework with Transfer Learning"""

from __future__ import annotations
from pathlib import Path
import timeit
import json
from typing import Tuple, List

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

torch.set_default_dtype(torch.float64)
torch.manual_seed(1004)
torch.cuda.manual_seed_all(1004)
np.random.seed(1004)

from repitframework.Dataset.fvmn import FVMNDataset
from repitframework.DataLoader.loader import train_val_split
from repitframework.Models.FVMN.fvmn import FVMNetwork
from repitframework.config import TrainingConfig, OpenfoamConfig
from repitframework.OpenFOAM import OpenfoamUtils
from repitframework.OpenFOAM import numpyToFoam
from repitframework.Metrics.ResidualNaturalConvection import (
	residual_mass, 
	residual_momentum, 
	residual_heat
)
from repitframework.Metrics.OperatorEmbeddings import compute_gradient, ceod_loss
from repitframework.plot_utils import save_loss
from utils import freeze_layers, prepare_model_and_optimizer

class Trainer:
	def __init__(self, training_config:TrainingConfig, 
				 model:torch.nn.Module, 
				 optimizer:torch.optim.Adam, 
				 loss:torch.nn.MSELoss, 
				 model_name:str=None):
		self.training_config = training_config
		self.device = training_config.device
		self.model = model
		self.model.to(self.device)
		self.optimizer:torch.optim.Adam = optimizer(
			self.model.parameters(), 
			lr=self.training_config.learning_rate)

		# Load the model if model_name is provided
		if model_name:
			self.load_model(model_name, 
				load_optimizer=False, 
				learning_rate=self.training_config.learning_rate)
			self.training_config.epochs = 0 # To skip initial training for 5000 epochs and use the best saved model from this training.

		self.loss = loss
		self.best_val_accuracy = float("inf")

		self.residual_threshold = training_config.residual_threshold
		self.relative_residual_mass = float()
		self.true_residual_mass = float()

		self.ux_matrix = torch.zeros((training_config.grid_y, training_config.grid_x))
		self.uy_matrix = torch.zeros((training_config.grid_y, training_config.grid_x))
		self.t_matrix = torch.zeros((training_config.grid_y, training_config.grid_x))
		self.t_matrix_prev = torch.zeros((training_config.grid_y, training_config.grid_x))
		self.ux_matrix_prev = torch.zeros((training_config.grid_y, training_config.grid_x))

		self.variables = self.training_config.get_variables()
		self.ux_index = self.variables.index("U_x")
		self.uy_index = self.variables.index("U_y")
		self.t_index = self.variables.index("T")

	def loss_fn(self, pred:torch.Tensor, true:torch.Tensor) -> torch.Tensor:
		'''
		Calculates the relative loss between the predicted and true values.
		'''
		return self.loss(pred,true)
	
	def train(self, train_loader:DataLoader, 
			  val_loader:DataLoader, 
			  epochs, freeze:bool) -> bool:
		
		if freeze: 
			self.model = freeze_layers(self.model, num_layers=2)
		for epoch in tqdm(range(epochs), desc="Epochs", leave=False):
			self.model.train()  # Set the model to training mode
			train_loss = 0.0
			for x_batch, y_batch in train_loader:
				x_batch = x_batch.to(self.device) 
				y_batch = y_batch.to(self.device)

				# Labels: 
				y_T = y_batch[:,self.t_index:self.t_index+1]
				y_ux = y_batch[:, self.ux_index:self.ux_index+1]
				y_uy = y_batch[:, self.uy_index:self.uy_index+1]

				# Forward pass: Hard coded
				predictions = self.model(x_batch)
				pred_T = predictions["T"]
				pred_ux = predictions["U_x"]
				pred_uy = predictions["U_y"]

				loss_T = self.loss_fn(pred_T, y_T)
				loss_ux = self.loss_fn(pred_ux, y_ux)
				loss_uy = self.loss_fn(pred_uy, y_uy)
				loss = loss_T + loss_ux + loss_uy

				# Backpropagation
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				train_loss += loss.item()*x_batch.size(0)
			
			train_loss /= len(train_loader.dataset)

			self.training_config.log_metrics(key="Epoch", value=epoch+1, metrics_type="training")
			self.training_config.log_metrics(key="Training Loss", value=train_loss, metrics_type="training")
			training_config.logger.debug(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}")

			# Validation loss
			val_loss = self.validate(val_loader)
			if val_loss < self.best_val_accuracy:
				self.best_val_accuracy = val_loss
				self.save_model(f"best_model.pth")
			self.training_config.log_metrics(key="Validation Loss", value=val_loss, metrics_type="training")

		return True

	def validate(self, val_loader:DataLoader):
		self.model.eval()  # Set the model to evaluation mode
		val_loss = 0.0
		with torch.no_grad():
			for x_val, y_val in val_loader:
				x_val = x_val.to(self.device) 
				y_val = y_val.to(self.device)

				# Labels: 
				y_T = y_val[:,self.t_index:self.t_index+1]
				y_ux = y_val[:, self.ux_index:self.ux_index+1]
				y_uy = y_val[:, self.uy_index:self.uy_index+1]

				# Forward pass: Hard coded
				predictions = self.model(x_val)
				pred_T = predictions["T"]
				pred_ux = predictions["U_x"]
				pred_uy = predictions["U_y"]

				loss_T = self.loss_fn(pred_T, y_T)
				loss_ux = self.loss_fn(pred_ux, y_ux)
				loss_uy = self.loss_fn(pred_uy, y_uy)

				# ceod_T = ceod_loss(pred_T, y_T, compute_gradient)
				# ceod_ux = ceod_loss(pred_ux, y_ux, compute_gradient)
				# ceod_uy = ceod_loss(pred_uy, y_uy, compute_gradient)

				# loss = 0.7*(loss_T+loss_ux+loss_uy)+0.3*(ceod_T + ceod_ux + ceod_uy)
				loss = loss_T + loss_ux + loss_uy
				val_loss += loss.item() * x_val.size(0)
		
		val_loss /= len(val_loader.dataset)
		training_config.logger.debug(f"Validation Loss: {val_loss:.4f}")
		return val_loss
	
	def train_with_PIL(self, train_loader:DataLoader, 
			  val_loader:DataLoader, 
			  epochs, freeze:bool,
			  switch_count:int=1) -> bool:
		
		self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.training_config.learning_rate)
		if freeze: freeze_layers(self.model, num_layers=4)
		for epoch in tqdm(range(epochs), desc="Epochs", leave=False):
			self.model.train()  # Set the model to training mode
			train_loss = 0.0
			for x_batch, y_batch in train_loader:
				x_batch = x_batch.to(self.device) 
				y_batch = y_batch.to(self.device)

				# Labels: 
				y_T = y_batch[:,self.t_index:self.t_index+1]
				y_ux = y_batch[:, self.ux_index:self.ux_index+1]
				y_uy = y_batch[:, self.uy_index:self.uy_index+1]

				# Forward pass: Hard coded
				predictions = self.model(x_batch)
				pred_T = predictions["T"]
				pred_ux = predictions["U_x"]
				pred_uy = predictions["U_y"]

				loss_T = self.loss_fn(pred_T, y_T)
				loss_ux = self.loss_fn(pred_ux, y_ux)
				loss_uy = self.loss_fn(pred_uy, y_uy)

				ceod_T = ceod_loss(pred_T, y_T, compute_gradient)
				ceod_ux = ceod_loss(pred_ux, y_ux, compute_gradient)
				ceod_uy = ceod_loss(pred_uy, y_uy, compute_gradient)
				#TODO: hard coded
				# Calculate the residuals
				loss_residual = residual_mass(ux_matrix=pred_ux.reshape(200,200).transpose(0,1), uy_matrix=pred_uy.reshape(200,200).transpose(0,1))
				loss =  0.7*(loss_T+loss_ux+loss_uy)+0.3*(ceod_T + ceod_ux + ceod_uy)


				# if switch_count % 10 == 0:
				# 	loss += loss_residual
				# 	self.training_config.logger.debug(f"Residual Loss: {loss_residual:.4f}")
				# Backpropagation
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				train_loss += loss.item()*x_batch.size(0)
			
			train_loss /= len(train_loader.dataset)

			self.training_config.log_metrics(key="Epoch", value=epoch+1, metrics_type="training")
			self.training_config.log_metrics(key="Training Loss", value=train_loss, metrics_type="training")
			training_config.logger.debug(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}")

			# Validation loss
			val_loss = self.validate(val_loader)
			if val_loss < self.best_val_accuracy:
				self.best_val_accuracy = val_loss
				self.save_model(f"best_model.pth")
			self.training_config.log_metrics(key="Validation Loss", value=val_loss, metrics_type="training")
		return True

	def _normalize(self, data:torch.Tensor, mean:np.ndarray, std:np.ndarray):
		data = data.numpy()
		data = (data-mean) / std
		return torch.from_numpy(data).double()

	def _denormalize(self, data:torch.Tensor, mean:np.ndarray, std:np.ndarray):
		data = data.numpy()
		data = (data * std) + mean
		return torch.from_numpy(data).double()
	
	
	def predict(self, prediction_start_time:int|float=None, 
				write_interval:int|float=None, 
				data_path:Path=None):
		"""
		Parameters
		----------
		prediction_start_time: int|float:
			From which time step to start the prediction.
			Default is the prediction_start_time from the training config.
		write_interval: int|float:
			The interval at which the data is written.
			Default is the write_interval from the training config.
		data_path: Path:
			The path to the data directory.
		"""


		start_time = prediction_start_time if prediction_start_time else self.training_config.prediction_start_time
		time_step = write_interval if write_interval else self.training_config.write_interval
		data_path = Path(data_path) if data_path else self.training_config.assets_path 

		self.model.eval()
		prediction_input = None
		running_time = start_time # Because we saving the prediction data at prepare_input_for_prediction function. But, output is after calling this function.
		# Load the mean and std from the training data: 
		metrics_path = self.training_config.model_dir / "denorm_metrics.json"
		with open(metrics_path, "r") as f:
			metrics = json.load(f)
		label_mean = np.array(metrics["label_MEAN"])
		label_std = np.array(metrics["label_STD"])
		input_mean = np.array(metrics["input_MEAN"])
		input_std = np.array(metrics["input_STD"])
		self.true_residual_mass = metrics["true_residual_mass"]

		# Initialize relative residual mass
		pseudo_ground_truth = self.get_ground_truth_data(running_time, data_path)
		self.relative_residual_mass = residual_mass(ux_matrix=pseudo_ground_truth[self.ux_index],
														uy_matrix=pseudo_ground_truth[self.uy_index])/self.true_residual_mass
		with torch.inference_mode():
			while (self.relative_residual_mass <= self.residual_threshold) and (running_time <= self.training_config.prediction_end_time):
				prediction_input = self.prepare_input_for_prediction(running_time, data_path, prediction_input)
				normalized_input  = self._normalize(prediction_input, input_mean, input_std)
				predicted_output:torch.Tensor = self.model(normalized_input.to(self.device))
				predicted_output_concat = torch.cat([output.cpu() for output in predicted_output.values()], dim=1)
				denormed_output = self._denormalize(predicted_output_concat, label_mean, label_std)
				prediction_input = prediction_input[:, ::5] + denormed_output
				running_time = round(running_time+time_step, self.training_config.round_to)
				
			# Because prepare_input_for_prediction function calculates the residual values.
			# Hence, even if the residue value exceeds the threshold, the running time will be updated.
			# So, we need to step down the running time by the write interval outside the loop.
		return round(running_time-time_step, self.training_config.round_to)

	def save_model(self, model_name:str) -> Path:
		'''
		Parameters
		----------
		model_name: str: 
			The name of the model to be saved.
			Example: "model.pth"

		Returns
		-------
		model_save_path: Path: 
			The path where the model is saved.
			It will be saved in the repiframework/ModelDump/{case_name} directory.
		'''
		path = Path.joinpath(self.training_config.model_dir, model_name)
		torch.save({
			"model_state_dict": self.model.state_dict()
			}, 
			path
		)
		self.training_config.logger.debug(f"Model saved as {model_name} at {self.training_config.model_dir}")
		return path

	def load_model(self, 
			model_name:str, 
			load_optimizer:bool=False,
			learning_rate:float=1e-3
			) -> torch.nn.Module:
		"""
		This is for transfer learning. We load the model from the saved model.
		Epochs= 20
		"""
		path = Path.joinpath(self.training_config.model_dir, model_name)
		checkpoint = torch.load(path, weights_only=True)
		self.model.load_state_dict(checkpoint["model_state_dict"])
		if load_optimizer and "optimizer_state_dict" in checkpoint:
			self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
			for param_group in self.optimizer.param_groups:
				param_group["lr"] = learning_rate
		else:
			trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
			self.optimizer = self.training_config.optimizer(trainable_params, lr=learning_rate)
		self.training_config.logger.debug(f"Model loaded from {path}")
		return self.model.to(self.device)
	
	def get_ground_truth_data(
			self, 
			time_step:int|float, 
			data_path:Path=None
		) -> List[np.ndarray]:
		'''
		Because in FVMN, we are only predicting the interior points, we need to add the boundary data to the model output.
		Also, we need to calculate the residue. Hence, we need the true data for the time step.

		Args
		---- 
		data_path: Path: 
			If we predict for time step 5.03 then we need the original data for the 
			time step 5.03 to get the boundary data.This is the path to that data.
		time_step: float: 
			The time step for which we are predicting. e.g., 5.03
		first_prediction: bool: # TODO: This is not used. Remove it.
			If this is the first prediction, we return the whole data. 
			Else, we set the all the other except boundary to zero.

		Returns
		-------
		Each numpy array is the data for each variable separated dimension wise:
		e.g., [U_x, U_y, T] for each variable.
		
		ground_truth_data: List[np.ndarray]:
			Along with boundary values, we send the true values also.

		Functionality
		-------------
		1. Get the boundary data for the time step from ground truth data.
		2. Parse the numpy data for the variables.
		3. Separate the dimensions of the data if present. 
		5. Because to calculate the residue, we need true values also. 
		   Hence, this method returns true values also. 
		'''
		data_path = data_path if data_path else self.training_config.assets_path
		variables = self.training_config.extend_variables()
		full_data_path = [data_path / f"{var}_{time_step}.npy" for var in variables]
		numpy_data = [FVMNDataset.parse_numpy(self.training_config, data_path) for data_path in full_data_path]
		temp = list()
		for data in numpy_data:
			if len(data.shape) > 2:
				for i in range(self.training_config.data_dim):
					temp.append(data[:,:, i])
			else:
				temp.append(data)
		return temp
	
	def apply_boundary_conditions(self, pred_data:torch.Tensor, 
									time_step:int|float=None,
									data_path:Path=None,
									bc_type:str=None) -> torch.Tensor:
		'''
		Apply the boundary conditions to the data.

		Args
		----
		data: torch.Tensor:
			The data for which the boundary conditions are to be applied.
		type: str:
			Whether to add boundary values from ground truth or soft-constrained boundary values.\n
			None -> hard-constrained boundary values.\n \
			ground_truth -> ground truth boundary values.
		'''
		pred_data = pred_data.numpy()
		assert time_step and data_path, "Time step and Data path are required!"
			
		if bc_type == "ground_truth":
			data_path = Path(str(data_path) + "_backup")
			ground_truth = self.get_ground_truth_data(time_step, data_path)

			# Modeling predicted data: adding zero padding to the predicted data.
			predicted_data_grid_x = self.training_config.grid_x - 2
			predicted_data_grid_y = self.training_config.grid_y - 2
			assert pred_data.shape[0] == predicted_data_grid_x * predicted_data_grid_y,\
				  f"Shape of the data is {pred_data.shape} but should be {(predicted_data_grid_x * predicted_data_grid_y, pred_data.shape[-1])}"
			pred_data = [pred_data[:, i].reshape(self.training_config.grid_y-2, self.training_config.grid_x-2,order="F") for i in range(pred_data.shape[-1])]

			# Copying just the boundary values from the ground truth data:
			for i in range(len(ground_truth)): ground_truth[i][1:-1, 1:-1] = 0 # Setting the internal nodes to zero.
			pad_pred_data = [np.pad(pred_data[i], 1, mode="constant",constant_values=0) for i in range(len(pred_data))]

			# Adding the zero padded predicted data to the zeroed internal nodes in the ground truth data.
			pred_data = [np.add(t,d) for t,d in zip(ground_truth, pad_pred_data)]

			# Saving the predicted data: True prediction are the ones after applying the boundary conditions.
			self.ux_matrix = pred_data[self.ux_index]
			self.uy_matrix = pred_data[self.uy_index]
			self.t_matrix = pred_data[self.t_index]
			
			return pred_data

		assert pred_data.shape[0] == self.training_config.grid_y * self.training_config.grid_x, \
			f"Shape of the data is {pred_data.shape} but should be {(self.training_config.grid_y * self.training_config.grid_x, pred_data.shape[-1])}"
		
		pred_data = [pred_data[:, i].reshape(self.training_config.grid_y, self.training_config.grid_x, order="F") for i in range(pred_data.shape[-1])]

		# Saving the predicted data: True prediction are the ones before applying the boundary conditions.
		self.ux_matrix = pred_data[self.ux_index]
		self.uy_matrix = pred_data[self.uy_index]
		self.t_matrix = pred_data[self.t_index]

		# Applying the boundary conditions
		pred_pad_data = self.training_config.hard_contraint_bc(pred_data)
		
		return pred_pad_data

	def calculate_residuals(self, time_step:int|float)->float:
		'''
		Calculate the residuals for the predicted data.
		'''
		# Calculate the residuals
		predicted_residual_mass = residual_mass(ux_matrix=self.ux_matrix, uy_matrix=self.uy_matrix)
		predicted_residual_momentum = residual_momentum(ux_matrix=self.ux_matrix, ux_matrix_prev=self.ux_matrix_prev,
															uy_matrix=self.uy_matrix, t_matrix=self.t_matrix)
		predicted_residual_heat = residual_heat(ux_matrix=self.ux_matrix, uy_matrix=self.uy_matrix,
													t_matrix=self.t_matrix, t_matrix_prev=self.t_matrix_prev)
		
		# Calculate the relative residual mass: 
		relative_residual_mass = predicted_residual_mass / self.true_residual_mass
		self.training_config.logger.debug(f"Relative Residual Mass: {relative_residual_mass}")

		self.training_config.log_metrics(key="Running Time", value=time_step)
		self.training_config.log_metrics(key="Predicted Residual Mass", value=predicted_residual_mass)
		self.training_config.log_metrics(key="True Residual Mass", value=self.true_residual_mass)
		self.training_config.log_metrics(key="Predicted Residual Momentum", value=predicted_residual_momentum)
		self.training_config.log_metrics(key="Predicted Residual Heat", value=predicted_residual_heat)
		self.training_config.log_metrics(key="Relative Residual Mass", value=relative_residual_mass)

		# Update the previous values
		self.ux_matrix_prev = self.ux_matrix
		self.t_matrix_prev = self.t_matrix

		return relative_residual_mass		
	
	def prepare_input_for_prediction(self, time_step:int|float, 
									 data_path:Path, 
									 data:torch.Tensor=None) -> torch.Tensor:
		'''
		Adds ground truth boundary data to the predicted data and returns the feature selected input for the model.

		Args
		----
		time_step: int|float:
			If we are predicting for t then time_step = t-dt.
		data: torch.Tensor: 
			The output from the model after denormalizing and adding with the input [batch_size, num_features]
		data_path: Path: 
			if we predict for time step 5.03 then we need the original data for the time step 5.03 to get the boundary data.
		first_prediction: bool: 
			If this is the first prediction, we return the whole data. Else, we set the all the other except boundary to zero.

		Functionality
		-------------
		1. Get the boundary data for the time step from ground truth data.
		2. If this is not the first prediction, do the zero padding on the boundary of predicted data [198,198] -> [200,200].
		3. If not first prediction, internal nodes in the true data are set to zero.
		4. Because, we are using the same training_config.get_variables() to get the variables.
		   We leverage this to get the index of U_x, U_y, T.
		5. If it is not the first prediction, we are setting U_x and T values in that iteration as previous values 
		   and as the process progresses, we update the previous values with the predicted values.
		6. We save the predicted values here. In the prediction loop, we get the output for time(running_time) + dt.
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
		if data is None:
			# If it is the first prediction, we need to get the ground truth data.
			ground_truth = self.get_ground_truth_data(time_step, data_path)
			self.ux_matrix_prev = ground_truth[self.ux_index]
			self.t_matrix_prev = ground_truth[self.t_index]
			# self.true_residual_mass = residual_mass(ground_truth[self.ux_index], ground_truth[self.uy_index])

			self.relative_residual_mass = residual_mass(ground_truth[self.ux_index], ground_truth[self.uy_index])/self.true_residual_mass
			
			if self.training_config.bc_type != "ground_truth":
				ground_truth = self.training_config.hard_contraint_bc(ground_truth)

			temp_ = [FVMNDataset.add_feature(data) for data in ground_truth]
			input_data = np.concatenate(temp_, axis=-1)
			return torch.from_numpy(input_data)
		
		temp = self.apply_boundary_conditions(data, time_step, data_path, bc_type=self.training_config.bc_type)

		##################### Saving the predicted values #####################
		u_vector = np.concatenate([self.ux_matrix.reshape(-1,1, order="F"),
									self.uy_matrix.reshape(-1,1, order="F")], axis=-1)
		t_scalar = self.t_matrix.reshape(-1, order="F")
		np.save(data_path / f"U_{time_step}_predicted.npy", u_vector)
		np.save(data_path / f"T_{time_step}_predicted.npy", t_scalar)
		self.training_config.logger.debug(f"Saved variables at {data_path}")
		##################### Saved the predicted values #####################

		self.relative_residual_mass = self.calculate_residuals(time_step)
		
		temp_ = [FVMNDataset.add_feature(data) for data in temp]
		data = np.concatenate(temp_, axis=1)

		return torch.from_numpy(data)

def dynamic_parameters(switch_count:int)-> Tuple[int, int]:
	'''
	Returns the number of epochs and cfd runs based on the switch count.
	Args
	----
	switch_count: int:
		The number of times ML-CFD switch has happened.
	Returns
	-------
	epochs: int:
		The number of epochs to train the model.
	cfd_runs: int:
		The number of cfd runs to perform.
	'''
	if switch_count % 10 == 0:
		increment_factor_epochs = np.log(switch_count).item()
		increment_factor_cfd = np.log(switch_count**2).item()
		return int(2 + increment_factor_epochs), int(10 + increment_factor_cfd)
	return 2, 10

# TODO: this is just to test the relative loss effectiveness of the model.
def relative_loss(pred:torch.Tensor, true:torch.Tensor) -> torch.Tensor:
	'''
	Calculates the relative loss between the predicted and true values.
	This is just to test the relative loss effectiveness of the model.
	'''
	return torch.mean(torch.abs(pred - true) / (torch.abs(true) + 1e-16))

def hybrid_training( 
	openfoam_config:OpenfoamConfig, 
	training_config:TrainingConfig,
	network_type:torch.nn.Module=FVMNetwork,
	dataset_type:Dataset=FVMNDataset,
	) -> None:

	

	# Variables:
	# Training
	training_start_time = training_config.training_start_time
	training_end_time = training_config.training_end_time
	running_time = training_start_time
	optimizer = training_config.optimizer
	loss = training_config.loss # TODO: Use relative_loss for testing the relative loss effectiveness of the model.

	# Create model instance
	model = network_type(training_config)
	openfoam_utils = OpenfoamUtils(openfoam_config)

	trainer = Trainer(
		training_config=training_config, 
		model=model, 
		optimizer=optimizer, 
		loss=loss,
		model_name= "init_model.pth" # Set this to None if you don't want to use pre-trained model.
	)
	# Create trainer instance
	first_training = True

	# Storing times 
	cfd_times = 0.0
	ml_times = 0.0
	update_times = 0.0

	switch_count = 0
	ml_timesteps = 0
	cfd_timesteps = 0

	##################### RePIT: START #####################
	framework_start_time = timeit.default_timer()

	training_config.logger.info(f"Framework started at {framework_start_time}")
	while running_time < training_config.prediction_end_time:
		# Run CFD first:
		cfd_start_time = timeit.default_timer()
		openfoam_utils.run_solver(
			start_time=running_time, 
			end_time=training_end_time,
			save_to_numpy=True
		)
		cfd_end_time = timeit.default_timer()
		if training_end_time >= trainer.training_config.prediction_end_time: break
		# Create dataset instance
		dataset = dataset_type(
			training_config=trainer.training_config,
			first_training=first_training, 
			start_time= training_start_time, 
			end_time=training_end_time, 
			time_step=trainer.training_config.write_interval
		)
		train_loader, val_loader = train_val_split(
			dataset, 
			batch_size=trainer.training_config.batch_size,
			train_size=2/3
		)

		# Train the model
		update_start_time = timeit.default_timer()
		trainer.train(
			train_loader, 
			val_loader, 
			trainer.training_config.epochs,
			freeze=not first_training
		)
		update_end_time = timeit.default_timer()

		trainer.best_val_accuracy = float("inf") # Reset the best validation accuracy for transfer learning
		# Before prediction, load the best model: because we are using the same instance of self.model for prediction, hence last trained parameters will be used.
		# trainer.model, trainer.optimizer = load_from_state_dict(
		# 	model=trainer.model,
		# 	model_save_path=trainer.training_config.model_dir,
		# 	model_name="best_model.pth",
		# 	optimizer=trainer.optimizer
		# )
		trainer.model = trainer.load_model(
			model_name="best_model.pth",
			load_optimizer=False,  # We don't need to load the optimizer state dict for prediction.
			learning_rate=trainer.training_config.learning_rate
		)

		if trainer.training_config.epochs == 5000: 
			trainer.save_model("init_model.pth")
			save_loss(
				training_config=training_config,
				save_initial_losses=True
			)
			framework_start_time = timeit.default_timer() # Reset the framework start time for the next iteration.
		
		print("\nStarting prediction from: ", 
			round(training_end_time+trainer.training_config.write_interval,2)
		)
		# Store times 
		if running_time > training_config.training_start_time:
			cfd_times += cfd_end_time - cfd_start_time
			update_times += update_end_time - update_start_time
	
		ml_start_time = timeit.default_timer()

		running_time = trainer.predict(
			prediction_start_time=training_end_time, 
			write_interval=trainer.training_config.write_interval
		)

		ml_end_time = timeit.default_timer()
		ml_times += ml_end_time - ml_start_time

		# ML timesteps per cross-computation
		ml_timesteps += round((running_time - training_end_time)/trainer.training_config.write_interval)
		print("ML timesteps: ", round((running_time - training_end_time)/trainer.training_config.write_interval))
		print("Switch count: ", switch_count)
		switch_count += 1
		print(f"Prediction ended at:{running_time}\n")

		# Convert predicted numpy to foam
		numpyToFoam_string = numpyToFoam(
			openfoam_config=openfoam_config, 
			latestML_time=float(running_time), 
			latestCFD_time=training_end_time,
		)

		openfoam_config.logger.info(f"Converted numpy to foam: {numpyToFoam_string}")

		# Transfer learning
		# trainer.training_config.epochs, cfd_runs = dynamic_parameters(switch_count)
		trainer.training_config.epochs = 2
		cfd_runs = 10
		# if switch_count == 2: break
		cfd_timesteps += cfd_runs
		training_end_time = round(running_time + cfd_runs*trainer.training_config.write_interval,
								  2)
		# Just using last three time steps for transfer learning: 
		training_start_time = round(training_end_time - 3*trainer.training_config.write_interval, 2)
		first_training = False

	framework_end_time = timeit.default_timer()
	training_config.logger.info(f"\n\nFramework ended at {framework_end_time}")

	training_config.logger.info(f"Transfer learning epochs: {trainer.training_config.epochs}")
	training_config.logger.info(f"Relative Residual Mass: {trainer.training_config.residual_threshold}\n")
	training_config.logger.info(f"Total CFD Time: {cfd_times}")
	training_config.logger.info(f"Total ML Time: {ml_times}")
	training_config.logger.info(f"Total Update Time: {update_times}")
	training_config.logger.info(f"Total Framework Time: {framework_end_time-framework_start_time}")
	training_config.logger.info(f"Total ML timesteps: {ml_timesteps}")
	training_config.logger.info(f"Total CFD runs: {cfd_timesteps}")
	training_config.logger.info(f"Total Switch Count: {switch_count}")

	if_CFD_alone = (ml_timesteps + cfd_timesteps)*(cfd_times/cfd_timesteps)
	training_config.logger.info("###############################################")
	training_config.logger.info(f"CFD alone time: {if_CFD_alone}")
	training_config.logger.info(f"CFD+ML+update times: {cfd_times+ml_times+update_times}")
	training_config.logger.info(f"Acceleration: {if_CFD_alone/(ml_times + cfd_times + update_times)}")
	training_config.logger.info(f"ML timesteps per cross-computation: {ml_timesteps/switch_count}")
	training_config.logger.info(f"t_ML: {ml_times/ml_timesteps}")
	training_config.logger.info(f"t_CFD: {cfd_times/cfd_timesteps}")
	training_config.logger.info(f"t_update: {update_times/switch_count}")
	training_config.logger.info(f"Real Acceleration: {if_CFD_alone/(framework_end_time-framework_start_time)}\n\n")
	training_config.logger.info("###############################################")
	save_loss(training_config=training_config, merge_initial_losses=True)

if __name__ == "__main__":
	openfoam_config = OpenfoamConfig()
	training_config = TrainingConfig()

	# Run the main function
	hybrid_training(openfoam_config, training_config)

	# Start Prediction
	# trainer = Trainer(training_config=training_config,
	#                   model=FVMNetwork(training_config=training_config),
	#                   optimizer=training_config.optimizer,
	#                   loss=training_config.loss,
	#                   model_name="best_model.pth")
	# trainer.predict(prediction_start_time=10.02, write_interval=0.01)