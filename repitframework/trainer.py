"""RePIT Framework: Hybrid CFD-ML Simulation Framework with Transfer Learning:
1. Model with best validation loss is saved as "best_model_{model_type}.pth".
2. metrics have special naming convention (in norm_denorm_metrics.json):
   - "label_mean", "label_std", "input_mean", "input_std", "true_residual_mass"
3. To load optimizer and scheduler from checkpoint, override "__init__" method and set `load_optimizer=True` and `load_scheduler=True`.
4. To change the loss logic, architectural nuances in loss calculation, override `process_one_batch` method.
"""

from __future__ import annotations
from typing import Tuple, Union

import torch
from torch.utils.data import DataLoader

from .config import NaturalConvectionConfig
from .model_selector import ModelSelector, OptimizerSelector, SchedulerSelector
from .utils import load_from_state_dict, save_to_state_dict, optimize_required_grads_only


class BaseHybridTrainer:
	def __init__(self, training_config:NaturalConvectionConfig,
				 saved_model_name:str=None,
				 load_optimizer:bool=False,
				 load_scheduler:bool=False):
		self.training_config = training_config
		self.device = training_config.device
		self.load_optimizer = load_optimizer
		self.load_scheduler = load_scheduler

		self.model = ModelSelector(
			training_config.model_type,
			training_config.model_kwargs
		)
		if saved_model_name:
			self.model, self.optimizer, self.scheduler = self.from_checkpoint(saved_model_name)
			self.training_config.epochs = 0  # Set epochs to 0 if loading from checkpoint
		else:
			self.optimizer = self.optimizer_selection()
			self.scheduler = self.scheduler_selection(self.optimizer)
			
		self.model.to(self.device)
		self.best_validation_loss = float('inf')
		self.variables = self.training_config.extend_variables()                                      


	def from_checkpoint(self, saved_model_name:str) -> Tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
		
		optim = self.optimizer_selection()
		scheduler = None
		if self.load_optimizer and self.load_scheduler:
			scheduler = self.scheduler_selection(optim)
			return load_from_state_dict(
				self.model,
				self.training_config.model_dump_dir,
				saved_model_name,
				self.optimizer,
				self.scheduler
			)
		elif self.load_optimizer:
			model, optim, *_ = load_from_state_dict(
				self.model,
				self.training_config.model_dump_dir,
				saved_model_name,
				self.optimizer,
				scheduler=scheduler
			)
			return model, optim, scheduler

		model, *_ = load_from_state_dict(
			self.model,
			self.training_config.model_dump_dir,
			saved_model_name
		)
		
		return model, optim, scheduler
	
	def optimizer_selection(self,) -> torch.optim.Optimizer:
		"""
		Get the optimizer used for training.
		Returns:
			torch.optim.Optimizer: The optimizer instance.
		"""
		return OptimizerSelector(
			self.training_config.optimizer_type,
			self.model.parameters(),
			self.training_config.optim_kwargs
		)
	
	def scheduler_selection(self, optim_instance:torch.optim.Optimizer) -> Union[torch.optim.lr_scheduler._LRScheduler, None]:
		"""
		Get the learning rate scheduler used for training.
		Returns:
			torch.optim.lr_scheduler._LRScheduler: The scheduler instance.
		"""
		if self.training_config.scheduler_type is None:
			return None
		
		return SchedulerSelector(
			scheduler_name=self.training_config.scheduler_type,
			optimizer=optim_instance
		)
	
	def _process_module_dict(self, predictions:torch.nn.ModuleDict, labels:torch.Tensor) -> float:
		
		loss = 0.0
		for var, pred in predictions.items():
			var_index = self.variables.index(var)
			label = labels[:, var_index:var_index + 1]  # Assuming labels are always present on dimension 1.
			loss += self.training_config.loss(pred, label)
		return loss

	def process_one_batch(self, inputs:torch.Tensor, labels:torch.Tensor) -> float:
		"""
		Train the model for one batch of data.
		Args:
			inputs (torch.Tensor): Input data.
			labels (torch.Tensor): Ground truth labels.
		Returns:
			Tuple[float, float]: Average loss and average residual for the batch.
		"""
		inputs, labels = inputs.to(self.device), labels.to(self.device)

		predictions = self.model(inputs)

		# Multiple models can be used, so we need to handle ModuleDict.
		if hasattr(self.model, "networks") and isinstance(self.model.networks, torch.nn.ModuleDict):
			loss = self._process_module_dict(predictions, labels)
		else:
			loss = self.training_config.loss(predictions, labels)

		if self.model.training: # returns True if model is in training mode.
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

		return loss.item()

		
	def process_one_epoch(self, dataloader:DataLoader) -> float:
		loss = 0.0
		num_batches = len(dataloader)

		for batch in dataloader:
			inputs, labels = batch
			loss += self.process_one_batch(inputs, labels)
		loss /= num_batches

		return loss

	def fit(self, 
		   train_loader:DataLoader, 
		   val_loader:DataLoader,
		   epochs:int=None,
		   freeze_layers:bool=True,):
		
		if freeze_layers:
			self.model, self.optimizer = optimize_required_grads_only(
				self.model,
				self.training_config
			)
		if epochs is None:
			epochs = self.training_config.epochs
		
		print("Training started...")
		for epoch in range(epochs):
			training_loss = self.train(train_loader)
			validation_loss = self.validate(val_loader)

			print(f"\rEpoch: {epoch}/{epochs}| Training Loss: {training_loss} | Validation Loss: {validation_loss}", end='', flush=True)
			# At the end of the epoch, after all batches are processed:
			if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
				self.scheduler.step(validation_loss) # Requires a metric
			elif self.scheduler:
				self.scheduler.step() # Other schedulers just step without a metric
			self.training_config.log_metrics(key="Epoch", value=epoch+1, metrics_type="training")
			self.training_config.log_metrics(key="Training Loss", value=training_loss, metrics_type="training")
			self.training_config.log_metrics(key="Validation Loss", value=validation_loss, metrics_type="training")
			

	def train(self, train_loader:DataLoader):
		self.model.train()
		train_loss = self.process_one_epoch(dataloader=train_loader)
		self.training_config.logger.debug(f"Training Loss: {train_loss:.4f}")
		return train_loss
	

	def validate(self, val_loader:DataLoader):
		self.model.eval()
		with torch.no_grad():
			val_loss = self.process_one_epoch(dataloader=val_loader)
		self.training_config.logger.debug(f"Validation Loss: {val_loss:.4f}")

		if val_loss < self.best_validation_loss:
			self.best_validation_loss = val_loss
			model_name = f"best_model_{self.training_config.model_type}.pth"
			save_to_state_dict(
				self.model,
				self.training_config.model_dump_dir,
				model_name,
				self.optimizer,
				self.scheduler
			)
			self.training_config.logger.info(f"Model saved with validation loss: {val_loss:.4f}")

		return val_loss