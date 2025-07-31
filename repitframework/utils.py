from typing import Tuple
import torch
import numpy as np
from pathlib import Path
from contextlib import ContextDecorator
from datetime import datetime

from .model_selector import OptimizerSelector
from .config import TrainingConfig

def load_from_state_dict(
		model: torch.nn.Module,
		model_save_path: str|Path,
		model_name: str, 
		optimizer:torch.optim.Optimizer=None,
		scheduler:torch.optim.lr_scheduler._LRScheduler=None,
		learning_rate: float = 1e-3
)-> Tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
	"""
	Load a model, optimizer, and scheduler state from a checkpoint file.
	"""

	checkpoint = torch.load(Path(model_save_path, model_name), weights_only=True)
	model.load_state_dict(checkpoint['model_state_dict'])
	
	if 'optimizer_state_dict' in checkpoint and optimizer is not None:
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		for param_group in optimizer.param_groups:
			param_group['lr'] = learning_rate
	if 'scheduler_state_dict' in checkpoint and scheduler is not None:
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
	print(f"\nModel loaded from: {model_save_path/model_name} with LR: {learning_rate}")
	return model, optimizer, scheduler

def save_to_state_dict(
		model: torch.nn.Module,
		model_save_path: str|Path,
		model_name: str,
		optimizer:torch.optim.Optimizer=None,
		scheduler:torch.optim.lr_scheduler._LRScheduler=None
	)-> Path:
	"""
	Save a model, optimizer, and scheduler state to a checkpoint file.
	"""
	checkpoint = {
		'model_state_dict': model.state_dict(),
	}
	if optimizer is not None:
		checkpoint['optimizer_state_dict'] = optimizer.state_dict()
	if scheduler is not None:
		checkpoint['scheduler_state_dict'] = scheduler.state_dict()
	Path(model_save_path).mkdir(parents=True, exist_ok=True)
	torch.save(checkpoint, Path(model_save_path, model_name))
	return Path(model_save_path, model_name)


def freeze_layers(model:torch.nn.Module, num_layers:int,layer_types=(torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)) -> None:
	'''
	Freeze the layers of a network.
	'''
	count = 0
	layers = list(model.children())
	for layer in layers:
		if isinstance(layer, layer_types):
			for param in layer.parameters():
				param.requires_grad = False
			count += 1
		if count >= num_layers:
			break
				
def optimize_required_grads_only(model: torch.nn.Module,
								training_config: TrainingConfig)-> Tuple[torch.nn.Module, torch.optim.Optimizer]:
	
	if hasattr(model, "networks") and isinstance(model.networks,torch.nn.ModuleDict):
		for network in model.networks.values():
			freeze_layers(network, training_config.layers_to_freeze)
			params = (p for net in model.networks.values() for p in net.parameters() if p.requires_grad)
	else:
		freeze_layers(model, training_config.layers_to_freeze)
		params = (p for p in model.parameters() if p.requires_grad)

	optimizer = OptimizerSelector(
		training_config.optimizer_type,
		params,
		training_config.optim_kwargs
	)
	return model, optimizer

class Timer(ContextDecorator):
    def __init__(self):
        self.start = None
        self.end = None
        self.elapsed = None

    def __enter__(self):
        self.start = datetime.now()
        return self

    def __exit__(self, *exc):
        self.end = datetime.now()
        self.elapsed = self.end - self.start
