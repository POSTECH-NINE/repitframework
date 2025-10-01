from __future__ import annotations
from typing import Tuple, Optional
from pathlib import Path
from contextlib import ContextDecorator
from datetime import datetime

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .model_selector import OptimizerSelector
from .config import TrainingConfig




# ---- helpers -----------------------------------------------------------------

_PREFIXES_TO_STRIP = ("_orig_mod.", "module.")

def _unwrap_model(m: Module) -> Module:
    """Return the underlying nn.Module if model was torch.compile()'d (or plain)."""
    # In PyTorch 2.x, compiled modules expose _orig_mod
    return getattr(m, "_orig_mod", m)

def _strip_prefixes(key: str) -> str:
    for p in _PREFIXES_TO_STRIP:
        if key.startswith(p):
            return key[len(p):]
    return key

def _remap_state_dict_to_model(ckpt_sd: dict, model: Module) -> dict:
    """
    Build a new state_dict whose keys match `model.state_dict().keys()`,
    by matching after stripping known wrapper prefixes on both sides.
    """
    target_sd = _unwrap_model(model).state_dict()

    # Canonicalize keys (strip wrappers) for matching
    canon_ckpt = { _strip_prefixes(k): v for k, v in ckpt_sd.items() }
    canon_model_keys = { _strip_prefixes(k): k for k in target_sd.keys() }

    new_sd = {}
    missing = []
    for canon_key, model_key in canon_model_keys.items():
        v = canon_ckpt.get(canon_key, None)
        if v is None:
            missing.append(model_key)
        else:
            new_sd[model_key] = v

    # Keep any unexpected (extra) tensors in ckpt (harmless with strict=False)
    return new_sd


# ---- API ---------------------------------------------------------------------

def load_from_state_dict(
    model: Module,
    model_save_path: str | Path,
    model_name: str,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    learning_rate: float = 1e-3,
) -> Tuple[Module, Optional[Optimizer], Optional[_LRScheduler]]:
    """
    Load a model, optimizer, and scheduler state from a checkpoint file.
    Works for plain, DataParallel/DDP-wrapped, and torch.compile()'d models.
    """
    ckpt_path = Path(model_save_path) / model_name
    checkpoint = torch.load(ckpt_path, weights_only=True, map_location="cpu")

    # 1) Model weights
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"'model_state_dict' not found in checkpoint: {ckpt_path}")

    # Try direct load first; if it fails, do prefix-agnostic remap.
    target_model = _unwrap_model(model)
    try:
        target_model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError:
        remapped = _remap_state_dict_to_model(checkpoint["model_state_dict"], model)
        target_model.load_state_dict(remapped, strict=False)

    # 2) Optimizer (optional)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Ensure desired LR wins (useful after resume)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        # Make sure optimizer state tensors live on the right device
        # (PyTorch usually fixes this on first step, but we can proactively move)
        model_device = next(_unwrap_model(model).parameters()).device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(model_device)

    # 3) Scheduler (optional)
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"\nModel loaded from: {ckpt_path} with LR: {learning_rate}")
    return model, optimizer, scheduler


def save_to_state_dict(
    model: Module,
    model_save_path: str | Path,
    model_name: str,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
) -> Path:
    """
    Save a model, optimizer, and scheduler state to a checkpoint file.
    If the model is torch.compile()'d, saves the original module weights (no '_orig_mod.' prefix).
    """
    model_save_path = Path(model_save_path)
    model_save_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_save_path / model_name

    base_model = _unwrap_model(model)  # <— avoid saving with '_orig_mod.' prefix
    checkpoint = {
        "model_state_dict": base_model.state_dict(),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, ckpt_path)
    return ckpt_path



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
	else:
		freeze_layers(model, training_config.layers_to_freeze)
	params = filter(lambda p: p.requires_grad, model.parameters())
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
