
import warnings
import inspect

import torch

from .Models.FVMN import FVMNetwork
from .Models.NeuralOperator import FNO2D, FVFNO2D, FNO1D, FVFNO1D


class ModelSelector:
    available_models = {
        "fvmn": FVMNetwork,
        "fno2d": FNO2D,
        "fvfno2d": FVFNO2D,
        "fno1d": FNO1D,
        "fvfno1d": FVFNO1D
    }

    def __new__(cls, model_type: str, model_kwargs: dict = None)->torch.nn.Module:
        if model_type not in cls.available_models:
            raise ValueError(
                f"Model '{model_type}' is not available. Choose from {list(cls.available_models.keys())}."
            )
        model_class = cls.available_models[model_type]
        if not model_kwargs:
            warnings.warn(
                "No model_kwargs provided. Using default parameters for the model."
            )
        valid_args = set(inspect.signature(model_class.__init__).parameters.keys()) - {"self"}
        # Filter model_kwargs to only include valid arguments
        filtered_kwargs = {k: v for k, v in model_kwargs.items() if k in valid_args}
        return model_class(**filtered_kwargs) if filtered_kwargs else model_class()

class OptimizerSelector:
    available_optimizers = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
        "adagrad": torch.optim.Adagrad,
        "adamax": torch.optim.Adamax
    }

    def __new__(cls, optimizer_type: str, model_params: torch.nn.Parameter, optim_kwargs: dict=None)->torch.optim.Optimizer:
        if optimizer_type not in cls.available_optimizers:
            raise ValueError(
                f"Optimizer '{optimizer_type}' is not available. Choose from {list(cls.available_optimizers.keys())}."
            )
        optimizer_class = cls.available_optimizers[optimizer_type]

        # Get the argument names for the optimizer (excluding 'self' and 'params')
        valid_args = set(inspect.signature(optimizer_class.__init__).parameters.keys()) - {"self", "params"}
        # Filter kwargs to only include valid arguments
        filtered_kwargs = {k: v for k, v in optim_kwargs.items() if k in valid_args}
        
        return optimizer_class(model_params, **filtered_kwargs)

class SchedulerSelector:
    available_schedulers = {
    "steplr": (torch.optim.lr_scheduler.StepLR, {"step_size": 10, "gamma": 0.1}),
    "multisteplr": (torch.optim.lr_scheduler.MultiStepLR, {"milestones": [30, 80], "gamma": 0.1}),
    "exponentiallr": (torch.optim.lr_scheduler.ExponentialLR, {"gamma": 0.95}),
    "reducelronplateau": (torch.optim.lr_scheduler.ReduceLROnPlateau, {"mode": "min", "factor": 0.1, "patience": 10}),
    "cosineannealinglr": (torch.optim.lr_scheduler.CosineAnnealingLR, {"T_max": 50}),
    "cycliclr": (torch.optim.lr_scheduler.CyclicLR, {"base_lr": 1e-4, "max_lr": 1e-2, "step_size_up": 10, "mode": "triangular"}),
    }

    def __new__(cls, scheduler_type: str, optimizer: torch.optim.Optimizer)->torch.optim.lr_scheduler._LRScheduler:
        if scheduler_type not in cls.available_schedulers:
            raise ValueError(
                f"Scheduler '{scheduler_type}' is not available. Choose from {list(cls.available_schedulers.keys())}."
            )
        scheduler_class, default_kwargs = cls.available_schedulers[scheduler_type]
        return scheduler_class(optimizer, **default_kwargs)
    
def test_selectors():
    # Test ModelSelector
    model = ModelSelector("fvmn", {"vars_list": ["T", "U"], "hidden_layers": 3, "hidden_size": 398})
    assert isinstance(model, FVMNetwork), "ModelSelector did not return the correct model type."

    # Test OptimizerSelector
    optimizer = OptimizerSelector("adam", model.parameters(), optim_kwargs={"lr":0.001})
    assert isinstance(optimizer, torch.optim.Adam), "OptimizerSelector did not return the correct optimizer type."

    # Test SchedulerSelector
    scheduler = SchedulerSelector("steplr", optimizer)
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR), "SchedulerSelector did not return the correct scheduler type."


if __name__ == "__main__":
    from repitframework.config import TrainingConfig

    training_config = TrainingConfig()
    # Run the test function to validate the selectors
    test_selectors()
    print("All selectors are working correctly.")
    
    # Example usage of ModelSelector, OptimizerSelector, and SchedulerSelector
    model = ModelSelector("fvmn", {"vars_list": ["T", "U"], "hidden_layers": 3, "hidden_size": 398})
    optimizer = OptimizerSelector("adam", model.parameters(), training_config.optim_kwargs)
    scheduler = SchedulerSelector("steplr", optimizer)
    
    print(f"Model: {model}, Optimizer: {optimizer}, Scheduler: {scheduler}")