import torch
from pathlib import Path

def load_from_state_dict(
        model: torch.nn.Module,
        model_save_path: str|Path,
        model_name: str, 
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler=None,
        learning_rate: float = 1e-3
):
    """
    Load a model and optimizer state from a checkpoint file.

    Args:
        model (torch.nn.Module): The model to load the state into.
        model_save_path (str|Path): Path to the directory containing the checkpoint.
        model_name (str): Name of the checkpoint file.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.

    Returns:
        None
    """
    checkpoint = torch.load(Path(model_save_path, model_name), weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    is_optimizer_loaded = "False"
    # if 'optimizer_state_dict' in checkpoint:
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = learning_rate
    # if 'scheduler_state_dict' in checkpoint:
    #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"Model loaded from: {model_save_path/model_name} with LR: {learning_rate}")
    return model, optimizer