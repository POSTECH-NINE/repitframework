import torch
import torch.nn.functional as F
from typing import Tuple

def compute_gradient(tensor: torch.Tensor, grid_dx: float = 1.0, grid_dy: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assumes input shape [B, 1], flattened spatial grid of shape [B] where B = H*W"""
    B, _ = tensor.shape
    H = W = int(B ** 0.5)
    field = tensor.reshape(H, W)

    grad_y = (field[2:, 1:-1] - field[:-2, 1:-1]) / (2 * grid_dy)
    grad_x = (field[1:-1, 2:] - field[1:-1, :-2]) / (2 * grid_dx)

    # pad to original shape
    gx = F.pad(grad_x, (1,1,1,1), mode='constant', value=0)
    gy = F.pad(grad_y, (1,1,1,1), mode='constant', value=0)
    return gx.reshape(-1, 1), gy.reshape(-1, 1)

def ceod_loss(y_pred: torch.Tensor, y_true: torch.Tensor, operator_fn) -> torch.Tensor:
    op_pred_x, op_pred_y = operator_fn(y_pred)
    op_true_x, op_true_y = operator_fn(y_true)
    return torch.mean((op_pred_x - op_true_x) ** 2 + (op_pred_y - op_true_y) ** 2)
