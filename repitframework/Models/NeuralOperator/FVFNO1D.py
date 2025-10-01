import torch
import torch.nn as nn
from .FNO1D import FNO1D

class FVFNO1D(nn.Module):
    def __init__(self,
                input_channels: int = 15,
                output_channels: int = 1,
                width: int = 64, 
                modes: tuple = 12,
                depth: int = 3,
                activation: nn.Module = nn.GELU(),
                vars: list = ["T", "U_x", "U_y"],
                padding: int = 9
        ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.networks = torch.nn.ModuleDict({
            var: FNO1D(
                input_channels=self.input_channels,
                output_channels=self.output_channels,
                modes=modes,
                width=width,
                depth=depth,
                padding=padding,
                activation=activation
            )
            for var in vars
        })
    
    def forward(self, x):
        """
        x: Tensor of shape [B, C_in, H, W]
        Returns: Dictionary of tensors for each variable
        """
        outputs = {}
        for var, net in self.networks.items():
            outputs[var] = net(x)
        return outputs

# Example usage
if __name__ == "__main__":
    # Example configuration
    n = 10000         # N = B*H*W
    input_channels = 15    # treated as sequence length L
    output_channels = 3

    # Input shaped exactly as requested: (B*H*W, C)
    x = torch.randn(n, input_channels)
    
    # Instantiate the FNO model
    model = FVFNO1D()
    
    # Forward pass
    y = model(x)
    print(f"Output shapes: {y['U_x'].shape}")
