import torch
import torch.nn as nn
from .FNO2D import FNO2D
    
class FVFNO2D(nn.Module):
    def __init__(self,
                input_channels: int = 15,
                output_channels: int = 1,
                width: int = 18, 
                modes: tuple = (12, 12),
                depth: int = 3,
                activation: nn.Module = nn.functional.gelu,
                vars: list = ["T", "U_x", "U_y"],
                padding: int = 9,
                x_coords: torch.Tensor = None,
                y_coords: torch.Tensor = None,
                include_grid: bool = False
        ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.networks = torch.nn.ModuleDict({
            var: FNO2D(
                input_channels=self.input_channels,
                output_channels=self.output_channels,
                modes1=modes[0],modes2=modes[1],
                width=width,
                depth=depth,
                padding=padding,
                activation=activation,
                x_coords=x_coords,
                y_coords=y_coords,
                include_grid=include_grid
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
    B, C_in, H, W = 2, 15, 200, 200       # Batch size, input channels, height, width
    C_out = 1                         # For example, predicting a single scalar field
    width = 32                        # Hidden layer channels
    modes1, modes2 = 12, 12           # Use 12 low-frequency modes along each spatial dimension
    depth = 4                         # Number of Fourier layers

    # Create a random input tensor
    x = torch.randn(B, C_in, H, W)
    
    # Instantiate the FNO model
    model = FVFNO2D()
    
    # Forward pass
    y = model(x)
    print(f"Output shapes: {y['T'].shape}")
