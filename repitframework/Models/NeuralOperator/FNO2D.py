import torch
import torch.nn as nn

# 2D Spectral Convolution: performs FFT --> Multiply by learned weights on low modes --> Inverse FFT
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        Parameters:
            in_channels  : Number of input channels. [width]
            out_channels : Number of output channels. [width]
            modes1       : Number of low-frequency modes to keep along the height dimension.
            modes2       : Number of low-frequency modes to keep along the width dimension.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # low freq modes in height
        self.modes2 = modes2  # low freq modes in width

        # Initialize weights with a small scaling factor; using torch.cfloat dtype
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat) # [width, width, modes1, modes2]
        )

    def compl_mul2d(self, input, weights):
        # Performs batch-wise complex multiplication:
        # input: [B, C_in, H_ft, W_ft]
        # weights: [C_in, C_out, modes1, modes2]
        # Returns: [B, C_out, modes1, modes2]
        return torch.einsum("bchw,cohw->bohw", input, weights)

    def forward(self, x):
        """
        x: Input tensor of shape [B, C_in, H, W]
        """
        B, C, H, W = x.shape

        # Compute the 2D FFT (rfft2 returns output with shape [B, C, H, W//2 + 1])
        x_ft = torch.fft.rfft2(x, norm='ortho')

        # Prepare an output tensor in Fourier space with the same shape as x_ft
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=x_ft.dtype, device=x.device)

        # Multiply only the lower frequency modes
        # Assumes that H and W are large enough so that modes1 and modes2 are within bounds.
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights.type(x_ft.dtype)
        )

        # Return to spatial domain using the inverse FFT
        x_out = torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')
        return x_out

# Fourier layer for 2D inputs: combines the spectral convolution with a pointwise convolution
class FourierLayer2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, activation=nn.ReLU()):
        """
        Parameters:
            in_channels  : Number of channels entering the layer.
            out_channels : Number of channels leaving the layer.
            modes1, modes2 : Low-frequency modes to use along height and width.
            activation   : Activation function.
        """
        super().__init__()
        self.spectral_conv = SpectralConv2d(in_channels, out_channels, modes1, modes2)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = activation

    def forward(self, x):
        # Apply spectral convolution and pointwise convolution separately and add them
        x_spec = self.spectral_conv(x)
        x_point = self.pointwise_conv(x)
        return self.activation(x_spec + x_point)
    


# Overall FNO for 2D case: Lift input to a latent channel space, apply Fourier layers, then project back.
class FNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, width, modes1, modes2, depth=4, activation=nn.ReLU()):
        """
        Parameters:
            in_channels  : Number of channels in the input.
            out_channels : Number of channels in the output.
            width        : Number of channels in the hidden layers.
            modes1, modes2 : Number of low-frequency modes kept.
            depth        : Number of Fourier layers.
            activation   : Activation function.
        """
        super().__init__()
        # Lifting layer: Project input to a higher-dimensional feature space
        self.lift = nn.Linear(in_channels, width)
        
        # Stack of Fourier layers
        layers = []
        for _ in range(depth):
            layers.append(FourierLayer2d(width, width, modes1, modes2, activation))
        self.fourier_layers = nn.Sequential(*layers)
        
        # Projection layer: Convert hidden representation to the desired output channels
        self.projection = nn.Linear(width, out_channels)

    def linear_lift(self, x, lift_axis=1):
        # Move the lift axis to the end for FFT compatibility
        x = x.permute(0, 2, 3, 1)  # [B, H, W, in_channels]
        x = self.lift(x)            # [B, H, W, width]
        x = x.permute(0, 3, 1, 2)  # [B, width, H, W]
        return x
    
    def projection_layer(self, x):
        # Move the channel axis to the end for FFT compatibility
        x = x.permute(0, 2, 3, 1)
        x = self.projection(x)       # [B, H, W, out_channels]
        x = x.permute(0, 3, 1, 2)     # [B, out_channels, H, W]
        return x
    
    def forward(self, x):
        """
        x: Tensor of shape [B, in_channels, H, W]
        Returns: Tensor of shape [B, out_channels, H, W]
        """
        x = self.linear_lift(x)           # [B, width, H, W]
        x = self.fourier_layers(x) # Process through Fourier layers
        x = self.projection_layer(x)     # [B, out_channels, H, W]
        return x
    
class FVMNetwork(nn.Module):
    def __init__(self,
                in_channels: int = 15,
                out_channels: int = 1,
                width: int = 64, 
                modes: tuple = (12, 12),
                depth: int = 9,
                activation: nn.Module = nn.ReLU(),
                vars: list = ["T", "U_x", "U_y"]
        ):
        super().__init__()
        self.networks = torch.nn.ModuleDict({
            var: FNO2d(
                in_channels=in_channels,
                out_channels=out_channels,
                width=width,
                modes1=modes[0],
                modes2=modes[1],
                depth=depth,
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
    B, C_in, H, W = 2, 15, 200, 200       # Batch size, input channels, height, width
    C_out = 1                         # For example, predicting a single scalar field
    width = 32                        # Hidden layer channels
    modes1, modes2 = 12, 12           # Use 12 low-frequency modes along each spatial dimension
    depth = 4                         # Number of Fourier layers

    # Create a random input tensor
    x = torch.randn(B, C_in, H, W)
    
    # Instantiate the FNO model
    model = FVMNetwork()
    
    # Forward pass
    y = model(x)
    print(f"Output shapes: {y["T"].shape}")
