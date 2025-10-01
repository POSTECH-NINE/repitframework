import torch
import numpy as np

# 2D Spectral Convolution: performs FFT --> Multiply by learned weights on low modes --> Inverse FFT
class SpectralConv2d(torch.nn.Module):
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
        self.weights1 = torch.nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = torch.nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))


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
        x_ft = torch.fft.rfft2(x, norm="ortho").to(torch.cfloat)

        # Prepare an output tensor in Fourier space with the same shape as x_ft
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=x_ft.dtype, device=x.device)

        # Multiply only the lower frequency modes
        # Assumes that H and W are large enough so that modes1 and modes2 are within bounds.
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        # Return to spatial domain using the inverse FFT
        x_out = torch.fft.irfft2(out_ft, s=(H, W)).to(x.dtype)
        return x_out

class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = torch.nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = torch.nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = torch.functional.F.gelu(x)
        x = self.mlp2(x)
        return x
    

class ResidualFNOBlock(torch.nn.Module):
    def __init__(self, width, modes1, modes2, activation=torch.nn.GELU(), dropout_p=0.2):
        super().__init__()
        self.norm = torch.nn.InstanceNorm2d(width, affine=False)
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.pointwise = torch.nn.Conv2d(width, width, 1)
        self.w = torch.nn.Conv2d(width, width, 1)     
        self.activation = activation
        self.dropout = torch.nn.Dropout2d(dropout_p) if dropout_p > 0 else torch.nn.Identity()

        # Optional learnable gate on the residual update (helps stabilization on long rollouts)
        self.beta = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # pre-norm -> spectral -> pointwise
        y = self.spectral(self.norm(x))
        y = self.pointwise(y)
        y = self.dropout(y)

        # block's internal skip path (W(x))
        y = y + self.w(x)

        # outer residual: x + beta * y
        out = x + self.beta * y
        return self.activation(out)

# Fourier layer for 2D inputs: combines the spectral convolution with a pointwise convolution
class FNO2D(torch.nn.Module):
    def __init__(self, 
                 input_channels=3, 
                 output_channels=3, 
                 modes1=12, modes2=12, 
                 width=32, 
                 depth=3,
                 padding=9, 
                 activation=torch.functional.F.gelu,
                 x_coords=None,
                 y_coords=None,
                 include_grid:bool=True):
        """
        Parameters:
            in_channels  : Number of channels entering the layer.
            out_channels : Number of channels leaving the layer.
            modes1, modes2 : Low-frequency modes to use along height and width.
            activation   : Activation function.
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.depth = depth
        self.activation = activation
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.padding = padding # pad if the input is not periodic.
        self.include_grid = include_grid

        if include_grid:
            self.p = torch.nn.Linear(self.input_channels+2, self.width) # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        else:
            self.p = torch.nn.Linear(self.input_channels, self.width)

        self.norm= torch.nn.InstanceNorm2d(self.width)
        self.spectral = torch.nn.ModuleList(
            [SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(depth)]
        )
        self.pointwise = torch.nn.ModuleList(
            [torch.nn.Conv2d(self.width, self.width, 1) for _ in range(depth)]
        )
        self.ws = torch.nn.ModuleList(
            [torch.nn.Conv2d(self.width, self.width, 1) for _ in range(depth)]
        )

        self.fno_blocks = torch.nn.ModuleList(
            [ResidualFNOBlock(self.width, self.modes1, self.modes2) for _ in range(self.depth)]
        )

        self.q = MLP(self.width, output_channels, self.width*self.depth)

    def forward(self, x:torch.Tensor):
        # Apply spectral convolution and pointwise convolution separately and add them
        if self.include_grid:
            grid = self.get_grid(x.shape, x.device)
            x = torch.cat((x, grid), dim=1)  # Concatenate grid coordinates to input
        x = x.permute(0, 2, 3, 1)  # Change to [B, H, W, C] for lifting.
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        x = torch.nn.functional.pad(x, (0, self.padding, 0, self.padding))  # Reflect padding to handle edge cases

        # for i in range(self.depth):
        #     x1 = self.norm(self.spectral[i](self.norm(x)))
        #     x1 = self.pointwise[i](x1)
        #     x2 = self.ws[i](x)
        #     x = x1 + x2
        #     if i < self.depth - 1:
        #         x = self.activation(x)
        for block in self.fno_blocks:
            x = block(x)

        x = x[..., :-self.padding, :-self.padding]  # Remove padding after processing
        x = self.q(x)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[-2], shape[-1]
        if self.x_coords is None or self.y_coords is None:
            gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
            gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        else:
            gridx = torch.tensor(self.x_coords, dtype=torch.float)
            gridy = torch.tensor(self.y_coords, dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = gridy.reshape(1, 1, 1,size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)

# Example usage
if __name__ == "__main__":
    # Example configuration
    B,Ch, H, W = 10, 1, 200, 200       # Batch size, input channels, height, width
    C_out = 1                         # For example, predicting a single scalar field
    width = 32                        # Hidden layer channels
    modes1, modes2 = 12, 12           # Use 12 low-frequency modes along each spatial dimension
    depth = 4                         # Number of Fourier layers

    # Create a random input tensor
    x = torch.randn(B,Ch, H, W)
    
    # Instantiate the FNO model
    model = FNO2D()
    
    # Forward pass
    y = model(x)
    print(f"Output shape: {y.shape}")
