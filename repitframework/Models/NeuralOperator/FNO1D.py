import torch
import torch.nn as nn
from einops import rearrange

# Complex multiplication
def compl_mul(input, weights):
    # (batch, in_channel, ,,,), (in_channel, out_channel, ...) -> (batch, out_channel, ...)
    return torch.einsum("bi...,io...->bo...", input, weights)


def get_scaled_uniform(in_channels, out_channels, modes):
    scale = 1 / (in_channels * out_channels)
    return scale * torch.rand(in_channels, out_channels, *modes, dtype=torch.cfloat)


################################################################
#  1,2,3d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super().__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            get_scaled_uniform(in_channels, out_channels, [self.modes1])
        )
        self.dim = 1

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=tuple(range(-1, -self.dim - 1, -1)))

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            dtype=x_ft.dtype,
            device=x.device,
        )
        out_ft[:, :, : self.modes1] = compl_mul(
            x_ft[:, :, : self.modes1], self.weights1.type(x_ft.dtype)
        )

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=x.shape[2:], dim=-1)
        return x


class Fourier_layer(nn.Module):
    def __init__(self, in_channels, out_channels, modes, activation) -> None:
        super().__init__()
        self.conv0 = SpectralConv1d(in_channels, out_channels, *modes)
        self.w0 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.activation = activation

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.w0(rearrange(x, "b c ... -> b c (...)")).view(x1.shape)
        x = x1 + x2
        return self.activation(x)


class FNO1D(nn.Module):
    def __init__(self, 
                 input_channels=3, 
                 output_channels=3, 
                 modes=12, 
                 width=64, 
                 depth=3,
                 padding=9,
                 activation=nn.GELU()):
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0.
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv.
        3. Project from the channel space to the output space by self.fc1.

        input: the discretized of function f(x)
        input shape: (batchsize, input_shape)
        output: the predicted solution
        output shape: (batchsize, input_shape)
        """
        super().__init__()
        self.padding = padding
        self.fc0 = nn.Linear(1, width)

        self.fno = nn.Sequential(
            *[Fourier_layer(width, width, [modes], activation) for _ in range(depth)]
        )

        self.fc1 = nn.Sequential(
            nn.Linear(width * input_channels, width), activation, nn.Linear(width, output_channels)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)

        x = self.fc0(x)
        x = rearrange(x, "b ... c -> b c ...")
        x = nn.functional.pad(x, [0, self.padding], mode="reflect")  # pad the domain if input is non-periodic

        x = self.fno(x)
        x = x[..., :-self.padding]
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        return x
    

# Example usage
if __name__ == "__main__":
    # Example configuration
    B,C = 80000, 15       # Batch size, input channels, height, width
    C_out = 3                         # For example, predicting a single scalar field
    width = 64                       # Hidden layer channels
    modes1, modes2 = 12, 12           # Use 12 low-frequency modes along each spatial dimension
    depth = 3                         # Number of Fourier layers

    # Create a random input tensor
    x = torch.randn(B,C)

    # Instantiate the FNO model
    model = FNO1D(input_channels=15)
    
    # Forward pass
    y = model(x)
    print(f"Output shape: {y.shape}")