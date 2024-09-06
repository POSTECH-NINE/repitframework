import torch

class FVMN(torch.nn.Module):
    def __init__(self, input_size:int=1, hidden_size:int=64, output_size:int=1, num_layers:int=4):
        super(FVMN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = torch.nn.ReLU()
        

    def forward(self, x):
        pass