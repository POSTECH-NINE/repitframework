import torch
from repitframework.config import TrainingConfig

class FVMNetwork(torch.nn.Module):
    def __init__(self, training_config:TrainingConfig, vars_list:list=None,
                 hidden_layers:int=9, hidden_size:int=512, activation:torch.nn.ReLU=None):
        '''
        Args
        ---- 
        training_config: TrainingConfig:
            configuration classes are set to give minimum arguments during initialization.
        vars_list: list
            list containing the variables to be predicted. If None, it will be taken from the training_config.
            e.g: ["U_x", "U_y", "T"]
        hidden_layers: int
            number of hidden layers in the network
        hidden_size: int 
            number of neurons in each hidden layer
        activation: 
            activation function to be used in the hidden
        '''
        super().__init__()
        self.num_dims = training_config.data_dim
        self.vars = training_config.get_variables() if vars_list is None else vars_list
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.activation = activation if activation is not None else training_config.activation
        
        # Create networks dynamically based on vars_list
        self.networks = torch.nn.ModuleDict(
            {
                f"{var}": self._build_network() for var in self.vars
            }
        )

    def forward(self,x:torch.Tensor):
        '''
        returns output for each variable as a tuple: 
        (ux_hat, uy_hat, t_hat)
        '''
        outputs = {var: net(x) for var, net in self.networks.items()}
        outputs_concat = torch.cat([output for output in outputs.values()], dim=1)
        return outputs

    def _build_network(self):
        """
        Build a single sub-network architecture.
        """
        input_shape = 15
        output_shape = 1
        layers = [torch.nn.Linear(input_shape, self.hidden_size), self.activation()]
        for _ in range(self.hidden_layers):
            layers.extend([torch.nn.Linear(self.hidden_size, self.hidden_size), self.activation()])
        layers.append(torch.nn.Linear(self.hidden_size, output_shape))
        return torch.nn.Sequential(*layers)
class ConvPhiNet(torch.nn.Module):
    def __init__(self):
        super(ConvPhiNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(2, 4, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(4)  

        self.conv2 = torch.nn.Conv2d(4, 2, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(2)

        self.conv3 = torch.nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(1)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = torch.nn.ReLU()

        # Fully connected layers
        self.fc = torch.nn.Linear(1 * 25 * 25, 4096)
        self.fc_bn = torch.nn.BatchNorm1d(4096)  # BatchNorm for fully connected layer
        self.fc2 = torch.nn.Linear(4096, 79600)

    def forward(self, x):
        x = self.relu(self.bn1(self.pool(self.conv1(x)))) # [b, 4, 100, 100]
        x = self.relu(self.bn2(self.pool(self.conv2(x)))) # [b, 2, 50, 50]
        x = self.relu(self.bn3(self.pool(self.conv3(x)))) # [b, 1, 25, 25]

        x = x.view(x.size(0), -1)  # Flatten [b, 1*25*25]
        x = self.relu(self.fc_bn(self.fc(x)))  # Apply BN to FC layer [b, 4096]
        x = self.fc2(x) # [b, 79600]
        return x

if __name__ == "__main__":
    # Test the network
    model = ConvPhiNet()
    input_tensor = torch.randn(1, 2, 200, 200)
    output = model(input_tensor)
    print(output.shape)