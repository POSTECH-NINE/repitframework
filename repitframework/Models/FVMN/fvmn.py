import torch
from repitframework.config import TrainingConfig

class FVMNetwork(torch.nn.Module):
    def __init__(self, training_config:TrainingConfig, vars_list:list=None,
                 hidden_layers:int=3, hidden_size:int=398, activation:torch.nn.ReLU=None, dropout=0.2):
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
        self.dropout_rate = dropout # Store the rate, not the module directly
        
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
        # outputs_concat = torch.cat([output for output in outputs.values()], dim=1)
        return outputs

    def _build_network(self):
        """
        Build a single sub-network architecture.
        """
        input_shape = 15
        output_shape = 1
        
        layers = []
        
        # Input Layer
        layers.append(torch.nn.Linear(input_shape, self.hidden_size))
        layers.append(torch.nn.BatchNorm1d(self.hidden_size)) # BN before Activation
        layers.append(self.activation()) # Activation after BN

        # Hidden Layers
        dropout_layers = [self.hidden_layers-1] # Only add dropout to the last hidden layer
        for i in range(self.hidden_layers):
            layers.append(torch.nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(torch.nn.BatchNorm1d(self.hidden_size)) # BN before Activation
            layers.append(self.activation()) # Activation after BN
            
            if self.dropout_rate is not None and i in dropout_layers and self.dropout_rate > 0: # Only add if dropout is enabled
                layers.append(torch.nn.Dropout(self.dropout_rate)) # Dropout after Activation (and BN)

        # Output Layer
        layers.append(torch.nn.Linear(self.hidden_size, output_shape))
        
        return torch.nn.Sequential(*layers)

if __name__ == "__main__":
    # Test the network
    model = FVMNetwork(TrainingConfig())
    
    dummy_input = torch.randn(10,15)
    output = model(dummy_input)
    print(output["T"].shape)