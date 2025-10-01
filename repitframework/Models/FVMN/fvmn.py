import torch


class ResLinear(torch.nn.Module):
    """
    A Linear layer with a residual connection.

    If the input and output dimensions of the layer are the same, the forward pass
    will compute `F(x) + x`, where `F(x)` is the standard linear transformation.
    This creates a residual block.

    If the input and output dimensions are different, it behaves exactly like a
    standard `torch.nn.Linear` layer, as the residual connection cannot be added.
    This is useful for changing dimensions at the input or output of a network.
    """
    def __init__(self, activation, in_features, out_features):
        super().__init__()
        self.activation = activation
        self.in_features = in_features
        self.out_features = out_features
        self.batchnorm1 = torch.nn.BatchNorm1d(self.in_features)
        # self.batchnorm2 = torch.nn.BatchNorm1d(self.out_features)
        self.linear1 = torch.nn.Linear(self.in_features, self.out_features)
        # self.linear2 = torch.nn.Linear(self.out_features, self.out_features)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Perform the standard linear transformation

        linear_output = self.batchnorm1(x)
        linear_output = self.activation(linear_output)
        linear_output = self.linear1(linear_output)
        # linear_output = self.batchnorm2(linear_output)
        # linear_output = self.activation(linear_output)
        # linear_output = self.linear2(linear_output)

        if self.in_features == self.out_features:
            return x + linear_output
        return linear_output

class FVMNetwork(torch.nn.Module):
    def __init__(self, vars:list=["U_x", "U_y", "T"],
                 hidden_layers:int=3, 
                 hidden_size:int=398, 
                 activation:torch.nn.ReLU=torch.nn.ReLU(), 
                 dropout=0.2,
                 input_channels:int=15):
        '''
        Args
        ---- 
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
        self.vars = vars
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.input_channels = input_channels
        self.activation = activation
        self.dropout_rate = dropout  # Store the rate, not the module directly

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
        input_channels = self.input_channels
        output_shape = 1
        
        layers = []
        
        # Input Layer
        layers.append(torch.nn.Linear(input_channels, self.hidden_size))
        layers.append(torch.nn.BatchNorm1d(self.hidden_size)) # BN before Activation
        layers.append(self.activation) # Activation after BN

        # Hidden Layers
        dropout_layers = [self.hidden_layers-1] # Only add dropout to the last hidden layer
        for i in range(self.hidden_layers):
            layers.append(torch.nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(torch.nn.BatchNorm1d(self.hidden_size)) # BN before Activation
            layers.append(self.activation) # Activation after BN
            
            if self.dropout_rate is not None and i in dropout_layers and self.dropout_rate > 0: # Only add if dropout is enabled
                layers.append(torch.nn.Dropout(self.dropout_rate)) # Dropout after Activation (and BN)

        # Output Layer
        layers.append(torch.nn.Linear(self.hidden_size, output_shape))
        
        return torch.nn.Sequential(*layers)
    
    def _build_res_network(self):
        """
        Build a single residual sub-network architecture.
        """
        input_shape = self.input_channels
        output_shape = 1
        
        layers = []
        
        # Input Layer
        layers.append(torch.nn.Linear(input_shape, self.hidden_size))

        # Hidden Layers
        dropout_layers = [self.hidden_layers-1] # Only add dropout to the last hidden layer
        for i in range(self.hidden_layers):
            layers.append(ResLinear(self.activation, self.hidden_size, self.hidden_size))
            if self.dropout_rate is not None and i in dropout_layers and self.dropout_rate > 0: # Only add if dropout is enabled
                layers.append(torch.nn.Dropout(self.dropout_rate)) # Dropout after Activation (and BN)

        layers.append(torch.nn.LayerNorm(self.hidden_size)) # BN before Activation
        layers.append(self.activation) # Activation after BN
        # Output Layer
        layers.append(torch.nn.Linear(self.hidden_size, output_shape))
        
        return torch.nn.Sequential(*layers)