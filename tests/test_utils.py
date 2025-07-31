from repitframework.config import TrainingConfig
from repitframework.Models import FVMNetwork
from repitframework.utils import optimize_required_grads_only, Timer
import torch
	
with Timer() as timer:
    training_config = TrainingConfig()
    model = FVMNetwork(vars_list=["U_x", "U_y", "T"],
                    hidden_layers=3, hidden_size=398, 
                    activation=torch.nn.ReLU, dropout=0.2)
    print(optimize_required_grads_only(model, training_config=training_config))
print(f"Time taken: {timer.elapsed}")
print(f"Timer (seconds): {timer.elapsed.total_seconds()}")