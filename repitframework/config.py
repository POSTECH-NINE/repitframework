from dataclasses import dataclass
from pathlib import Path
import torch.cuda as cuda

@dataclass
class BaseConfig:
    # Directories:
    root_dir:Path = Path(__file__).parent.resolve()
    dataloader_dir:Path = Path(root_dir, "DataLoader")
    logs_dir:Path = Path(root_dir, "logs")
    metrics_dir:Path = Path(root_dir, "metrics")
    modelselector_dir:Path = Path(root_dir, "ModelSelector")
    openfoam_dir:Path = Path(root_dir, "OpenFOAM")
    assets_dir:Path = Path(root_dir, "Assets", "natural_convection")
    solver_dir:Path = Path(root_dir, "Solvers","natural_convection")

    # Logging Level
    logger_level:str = "DEBUG"

    # Data parameters
    time_step:int = 0.01
    start_time = 10
    end_time = 20
    data_dim:int = 2 # It is to denote either data is 1D or 2D or 3D.
    grid_x: int = 200 # Number of grid points in x-direction
    grid_y: int = 200 # Number of grid points in y-direction
    grid_z: int = 1 # Number of grid points in z-direction



class OpenfoamConfig(BaseConfig):
    def __init__(self):
        super(OpenfoamConfig, self).__init__()
        self.case_name:str = None if self.solver_dir is None else self.solver_dir.name
        self.mesh_type:str = "blockMesh"
        self.solver_type:str = "buoyantFoam"
        self.data_vars: list = ["U", "T"] # TODO: define this in base class: see for how to include list in dataclass.


class TrainingConfig(BaseConfig):
    def __init__(self):
        super(TrainingConfig, self).__init__()
        self.batch_size: int = 16
        self.epochs: int = 5000
        self.learning_rate: float = 0.0001
        self.device: str = "cuda" if cuda.is_available() else "cpu"
        self.optimizer: str = "adam"
        self.loss: str = "mse"
        self.activation: str = "relu"

if __name__ == "__main__":
    print(OpenfoamConfig().solver_type, OpenfoamConfig().dataloader_dir,OpenfoamConfig().root_dir)