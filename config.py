from dataclasses import dataclass
from pathlib import Path
from OpenFOAM.utils import read_mesh_type, read_solver_type, run_the_solver


@dataclass
class BaseConfig:
    root_dir:Path = Path(__file__).parent.resolve()
    dataloader_dir:Path = Path(root_dir, "DataLoader")
    logs_dir:Path = Path(root_dir, "logs")
    metrics_dir:Path = Path(root_dir, "metrics")
    modelselector_dir:Path = Path(root_dir, "ModelSelector")
    openfoam_dir:Path = Path(root_dir, "OpenFOAM")
    assets_dir:Path = Path(root_dir, "Assets")
class OpenfoamConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.solver_dir:str = Path(self.root_dir, "Solvers","natural_convection")
        self.case_name:str = None if self.solver_dir is None else self.solver_dir.name
        self.mesh_type:str = "blockMesh"
        self.solver_type:str = read_solver_type(solver_dir=self.solver_dir)
        self.data_vars: list = ["U", "p", "T"]

if __name__ == "__main__":
    print(OpenfoamConfig.solver_type, OpenfoamConfig().dataloader_dir)