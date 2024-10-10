from OpenFOAM.utils import run_the_solver
from config import OpenfoamConfig

if __name__ == "__main__":
    openfoam_config = OpenfoamConfig()
    _ = run_the_solver(solver_dir=openfoam_config.solver_dir, assets_dir=openfoam_config.assets_dir)
    