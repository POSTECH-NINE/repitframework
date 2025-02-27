#!/bin/bash
import os
import sys

from config import BaseConfig

def foamRemoveTimes(solver_dir:str)->None:
    """
    Removes all the time directories from the solver directory.
    
    Args
    ----
    solver_dir: Path
        The path to the solver directory.
    """
    directories_to_preserve = ["0", "constant", "system","10"]
    for file in os.scandir(solver_dir):
        if file.is_dir() and file.name not in directories_to_preserve:
            os.system(f"rm -r {file.path}")
    print("Removed all the time directories.")

def cleanAssets(assets_dir:str)->None:
    """
    Cleans the solver directory by removing unnecessary files.
    
    Args
    ----
    solver_dir: Path
        The path to the solver directory.
    """
    assets_dir = assets_dir.replace("/natural_convection","")
    os.system(f"rm -r {assets_dir}/natural_convection")
    os.system(f"cp -r {assets_dir}/natural_convection_backup {assets_dir}/natural_convection")
    print("Cleaned the assets directory.")

def cleanMetrics(metrics_dir:str)->None:
    """
    Cleans the metrics directory by removing unnecessary files.
    
    Args
    ----
    metrics_dir: Path
        The path to the metrics directory.
    """
    files_to_clean = ["prediction_metrics.json", "training_metrics.json","best_model.pth"]
    for file in os.scandir(metrics_dir):
        if file.name in files_to_clean:
            os.system(f"rm {file.path}")
    os.system(f"cp {metrics_dir}/enforced_model.pth {metrics_dir}/best_model.pth")
    print("Cleaned the metrics directory.")

if __name__ == "__main__":
    base_config = BaseConfig()
    solver_dir = sys.argv[1] if len(sys.argv) > 1 else str(base_config.solver_dir)
    assets_dir = solver_dir.replace("Solvers", "Assets")
    metrics_dir = solver_dir.replace("Solvers", "ModelDump")
    foamRemoveTimes(solver_dir)
    cleanAssets(assets_dir)
    cleanMetrics(metrics_dir)
