from pathlib import Path
import subprocess
import re
from copy import deepcopy
import json

import numpy as np
import torch
import Ofpp

from repitframework.config import OpenfoamConfig, TrainingConfig
from repitframework.OpenFOAM import OpenfoamUtils
from repitframework.Models.FVMN.fvmn import ConvPhiNet


'''
To calculate rho: 
rho = P*W / R*T
	P: latest CFD time kg/ms2
	W: 28.96 gm/mol | 0.02896 kg/mol
	R: 8.31446261815324 J/mol.K
	T: Predicted field K

OR: 
rho = rho_0 - alpha*rho_0(T-T_0): https://www.simscale.com/docs/simwiki/cfd-computational-fluid-dynamics/what-is-boussinesq-approximation/
'''

def calculate_rho(pressure_data:np.ndarray, temperature_data: np.ndarray) -> np.ndarray:
	mol_wt = 0.02896
	gas_constant = 8.31446261815324
	temperature_data = temperature_data.reshape(-1)

	rho_idealgas = (pressure_data * mol_wt) / (gas_constant * temperature_data)
	return rho_idealgas

def calculate_phi(phi_model_path:Path, velocity_data:np.ndarray, metrics_path:Path=None) -> np.ndarray:
	'''
	We need to calculate phi using the model. 
	'''
	ux = velocity_data[:,0].reshape(200,200, order='F')
	uy = velocity_data[:,1].reshape(200,200, order='F')
	velocity = np.stack([ux, uy], axis=0)
	velocity = velocity.reshape(1,2,200,200)

	with open(metrics_path, "r") as f:
		metrics = json.load(f)

	phi_input_mean = np.array(metrics["phi_input_MEAN"])
	phi_input_std = np.array(metrics["phi_input_STD"])
	phi_label_mean = np.array(metrics["phi_label_MEAN"])
	phi_label_std = np.array(metrics["phi_label_STD"])

	velocity = (velocity - phi_input_mean) / phi_input_std
	velocity = torch.Tensor(velocity).to("cuda" if torch.cuda.is_available() else "cpu")
	torch.cuda.empty_cache()
	phi_model = ConvPhiNet()
	phi_model.to("cuda" if torch.cuda.is_available() else "cpu")
	phi_model.load_state_dict(torch.load(phi_model_path, weights_only=True))
	phi_model.eval()
	phi = phi_model(velocity)
	phi = phi.cpu().detach().numpy()
	phi = phi * phi_label_std + phi_label_mean
	return phi.reshape(-1, order='F')
	

def calculate_prgh(pressure_data:np.ndarray, temperature_data:np.ndarray) -> np.ndarray:
	'''
	The height is exactly this: 
	array([[0.005, 0.01 , 0.015, ..., 0.99 , 0.995, 1.   ],
       [0.005, 0.01 , 0.015, ..., 0.99 , 0.995, 1.   ],
       [0.005, 0.01 , 0.015, ..., 0.99 , 0.995, 1.   ],
       ...,
       [0.005, 0.01 , 0.015, ..., 0.99 , 0.995, 1.   ],
       [0.005, 0.01 , 0.015, ..., 0.99 , 0.995, 1.   ],
       [0.005, 0.01 , 0.015, ..., 0.99 , 0.995, 1.   ]])
	'''
	
	gravity = 9.81
	temperature_data = temperature_data.reshape(-1)
	temp_avg = np.mean(temperature_data)
	mol_wt = 0.02896
	gas_constant = 8.31446261815324

	spatial_range = OpenfoamUtils.generate_intervals(
													0.005, 200*0.005, 
												  	time_step=0.005, round_to=3
													)
	spatial_range = np.array(spatial_range).reshape(-1,)
	height = np.tile(spatial_range, (200,1))
	
	pressure_data = pressure_data.reshape(200,200, order='F')
	temperature_data = temperature_data.reshape(200,200, order='F')

	p_rgh = pressure_data - ((mol_wt * gravity)/(gas_constant * temp_avg))* (pressure_data * height)
	return p_rgh.reshape(-1, order='F')

def include_all_features_NC(temperature_data:np.ndarray, latestML_time_dir:Path, velocity_data:np.ndarray) -> bool:
	
	pressure_path = latestML_time_dir / "p"
	assert pressure_path, '''You must have "pressure file" -- we are using pressure value from the latest CFD simulation;\n
	Because they are almost constant all over the simulation, so it does not matter.
	'''
	pressure_data = Ofpp.parse_internal_field(pressure_path)
	rho_data = calculate_rho(pressure_data, temperature_data)
	p_rgh = calculate_prgh(pressure_data, temperature_data)
	phi = calculate_phi(
					phi_model_path=Path("/home/shilaj/repitframework/repitframework/ModelDump/natural_convection/best_phi_model.pth"),
					velocity_data=velocity_data,
					metrics_path=Path("/home/shilaj/repitframework/repitframework/ModelDump/natural_convection/phi_denorm_metrics.json")
					)
	for file in latestML_time_dir.iterdir():
		if file == latestML_time_dir / "rho":
			data_str = "(\n" + parse_numpy(rho_data) + "\n)\n;"
			with open(file, "r") as f: 
				foam_data = f.read()
				foam_data = re.sub(r'\([\s\S]*?\)\n;', f'{data_str}',foam_data,count=1)
			with open(file, "w") as f: 
				f.write(foam_data)
		elif file == latestML_time_dir / "p_rgh":
			pass
			# data_str = "(\n" + parse_numpy(p_rgh) + "\n)\n;"
			# with open(file, "r") as f: 
			# 	foam_data = f.read()
			# 	foam_data = re.sub(r'\([\s\S]*?\)\n;', f'{data_str}',foam_data,count=1)
			# with open(file, "w") as f: 
			# 	f.write(foam_data)
		elif file == latestML_time_dir / "phi":
			data_str = "(\n" + parse_numpy(phi) + "\n)\n;"
			with open(file, "r") as f: 
				foam_data = f.read()
				foam_data = re.sub(r'\([\s\S]*?\)\n;', f'{data_str}',foam_data,count=1)
			# with open(file, "w") as f: 
			# 	f.write(foam_data)

	return True

def format_number(x):
	"""Format a number to 12 significant digits without scientific notation."""
	return f"{x:.17g}"  # Uses 12 significant figures, trims trailing zeros

def parse_numpy(data: np.ndarray) -> str:
    """
    Convert a NumPy array to a string representation suitable for OpenFOAM field files with writePrecision of 12.
    This ensures 12 significant digits while removing unnecessary trailing zeros.

    Args
    ----
    data: np.ndarray
        The NumPy array to convert.

    Returns
    -------
    parsed_output: str
        The string representation of the data enclosed by parentheses.
    
    Example
    -------
        "(0.000123456789102 1.23456789012 1234567890.12)"
    """
    
    if data.ndim == 1:
        return '\n'.join(map(format_number, data))
    elif data.ndim == 2:
        if data.shape[1] == 1:  # 1D array stored as column vector
            return '\n'.join(map(format_number, data[:, 0]))
        elif data.shape[1] == 2:  # 2D array (vector fields, need (x y z))
            lines = [f"({format_number(x[0])} {format_number(x[1])} 0)" for x in data]
            return '\n'.join(lines)
        else:  # 3D array (full vectors)
            lines = [f"({format_number(x[0])} {format_number(x[1])} {format_number(x[2])})" for x in data]
            return '\n'.join(lines)
    else:
        raise ValueError("Data shape not supported. Aborting conversion from numpy to OpenFOAM.")

def manage_time_uniform(solver_dir:Path, latestML_time:int|float) -> str:
	'''
	Changing time folder
	---------------------
	command::

		foamDictionary -case solver_dir -entry value -set latestML_time latestCFD_time/uniform/time

		foamDictionary -case solver_dir -entry name -set '"latestML_time"' latestCFD_time/0/time

		foamDictionary -case solver_dir -entry index -set latestML_time_without_decimal latestCFD_time/constant/time

	It also replaces the location values in every files inside the time directory. 
	Files like U, p, T, uniform/time, etc.
	'''
	# Because in the directory, other files are also present which still have the old time values.
	files_list = []
	time_dir = Path.joinpath(solver_dir, f"{latestML_time}")
	for file in time_dir.iterdir():
		if file.is_file():
			files_list.append(file)

	uniform_time_dir = Path.joinpath(solver_dir, f"{latestML_time}/uniform/time")
	if uniform_time_dir.exists(): files_list.append(uniform_time_dir)

	for file in files_list:
		if file == uniform_time_dir:
			replace_string = "/uniform"
		else:
			replace_string = ""

		with open(file, "r") as f:
			data = f.read()
			foam_data = re.sub(r'(location\s*)"([^"]*)"',rf'\1"{latestML_time}{replace_string}"',data)
		with open(file, "w") as f:
			f.write(foam_data)

	command_to_change_time_value = ["foamDictionary", 
								 "-case", 
								 solver_dir, 
								 "-entry", 
								 "value", 
								 "-set", 
								 f"{latestML_time}", 
								 f"{latestML_time}/uniform/time"]
	
	command_to_change_time_name = ["foamDictionary",
								 "-case",
								 solver_dir,
								 "-entry",
								 "name",
								 "-set",
								 f'"{latestML_time}"',
								 f"{latestML_time}/uniform/time"]
	

	if str(latestML_time).isdigit():
		latestML_time_without_decimal = int(f"{latestML_time}00")
	else:
		latestML_time_without_decimal = int(str(latestML_time).replace(".",""))
		
	command_to_change_time_index = ["foamDictionary",
								 "-case",
								 solver_dir,
								 "-entry",
								 "index",
								 "-set",
								 f"{latestML_time_without_decimal}",
								 f"{latestML_time}/uniform/time"]
	
	output_value = subprocess.run(command_to_change_time_value, check=True, capture_output=True, text=True)
	output_name = subprocess.run(command_to_change_time_name, check=True, capture_output=True, text=True)
	output_index = subprocess.run(command_to_change_time_index, check=True, capture_output=True, text=True)

	output_string = f"{output_value.stdout}\n{output_name.stdout}\n{output_index.stdout}"

	return output_string

def numpyToFoam(openfoam_config:OpenfoamConfig, 
				latestML_time:float,
				latestCFD_time:int|float=None,
				variables:list=None,
				solver_dir:Path=None,
				assets_path:Path=None,
				is_ground_truth:bool=False) -> str:
	"""
	This function takes a numpy file and writes it to an OpenFOAM file.

	Args
	----
	openfoam_config: OpenfoamConfig:
		The OpenFOAM configuration object.
	latestML_time: float 
		The final time step for which ML simulation is present. 
	latestCFD_time: int|float
		The final time step for which OpenFOAM file is already present.
	variables: list()
		The OpenFOAM variables list.
	solver_dir: Path
		The directory of the solver insider "Solvers" directory e.g: "Solvers/natural_convection"
	assets_path: Path
		The path to the assets directory where the numpy files are stored: e.g: "Assets/natural_convection"
	is_ground_truth: bool
		If True, it will load the ground truth data. If False, it will load the predicted data.
		Because, for the predicted cases we will have var_timestamp_predicted.npy files.

	NOTE
	----
	The latestCFD_time should be the time step for which the OpenFOAM file is already present. 
	Because, we need to copy format to the present time step for which we are trying to run the 
	simulation.

	Returns
	-------
	True if the function executes successfully.

	Remember
	--------
	latestML_time should always be float value. Because, while saving any value to numpy, we save it as float.
	see: repitframework/OpenFOAM/utils.py: parse_numpy

	Example
	-------
	If we have a numpy file U_3.npy, we can write it to the OpenFOAM file U at time t=3.
	"""
	solver_dir = Path(solver_dir) if solver_dir else openfoam_config.solver_dir
	assets_path = Path(assets_path) if assets_path else openfoam_config.assets_path
	variables = variables if variables else openfoam_config.extend_variables()

	if not latestCFD_time: 
		latestCFD_time = OpenfoamUtils.max_time_directory(solver_dir, round_to=openfoam_config.round_to)
	else:
		latestCFD_time = int(latestCFD_time) if latestCFD_time.is_integer() else latestCFD_time

	latestCFD_time_dir = Path.joinpath(solver_dir,f"{latestCFD_time}") # time directory for current time


	# Because, in openfoam if the time directory is float at the terminal values like; 11.0, 12.0, 13.0 are converted to 11, 12, 13.
	ml_dir_time_name = int(latestML_time) if latestML_time.is_integer() else latestML_time
	latestML_time_dir  = Path.joinpath(solver_dir, f"{ml_dir_time_name}") # time directory for next time (latestCFD_time + 1)

	# copy the contents of latest CFD simulation time to the latest ML simulation time.
	if not latestML_time_dir.exists(): subprocess.run(["cp", "-r", latestCFD_time_dir, latestML_time_dir], check=True)

	output_string = manage_time_uniform(solver_dir, ml_dir_time_name)
	
	for variable in variables:
		numpy_file_name = f"{variable}_{latestML_time}.npy" if is_ground_truth else f"{variable}_{latestML_time}_predicted.npy"
		openfoam_var_path = Path.joinpath(latestML_time_dir, f"{variable}") # openfoam variable path: where we write the numpy data    
		numpy_file_path =   Path.joinpath(assets_path, numpy_file_name) # numpy file path: where the numpy data is stored

		# numpy file processing:
		data = np.load(numpy_file_path)
		if variable == "T":
			temperature_data = deepcopy(data)
		elif variable == "U":
			velocity_data = deepcopy(data)
		data_str = "(\n" + parse_numpy(data) + "\n)\n;" # convert numpy data to OpenFOAM format
		
		with open(openfoam_var_path, "r") as file:
			foam_data_temp = file.read()
			foam_data = re.sub(r'(location\s*)"([^"]*)"',rf'\1"{latestML_time}"',foam_data_temp) # update the location to the next time step
			foam_data = re.sub(r'\([\s\S]*?\)\n;', f'{data_str}',foam_data,count=1) # Update the data in the OpenFOAM file; count=1 to replace only the first occurrence.

		with open(openfoam_var_path, "w") as file:
			file.write(foam_data)
	
	include_all_features_NC(temperature_data, latestML_time_dir, velocity_data)
	return output_string   

	
if __name__ == "__main__":
	openfoam_config = OpenfoamConfig()
	output_string = numpyToFoam(openfoam_config, latestCFD_time=10.0, latestML_time=10.0, is_ground_truth=True)
	print(output_string)