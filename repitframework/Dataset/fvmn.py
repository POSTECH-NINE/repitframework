from typing import List, Union, Literal, Optional
from pathlib import Path
import numpy as np

from .baseline import BaseDataset
from .utils import hard_constraint_bc, add_feature


class FVMNDataset(BaseDataset):
	"""
	Args
	----
		start_time: Union[int, float]
			- Start time of the data.
		end_time: Union[int, float]
			- End time of the data.
		time_step: Union[int, float]
			- Time step of the data.
		dataset_dir: Union[str, Path]
			- Directory where the numpy files are stored.
		first_training: bool
			- If True, then the mean and std will be calculated and saved.
		vars_list: List[str]
			- List of variables. Example: ["T", "U"]
		extended_vars_list: List[str]
			- List of variables extended to dims. Example: ["T", "U_x", "U_y"]
		dims: int
			- Number of dimensions. Example: 2
		round_to: int
			- Have to use it to dodge rounding errors. Example: 2
		grid_x: int
			- Number of grid points in x direction. Example: 200
		grid_y: int
			- Number of grid points in y direction. Example: 200
		grid_z: int
			- Number of grid points in z direction. Example: 1
		grid_step: float
			- Grid step size. Example: 0.005
		output_dims: Literal["BD", "BCD", "BCHW"]
			- The shape you want to get the data. Example: ["BD"],
		do_normalize: bool
			- If True, then the data will be standardized using the mean and std of the data.
		left_wall_temperature: float
			- Temperature of the left wall. Default: 288.15 (only for natural convection 2D)
		right_wall_temperature: float
			- Temperature of the right wall. Default: 307.75 (only for natural convection 2D)
		do_feature_selection: bool
			- If True, then the correlated features will be added to the data.
	"""
	def __init__(self,
		start_time: Union[int, float],
		end_time: Union[int, float],
		time_step: Union[int, float],
		dataset_dir: Union[str, Path],
		first_training: bool = False,
		vars_list: Optional[List[str]] = ["T","U"],
		extended_vars_list: Optional[List[str]] = ["T", "U_x", "U_y"],
		dims: int = 2,
		round_to: int = 2,
		grid_x: int = 200,
		grid_y: int = 200,
		grid_z: int = 1,
		grid_step: float = 0.005,
		output_dims: Literal["BD", "BCD", "BCHW"] = "BD",
		do_normalize: bool = True,
		left_wall_temperature: float = 288.15,
		right_wall_temperature: float = 307.75,
		bc_type: str = "enforced",
		do_feature_selection: bool = True
		):

		self.left_wall_temperature = left_wall_temperature
		self.right_wall_temperature = right_wall_temperature
		self.bc_type = bc_type
		self.do_feature_selection = do_feature_selection

		super().__init__(
		start_time=start_time,
		end_time=end_time,
		time_step=time_step,
		dataset_dir=dataset_dir,
		first_training=first_training,
		vars_list=vars_list,
		extended_vars_list=extended_vars_list,
		dims=dims,
		round_to=round_to,
		grid_x=grid_x,
		grid_y=grid_y,
		grid_z=grid_z,
		grid_step=grid_step,
		output_dims=output_dims,
		do_normalize=do_normalize,
		)

		self.left_wall_temperature = left_wall_temperature
		self.right_wall_temperature = right_wall_temperature
		self.bc_type = bc_type
		self.do_feature_selection = do_feature_selection

		self.inputs, self.labels = self._inputs_labels()

	def _prepare_input(self, time) -> np.ndarray:
		'''
		Regarding the order of the variables in input data, two things matter: 
		1. The list of variables in "data_vars"
		2. The dimension of the data: 1D, 2D, 3D defined in "data_dim"

		Example
		-------
		1. If data_vars = ["U", "T"] and data_dim = 2, then the order of the variables in the input data will be: 
			U_x, U_y, T
		2. If data_vars = ["T", "U"] and data_dim = 3, then the order of the variables in the input data will be:
			T, U_x, U_y, U_z

		Steps
		-----
		1. Load the numpy files from the data_path directory. [U_0.npy, T_0.npy]
		2. Parse the numpy files.
		   a. If the data is VECTOR, split the data into x, y, z components. From this function: we get [200,200,2] shape.
		   b. If the data is SCALAR, keep the data as it is. From this function: we get [200,200] shape.
		3. Add zero padding to the data. From this function: we get [202,202] shape.
		4. Add correlated features to the data. From this function: we get [40000,5] shape.
		5. We exclude the boundary cells from the data. From this function: we get [39204,5] shape i.e. (200-2) * (200-2) = 39204
		5. Concatenate the data. From this function: we get [39204,15] shape. if we have 3 variables in the data_vars. 
		'''
		temp = super()._prepare_input(time) #Shape: [vars, grid_y, grid_x]
		if not self.do_feature_selection:
			return temp
		
		temp = hard_constraint_bc(
					temp,
					self.extended_vars_list,
					self.left_wall_temperature,
					self.right_wall_temperature
				) if self.bc_type == "enforced" else temp
		data = [add_feature(data) for data in temp]  
		return np.concatenate(data, axis=0) # Always concatenate (already stacked in add_feature) in axis=0: ["BD", "BCD", "BCHW"]
	
	def _prepare_label(self, time: float) -> np.ndarray:
		"""
		Returns difference between input at t+dt and t for all variables.
		Args
		----
		time: float
			- The time at which the input is prepared.

		Note:
		-----
		Because of feature selection, we are extending the labels including the neighboring cells.
		But we want the model to predict the central cells, from which we can again do the feature selection for
		prediction. Until now, for [BD, BCD, BCHW]-> D,C,C contain these features, so hard coded it.
		"""
		data_t = self._prepare_input(time)
		next_time = round(time + self.time_step, self.round_to)
		data_t_next = self._prepare_input(next_time)
		if self.do_feature_selection:
			skip_step = (2*self.dims)+1
			return data_t_next[::skip_step] - data_t[::skip_step] # Because prepare input always adds features in dimension 0.
		return data_t_next - data_t
	
	

if __name__ == "__main__":
	
	dataset_dir = "/home/shilaj/repitframework/repitframework/Assets/natural_convection_case1_backup"
	dataset = FVMNDataset(
		10.0,
		10.02,
		0.01,
		dataset_dir,
		bc_type="enforced",
		do_feature_selection=True,
		left_wall_temperature=288.15,
		right_wall_temperature=307.75,
		first_training=True,
		vars_list=["U", "T"],
		extended_vars_list=["U_x", "U_y", "T"],
		dims=2,
		grid_x=200,
		grid_y=200,
		grid_z=1,
		output_dims="BD",
		do_normalize=True
	)
	print(dataset)
