from torch.utils.data import Dataset
from repitframework import config, OpenFOAM
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

foam_config = config.OpenfoamConfig()

class MLPDataset(Dataset):
    def __init__(self, data, target):
        super(MLPDataset, self).__init__(data, target)
        self.data = data
        self.target = target
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

class FVMNDataset(Dataset):
    def __init__(self, data_path:Path, start_time, end_time):
        '''
        Keep in mind:
        1. To prepare for the training data, we must have numpy files from start_time to end_time mentioned here.
        Example: 
        start_time = 0, end_time=5, then we must have numpy files from 0 to 5.
        2. We don't use the fifth time step data as a network input. 
        but, we use the difference between the fifth and fourth time step as a network label.
        Example: 
        we give input from start_time(0) to end_time(5) - time_step(1) then labels will be:
        1 - 0 : 1st label
        2 - 1 : 2nd label
        3 - 2 : 3rd label
        4 - 3 : 4th label
        5 - 4 : 5th label

        input_shape from 0 to 4: [(grid_x-2) * (grid_y-2) * 5, 15]
        label_shape from 1 to 5: [(grid_x-2) * (grid_y-2) * 5, 3]

        '''
        super().__init__()
        self.start_time = start_time
        self.end_time = end_time
        self.vars: list = foam_config.data_vars
        self.time_step = foam_config.time_step    


        ############## ------------ Data Integrity Check --------------############

        # First thing first, we must ensure that we have the data from start_time to end_time
        # in the data_path directory.
        self.data_path = data_path if isinstance(data_path, Path) else Path(data_path)
        assert self.data_path.exists(), f"Data path: {self.data_path} doesn't exist."
        assert self.is_present(), f"Data is missing in the directory: {self.data_path}.\n\
                                    You must have data from {start_time} to {end_time} for variables: {self.vars}"      
        ############ ----------------------------------------------############
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    def is_present(self):
        for var in self.vars:
            for time in np.arange(self.start_time, self.end_time+self.time_step, self.time_step):
                if not (self.data_path / f"{var}_{time}.npy").exists():
                    return False
        return True
    def parse_numpy(self, data_path:Path):
        data:np.ndarray = np.load(data_path)
        if len(data.shape) == 2: # (40000, 3): when get from OpenFOAM. VECTOR data
            if foam_config.data_dim == 1: # 1D
                return data[:,0].reshape(foam_config.grid_y, foam_config.grid_x, order="C")
            elif foam_config.data_dim == 2: #2D
                assert data.shape[0] == foam_config.grid_x * foam_config.grid_y, "check data shape and grid size mentioned in config."
                # Why order="F"? Check: https://github.com/JBNU-NINE/repit_container/blob/main/repit_wiki/Data-Loader-for-FVMN.md
                x_data = data[:,0].reshape(foam_config.grid_y, foam_config.grid_x, order="C")
                y_data = data[:,1].reshape(foam_config.grid_y, foam_config.grid_x, order="C")
                return np.stack([x_data, y_data], axis=-1)
            elif foam_config.data_dim == 3: #3D
                x_data = data[:,0].reshape(foam_config.grid_y, foam_config.grid_x, order="C")
                y_data = data[:,1].reshape(foam_config.grid_y, foam_config.grid_x, order="C")
                z_data = data[:,2].reshape(foam_config.grid_y, foam_config.grid_x, order="C")
                return np.stack([x_data, y_data, z_data], axis=-1)
            else: # Beyond 3D
                raise NotImplementedError("This framework doesn't support beyond 3D.")
        elif len(data.shape) > 2:
            raise NotImplementedError("Till now, we have not come across this use case.")
        else: # SCALAR data
            return data.reshape(foam_config.grid_y, foam_config.grid_x, order="C")
        
    def add_zero_padding(self, data:np.ndarray):
        return np.pad(data, 1, mode="constant", constant_values=0)
        
    def add_feature(self, padded_matrix:np.ndarray):
        window_shape = (3, 3)
        sliding_window = np.lib.stride_tricks.sliding_window_view(padded_matrix, window_shape)
        x,y = window_shape[0] // 2, window_shape[1] // 2 
        correlated_features = np.stack([
            sliding_window[:,:,x,y],
            sliding_window[:,:,x-1,y],
            sliding_window[:,:,x+1,y],
            sliding_window[:,:,x,y-1],
            sliding_window[:,:,x,y+1]
        ], axis=-1)
        return correlated_features.reshape(-1, 5)
    
    def prepare_input(self, time):
        '''
        Regarding the order of the variables in input data, two things matter: 
        1. The list of variables in the config file: "data_vars"
        2. The dimension of the data: 1D, 2D, 3D defined in the config file as "data_dim"
        Example: 
        1. If data_vars = ["U", "T"] and data_dim = 2, then the order of the variables in the input data will be: 
            U_x, U_y, T
        2. If data_vars = ["T", "U"] and data_dim = 3, then the order of the variables in the input data will be:
            T, U_x, U_y, U_z
        '''
        data_path = self.data_path
        full_data_path = [data_path / f"{var}_{time}.npy" for var in self.vars]
        numpy_data = [self.parse_numpy(data_path) for data_path in full_data_path]
        temp = list()
        for data in numpy_data:
            if len(data.shape) > 2:
                for i in range(2):
                    temp.append(data[:,:,i])
            else:
                temp.append(data)
        data = [self.add_feature(self.add_zero_padding(data)) for data in temp]
        grid_number_excluding_bc = (foam_config.grid_x-2) * (foam_config.grid_y - 2)
        return np.concatenate(data, axis=1)[:grid_number_excluding_bc, :]

    def calculate_difference(self, time):
        '''
        If we have data from 0s to 5s and input data should be from 0s to 4s. 
        This function calculates the labels(target data) as difference between two consecutive time steps.
        Example: 
        1 - 0: 1st label
        2 - 1: 2nd label
        3 - 2: 3rd label
        4 - 3: 4th label
        5 - 4: 5th label
        '''
        data_t = self.prepare_input(time)
        data_t_next = self.prepare_input(time + self.time_step)
        return data_t_next[:,::5] - data_t[:,::5]
    
    @staticmethod
    def normalize(data):
        return (data - np.mean(data,axis=0))/np.std(data, axis=0)
    @staticmethod
    def denormalize(data, mean, std):
        return (data * std) + mean
    
    def normalize_input(self):
        '''
        If we have data from 0s to 5s then we will have input data from 0s to 4s.
        '''
        start_time = self.start_time
        end_time = self.end_time
        data = [self.prepare_input(time) for time in np.arange(start_time, end_time, self.time_step)]
        data = np.concatenate(data, axis=0)
        return self.normalize(data)
    
    def normalize_labels(self):
        pass 

    def test_train_split(self):
        pass 

class CNNDataset(Dataset):
    def __init__(self, data, target):
        super(CNNDataset, self).__init__(data, target)

class RNNDataset(Dataset):
    def __init__(self, data, target):
        super(RNNDataset, self).__init__(data, target)


if __name__ == "__main__":
    data_path = Path("/home/openfoam/repitframework/repitframework/Assets/natural_convection")
    start_time = 5.0
    end_time = 5.02
    data = FVMNDataset(data_path, start_time, end_time)
    input_data = data.normalize_input()
    print(input_data.shape)
    