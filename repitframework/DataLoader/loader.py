from torch.utils.data import DataLoader, Subset, Dataset
from typing import Tuple

def train_val_split(
		dataset: Dataset, 
		batch_size: int,
		train_size: float = 2/3,
	) -> Tuple[DataLoader, DataLoader]:
	"""
	Function to split the dataset into training and validation sets.
    Args
	----
        dataset (Dataset): The dataset to be split.
        batch_size (int): The batch size for the DataLoader.
        train_frac (float): The fraction of the dataset to be used for training. Default is 2/3.
    Returns
    -------
        tuple: A tuple containing the training and validation DataLoaders.  
	"""
	# Split indices for train/validation
	data_size = len(dataset)
	indices = list(range(data_size))
	train_size = int(data_size * train_size)
	# train_indices, val_indices = train_test_split(indices, test_size=1/3, random_state=1004)
	train_indices = indices[:train_size]
	val_indices = indices[train_size:]

	train_dataset = Subset(dataset, train_indices)
	val_dataset = Subset(dataset, val_indices)

	# Create DataLoaders for X (dataset)
	train_loader = DataLoader(train_dataset, 
							batch_size=batch_size, 
							shuffle=True, 
							drop_last=False, 
							pin_memory=True)
	val_loader = DataLoader(val_dataset, 
							batch_size=batch_size, 
							shuffle=False, 
							drop_last=False, 
							pin_memory=True)

	return train_loader, val_loader