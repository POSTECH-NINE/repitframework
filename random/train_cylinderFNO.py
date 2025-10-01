import numpy as np
import matplotlib.pyplot as plt
from repitframework.Models import FNO2D
from repitframework.plot_utils import load_metrics
import torch
from timeit import default_timer
import json
import os
import imageio
from typing import Union
from pathlib import Path
from datetime import datetime


def visualize_output(data, timestep:int,mode:str="rgb_array"
):
	fig, ax = plt.subplots(1, 1, figsize=(13, 5))
	ax = np.atleast_1d(ax)
	im = ax[0].imshow(data.T, origin="lower", cmap="coolwarm")
	fig.tight_layout()
	fig.suptitle(f"{timestep}s")
	if mode == "image":
		plt.savefig(f"{timestep}.png")
		plt.tight_layout()
		plt.close()
		return True
	elif mode == "rgb_array":
		fig.canvas.draw()
		rgb_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
			fig.canvas.get_width_height()[::-1] + (4,))
		plt.close()
		return rgb_array
	else:
		plt.close()
		raise ValueError("Invalid mode. Must be either 'image' or 'rgb_array'.")
	
def make_animation(prediction_data_path, ground_truth, type="predictions", suffix="0.4"):
	image_list = []
	for timestep in range(1, ground_truth.shape[1]):
		if type == "predictions":
			prediction_data = np.load(prediction_data_path + f"/predictions_{suffix}.npy")[0, timestep, :, :]
			image_list.append(visualize_output(prediction_data, timestep))
		else:
			image_list.append(visualize_output(ground_truth[:, timestep, :, :], timestep))
		# Optionally, you can also visualize the ground truth if needed
		# visualize_output(ground_truth[:, timestep, :, :], timestep, mode="image")
	save_name = f"prediction_{suffix}.gif" if type == "predictions" else f"ground_truth_{suffix}.gif"
	imageio.mimsave(save_name, image_list, fps=10, loop=0)

def plot_probe_points(ground_truth:np.ndarray,
					  predicted:np.ndarray,
					  locations:tuple=((120,141),(220,141), (320,141), (420,141)),
					  save_dir:Union[str, Path]=""):
	save_dir = Path(save_dir)
	fig, ax = plt.subplots(1, 1, figsize=(13, 5))
	ax = np.atleast_1d(ax)

	colors_list = ['r', 'g', 'b', 'y']

	for i, (x, y) in enumerate(locations):
		# Only add labels on the first iteration to avoid duplicate legend entries
		if i == 0:
			ax[0].plot(ground_truth[0, :, x, y], color=colors_list[i], label='Ground Truth')
			ax[0].plot(predicted[0, :, x, y], color=colors_list[i], linestyle='--', label='Predicted')
		else:
			# Plot the rest of the lines without labels
			ax[0].plot(ground_truth[0, :, x, y], color=colors_list[i])
			ax[0].plot(predicted[0, :, x, y], color=colors_list[i], linestyle='--')

	# Add the legend to the plot
	ax[0].legend()
	
	plt.title("Probe Points Comparison")
	plt.xlabel("Time Steps")
	plt.savefig(save_dir / "probe_points_0.4.png")
	plt.close()


def plot_loss(loss_metrics_path:Union[str, Path], suffix:str="0.4",
			  save_dir:Union[str, Path]=""):

	save_dir = Path(save_dir)
	metrics = load_metrics(Path(loss_metrics_path))

	plt.plot(metrics["epoch"], metrics["train_loss"], label="Training Loss", color="r")
	plt.plot(metrics["epoch"], metrics["validation_loss"], label="Validation Loss", color="b")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.yscale("log")
	plt.legend()
	plt.savefig(save_dir / f"loss_curves_{suffix}.png")
	plt.close()


class LpLoss(object):
	def __init__(self, d=2, p=2, size_average=True, reduction=True):
		super(LpLoss, self).__init__()

		#Dimension and Lp-norm type are postive
		assert d > 0 and p > 0

		self.d = d
		self.p = p
		self.reduction = reduction
		self.size_average = size_average

	def abs(self, x, y):
		num_examples = x.size()[0]

		#Assume uniform mesh
		h = 1.0 / (x.size()[1] - 1.0)

		all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

		if self.reduction:
			if self.size_average:
				return torch.mean(all_norms)
			else:
				return torch.sum(all_norms)

		return all_norms

	def rel(self, x, y):
		num_examples = x.size()[0]

		diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
		y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

		if self.reduction:
			if self.size_average:
				return torch.mean(diff_norms/y_norms)
			else:
				return torch.sum(diff_norms/y_norms)

		return diff_norms/y_norms

	def __call__(self, x, y):
		return self.rel(x, y)
	

class FNODataset(torch.utils.data.Dataset):
	def __init__(self, data_path, t_input=10):
		self.data_path = data_path
		self.t_input = t_input
		self.data = np.load(data_path)
		if not isinstance(self.data_path, Path):
			self.data_path = Path(self.data_path)
		self.root_dir = self.data_path.parent
		
	def get_dataset(self, metrics_save_path:Path):
		data = self.data

		inputs = data['inputs']
		outputs = data['outputs']
		batchsize = inputs.shape[0]
		if inputs.ndim == 3:
			inputs = inputs[:, np.newaxis, ...]

		train_mask = torch.ones(batchsize, dtype=bool)
		test_mask = torch.zeros(batchsize, dtype=bool)
		train_mask[::10] = False  # Set every 10th sample to False for training
		test_mask[::10] = True    # Set every 10th sample to True for testing	

		input_mean = inputs.mean(axis=(0, 2, 3), keepdims=True)
		input_std = inputs.std(axis=(0, 2, 3), keepdims=True)

		label_mean = outputs.mean(axis=(0, 2, 3), keepdims=True)
		label_std = outputs.std(axis=(0, 2, 3), keepdims=True)

		inputs_ = (inputs - input_mean) / input_std
		outputs_ = (outputs - label_mean) / label_std

		with open(metrics_save_path, "w") as f:
			json.dump({
				"input_mean": input_mean.tolist(),
				"input_std": input_std.tolist(),
				"label_mean": label_mean.tolist(),
				"label_std": label_std.tolist()
			}, f)

		train_input = inputs_[train_mask]
		train_label = outputs_[train_mask]

		test_input = inputs_[test_mask]
		test_label = outputs_[test_mask]

		return train_input, train_label, test_input, test_label

	def get_dataloader(self, metrics_save_path:Path, batch_size=32, step_size=1):

		train_input, train_label, test_input, test_label = self.get_dataset(metrics_save_path=metrics_save_path)

		train_dataset = torch.utils.data.TensorDataset(
			torch.tensor(train_input[:, :, ::step_size, ::step_size], dtype=torch.float32),
			torch.tensor(train_label[:, :, ::step_size, ::step_size], dtype=torch.float32)
		)
		test_dataset = torch.utils.data.TensorDataset(
			torch.tensor(test_input[:, :, ::step_size, ::step_size], dtype=torch.float32),
			torch.tensor(test_label[:, :, ::step_size, ::step_size], dtype=torch.float32)
		)

		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

		return train_loader, test_loader
	
	def get_coordinates(self):
		data = self.data
		if 'x_coords' in data and 'y_coords' in data:
			return data['x_coords'], data['y_coords']
		else:
			raise KeyError("Coordinates not found in the dataset.")
	

def log_metrics(key:str, value:int|float, metrics_type:str="training", save_dir:Union[str, Path]=""):

	save_dir = Path(save_dir)
	logging_path = save_dir / f"{metrics_type}_metrics.ndjson"

	# build a single JSON record for this metric
	record = {"key": key, "value": value}
	
	# append one JSON object per line
	with open(logging_path, "a") as f:
		f.write(json.dumps(record) + "\n")


def normalize(data, data_path, flag:str="input")-> torch.tensor:
	if not isinstance(data_path, Path):
		data_path = Path(data_path)
	
	with open(data_path, "r") as f:
		metrics = json.load(f)

	if flag == "input":
		mean = torch.tensor(metrics["input_mean"], dtype=torch.float32)
		std = torch.tensor(metrics["input_std"], dtype=torch.float32)
	elif flag == "label":
		mean = torch.tensor(metrics["label_mean"], dtype=torch.float32)
		std = torch.tensor(metrics["label_std"], dtype=torch.float32)
	else:
		raise ValueError("Invalid flag. Use 'input' or 'label'.")

	return (data - mean) / std

def denormalize(data, data_path, flag:str="input")-> torch.tensor:
	if not isinstance(data_path, Path):
		data_path = Path(data_path)
	
	with open(data_path, "r") as f:
		metrics = json.load(f)

	if flag == "input":
		mean = torch.tensor(metrics["input_mean"], dtype=torch.float32)
		std = torch.tensor(metrics["input_std"], dtype=torch.float32)
	elif flag == "label":
		mean = torch.tensor(metrics["label_mean"], dtype=torch.float32)
		std = torch.tensor(metrics["label_std"], dtype=torch.float32)
	else:
		raise ValueError("Invalid flag. Use 'input' or 'label'.")

	return data * std + mean

class FNO2DTrainer:
	def __init__(self, 
			  input_channels=1, 
			  output_channels=100, 
			  modes1=12, modes2=12, 
			  width=128, 
			  depth=4,
			  t_input=10,
			  learning_rate=0.001,
			  x_coords=None,
			  y_coords=None,
			  include_grid:bool=False,
			  epochs:int=10000):
		self.model = FNO2D(input_channels=input_channels, output_channels=output_channels,
						   modes1=modes1, modes2=modes2, width=width, depth=depth,
						   x_coords=x_coords,
						   y_coords=y_coords,
						   include_grid=include_grid)
		self.t_input = t_input
		self.epochs = epochs
		self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
		self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=4*self.epochs) # Hard coded T_max
		self.criterion = LpLoss(size_average=False)
		self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
		self.model.to(self.device)

	def predict(self, input_tensor:torch.Tensor, save_dir:Path, suffix:str="0.4"):
		model:torch.nn.Module = self.load_model(save_dir / "best_model.pth")
		model.eval()

		metrics_save_path = save_dir / "norm_denorm_metrics.json"
		input_tensor = normalize(input_tensor, metrics_save_path, flag="input")
		with torch.inference_mode():
			assert input_tensor.ndim == 4, "Input tensor must be 4D (B, C, H, W)"
			
			input_tensor = input_tensor.to(self.device)
			# Predict the next time step
			y_pred = model(input_tensor).cpu()
			# Denormalize the prediction
			y_pred = denormalize(y_pred, metrics_save_path, flag="label")
			np.save(save_dir/f"predictions_{suffix}.npy", y_pred.numpy())  # Save the prediction if needed
			# Append the prediction to the input tensor
			# input_tensor = torch.cat((input_tensor[:, 1:, ...], y_pred), dim=1)
			# running_time += 1
		return save_dir/f"predictions_{suffix}.npy"
	
	def training_loop(self, data_loader):
		step_loss = 0.0
		for xx, yy in data_loader:
			loss = 0.0 
			xx = xx.to(self.device)
			yy = yy.to(self.device)

			# for t in range(0, xx.shape[1], 1):
			# 	y = yy[:, t:t + 1, :, :]
			y_pred = self.model(xx)
			loss += self.criterion(y_pred.reshape(y_pred.shape[0], -1), yy.reshape(yy.shape[0], -1))

				# xx = torch.cat((xx[:, 1:, ...], y_pred), dim=1)
			step_loss += loss.item()
			if self.model.training:
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				self.scheduler.step()

		return step_loss
	
	def train(self, train_loader, test_loader, save_dir:Path):

		validation_loss = float('inf')
		for epoch in range(self.epochs):
			self.model.train()

			train_l2_step_loss = self.training_loop(train_loader)

			self.model.eval()
			with torch.no_grad():
				test_l2_step_loss = self.training_loop(test_loader)
			if test_l2_step_loss < validation_loss:
				self.save_model(save_dir / "best_model.pth")
				validation_loss = test_l2_step_loss

			print("\033[4A", end="")
			print(f"\nEpoch: {epoch},\nTraining loss: {train_l2_step_loss},\nValidation loss: {test_l2_step_loss}\n",  end='', flush=True)

			log_metrics("train_loss", train_l2_step_loss, "training", save_dir)
			log_metrics("validation_loss", test_l2_step_loss, "training", save_dir)
			log_metrics("epoch", epoch, "training", save_dir)

	def save_model(self, path):
		torch.save(
			{"model_dict":self.model.state_dict(),
				"optimizer_dict":self.optimizer.state_dict(),
				"scheduler_dict":self.scheduler.state_dict()
			}, path)
		# print(f"Model saved to {path}")

	def load_model(self, path):
		checkpoint = torch.load(path, map_location=self.device, weights_only=True)
		self.model.load_state_dict(checkpoint["model_dict"])
		if "optimizer_dict" in checkpoint:
			self.optimizer.load_state_dict(checkpoint["optimizer_dict"])
		if "scheduler_dict" in checkpoint:
			self.scheduler.load_state_dict(checkpoint["scheduler_dict"])
		self.model.to(self.device)
		print(f"Model loaded from {path}")
		return self.model

	
	def fit(self, train_loader, test_loader, save_dir:Path):
		metrics = self.train(train_loader, test_loader, save_dir=save_dir)
		return metrics
	

if __name__ == "__main__":
	data_path1 = "/home/shared_resource/2D_VSMR_structure_data/structure_vsmr_press_ic010_to_060_except040.npz"
	data_path2 = "/home/shared_resource/2D_VSMR_structure_data/structure_vsmr_press_ic040.npz"
	data_path3 = "/home/shared_resource/2D_VSMR_structure_data/structure_vsmr_press_ic060.npz"
	data_path4 = "/home/shared_resource/2D_VSMR_structure_data/structure_vsmr_press_ic070.npz"
	step_size = 2  # Adjust step size as needed
	case_name = "CFNO_2StepSize_64Width24Modes5Test"
	dataset = FNODataset(data_path1, t_input=10)

	date_hr = datetime.now().strftime("%m-%d_%H")
	save_dir = Path("/home/shilaj/repitframework/random/data/predictions") / date_hr / case_name
	save_dir.mkdir(parents=True, exist_ok=True)
	metrics_save_path = save_dir / "norm_denorm_metrics.json"
	# Get coordinates
	x_coords, y_coords = dataset.get_coordinates()

	cylinderFNO = FNO2DTrainer(
		epochs=10000,
		x_coords=x_coords,
		y_coords=y_coords,
		include_grid=False,
		modes1=16,
		modes2=16,
		width=128,
		depth=4,
		learning_rate=1e-4
	)

	train_loader, test_loader = dataset.get_dataloader(batch_size=10, step_size=step_size, metrics_save_path=metrics_save_path)
	# cylinderFNO.load_model("best_model_FNO2D_cylinder.pth")
	cylinderFNO.fit(train_loader, test_loader, save_dir=save_dir)

	# Testing
	test_data = np.load(data_path2)
	test_input = torch.from_numpy(np.expand_dims(test_data['inputs'], axis=1))  # Add a channel dimension if needed
	prediction_path = cylinderFNO.predict(test_input.to(torch.float32), suffix="0.4",
									   save_dir=save_dir)

	predicted_results = np.load(prediction_path)
	
	plot_probe_points(
		ground_truth=test_data["outputs"],
		predicted=predicted_results,
		locations=((110, 141), (210, 141), (310, 141), (410, 141)),
		save_dir=save_dir
	)
	plot_loss(
		loss_metrics_path=save_dir / "training_metrics.ndjson",
		suffix=f"{case_name}_0.4",
		save_dir=save_dir
	)