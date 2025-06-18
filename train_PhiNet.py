import torch
import torch.optim as optim
import tqdm
import json
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from repitframework.Models.FVMN.fvmn import ConvPhiNet
from repitframework.config import TrainingConfig
from repitframework.Dataset.fvmn import PhiDataset
from pathlib import Path

torch.set_default_dtype(torch.float64)
torch.cuda.empty_cache()
torch.manual_seed(0)

training_config = TrainingConfig()
training_config.assets_path = Path("/home/shilaj/repitframework/repitframework/Assets/natural_convection_backup")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2  # Reduced batch size for memory efficiency

phi_model = ConvPhiNet()
phi_model = phi_model.to(device)

phi_dataset = PhiDataset(training_config, start_time=10.0, end_time=19.9)
phi_dataloader = DataLoader(phi_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

phi_val_dataset = PhiDataset(training_config, start_time=19.91, end_time=20.0, calculate_stat=False)
phi_val_dataloader = DataLoader(phi_val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

optimizer = optim.AdamW(phi_model.parameters(), lr=1e-3, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
loss_fn = torch.nn.MSELoss()
val_loss_check = float("inf")
train_loss_check = float("inf")

for epoch in tqdm.tqdm(range(1000), desc="Epochs", leave=False):
    phi_model.train()
    train_loss = 0.0
    for x_batch, y_batch in phi_dataloader:
        x_batch, y_batch = x_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
        pred = phi_model(x_batch)
        loss = loss_fn(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x_batch.size(0)

    train_loss /= len(phi_dataloader.dataset)
    if train_loss < train_loss_check:
        train_loss_check = train_loss
        torch.save(phi_model.state_dict(), training_config.model_dir / "phi_model_ConvNet_lowestTrainLoss.pth")
    phi_model.eval()
    with torch.no_grad():
        for x_batch, y_batch in phi_val_dataloader:
            x_batch, y_batch = x_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            pred_val = phi_model(x_batch)
            val_loss = loss_fn(pred_val, y_batch).item()
            if val_loss < val_loss_check:
                val_loss_check = val_loss
                torch.save(phi_model.state_dict(), training_config.model_dir / "phi_model_ConvNet.pth")
    
    training_config.log_metrics("train_loss", train_loss, "PhiNet")
    training_config.log_metrics("val_loss", val_loss, "PhiNet")
# def generate_intervals(start_time, end_time, time_step, round_to):
#     time_list = []
#     running_time = start_time
#     while running_time <= end_time:
#         time_list.append(round(running_time, round_to))
#         running_time = round(running_time +time_step, round_to)
#     return time_list

# def load_inputs_labels(start_time, end_time, time_step, round_to, assets_path):
#     time_list = generate_intervals(start_time, end_time, time_step, round_to)
#     inputs = []
#     labels = []
#     for time in time_list:
#         input_path = assets_path / f"U_{time}.npy"
#         label_path = assets_path / f"phi_{time}.npy"
#         inputs.append(np.load(input_path)[:,:2])
#         labels.append(np.pad(np.load(label_path), 200, "constant", constant_values=1e-20).reshape(-1,2, order="F"))
#     inputs = np.concatenate(inputs, axis=0)
#     labels = np.concatenate(labels, axis=0)

#     phi_input_mean = np.mean(inputs, axis=0)
#     phi_input_std = np.std(inputs, axis=0)
#     phi_label_mean = np.mean(labels, axis=0)
#     phi_label_std = np.std(labels, axis=0)

#     inputs = (inputs - phi_input_mean) / phi_input_std
#     labels = (labels - phi_label_mean) / phi_label_std

#     model_dump_path = str(assets_path).replace("Assets", "ModelDump")
#     model_dump_path = Path(str(model_dump_path).replace("natural_convection_backup", "natural_convection"))
#     with open(model_dump_path / "phi_denorm_metrics_1.json", "w") as f:
#         json.dump({
#             "phi_input_MEAN": phi_input_mean.tolist(),
#             "phi_input_STD": phi_input_std.tolist(),
#             "phi_label_MEAN": phi_label_mean.tolist(),
#             "phi_label_STD": phi_label_std.tolist()
#         }, f, indent=4)
#     return inputs, labels

# inputs, labels = load_inputs_labels(10.0, 20.0, 0.01, 2, Path("/home/shilaj/repitframework/repitframework/Assets/natural_convection_backup"))
# inputs = torch.from_numpy(inputs)
# targets = torch.from_numpy(labels)

# # Define dataset
# train_ds = TensorDataset(inputs, targets)

# # Define data loader
# batch_size = 100000
# train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# #Define model
# model = torch.nn.Linear(2,2)
# model = model.to("cuda")

# #Define optimizer
# opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

# #Define loss function
# loss_fn = F.mse_loss

# #Define a utility function to train the model
# def fit(num_epochs, model, loss_fn, opt):
#     for epoch in range(num_epochs):
#         train_loss = 0.0
#         for xb,yb in train_dl:
#             xb = xb.to("cuda")
#             yb = yb.to("cuda")
#             #Generate predictions
#             pred = model(xb)
#             loss = loss_fn(pred,yb)
#             #Perform gradient descent
#             loss.backward()
#             opt.step()
#             opt.zero_grad()
#             train_loss += loss.item() * xb.size(0)
#         train_loss /= len(train_dl.dataset)
#         print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, train_loss))

# #Train the model for 100 epochs
# fit(1000, model, loss_fn, opt)
# torch.save(model.state_dict(), "/home/shilaj/repitframework/repitframework/ModelDump/natural_convection/phi_model_linear.pth")
