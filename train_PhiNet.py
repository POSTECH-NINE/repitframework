import torch
import torch.optim as optim
import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from repitframework.Models.FVMN.fvmn import ConvPhiNet
from repitframework.config import TrainingConfig
from repitframework.Dataset.fvmn import PhiDataset

torch.cuda.empty_cache()
torch.manual_seed(0)

training_config = TrainingConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2  # Reduced batch size for memory efficiency

phi_model = ConvPhiNet()
phi_model = phi_model.to(device)

phi_dataset = PhiDataset(training_config, start_time=10.0, end_time=19.0)
phi_dataloader = DataLoader(phi_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

phi_val_dataset = PhiDataset(training_config, start_time=19.01, end_time=20.0, calculate_stat=False)
phi_val_dataloader = DataLoader(phi_val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

optimizer = optim.AdamW(phi_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
loss_fn = torch.nn.MSELoss()
val_loss_check = float("inf")

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
    scheduler.step(train_loss)

    phi_model.eval()
    with torch.no_grad():
        for x_batch, y_batch in phi_val_dataloader:
            x_batch, y_batch = x_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            pred_val = phi_model(x_batch)
            val_loss = loss_fn(pred_val, y_batch).item()
            if val_loss < val_loss_check:
                val_loss_check = val_loss
                torch.save(phi_model.state_dict(), training_config.model_dir / "phi_model.pth")
    
    training_config.logger.info(f"Epoch {epoch + 1}, Train Loss: {train_loss:.17f}, Val Loss: {val_loss:.17f}")
    training_config.log_metrics("train_loss", train_loss, "PhiNet")
    training_config.log_metrics("val_loss", val_loss, "PhiNet")
