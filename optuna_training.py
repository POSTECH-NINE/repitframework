import optuna
import torch
from torch.utils.data import DataLoader, Subset

from repitframework.Models.FVMN.fvmn import FVMNetwork
from repitframework.Dataset.fvmn import FVMNDataset
from repitframework.config import TrainingConfig, OpenfoamConfig
from repitframework.OpenFOAM.utils import OpenfoamUtils

device = "cuda:0" if torch.cuda.is_available() else "cpu"
optuna.logging.set_verbosity(optuna.logging.INFO)
torch.manual_seed(42)


def get_dataloader(training_config: TrainingConfig, dataset, batch_size=None):
    """
    Returns DataLoaders that provide (x, y) batches for training and validation.
    
    If `dataset_phi` is provided, it ensures `x` and `y` batches are aligned correctly.
    """
    batch_size = batch_size if batch_size else training_config.batch_size

    # Split indices for train/validation
    data_size = len(dataset)
    indices = list(range(data_size))
    train_indices = indices[:int(2 * data_size / 3)]
    val_indices = indices[int(2 * data_size / 3):]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader


if __name__ == "__main__":
    training_config = TrainingConfig()
    epochs = 50
    variables = training_config.get_variables()
    ux_index = variables.index("U_x")
    uy_index = variables.index("U_y")
    t_index = variables.index("T")

    patience = 10
    openfoam_config = OpenfoamConfig()
    openfoam_utils = OpenfoamUtils(openfoam_config)
    # openfoam_utils.run_solver(
    #     start_time=10.0,
    #     end_time=10.03
    # )
    # Define the objective function
    def objective(trial):
        # Suggest hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1e-8, 1e-1, log=True)
        # optimizer_name = trial.suggest_categorical("optimizer", ["adam", "SGD", "RMSprop", "adamw"])
        hidden_size = trial.suggest_int("hidden_size", 10, 512)
        hidden_layers = trial.suggest_int("hidden_layers", 1, 12)

        # Create the model and dataset
        model = FVMNetwork(
            training_config=training_config,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
            activation=training_config.activation
        )
        dataset = FVMNDataset(
            training_config=training_config,
            start_time=10.62,
            end_time=10.64,
            first_training=True
        )
        train_loader, val_loader = get_dataloader(training_config, dataset)

        # Move model to device
        model.to(device)

        # Define loss function and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        # if optimizer_name == "adam":
        #     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # elif optimizer_name == "SGD":
        #     momentum = trial.suggest_float("momentum", 0.0, 0.99)
        #     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        # elif optimizer_name == "RMSprop":
        #     optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        # elif optimizer_name == "adamw":
        #     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        # elif optimizer_name == "lbfgs":
        #     # optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate)
        #     pass

        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(epochs):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                # if optimizer_name == "lbfgs":
                #     # Define the closure function required for LBFGS
                #     def closure():
                #         optimizer.zero_grad()
                #         predictions = model(x_batch)
                #         loss = (
                #             criterion(predictions["T"], y_batch[:, t_index:t_index+1]) +
                #             criterion(predictions["U_x"], y_batch[:, ux_index:ux_index+1]) +
                #             criterion(predictions["U_y"], y_batch[:, uy_index:uy_index+1])
                #         )
                #         loss.backward()
                #         return 0
                #     # optimizer.step(closure)
                # else:
                optimizer.zero_grad()
                predictions = model(x_batch)
                loss = (
                    criterion(predictions["T"], y_batch[:, t_index:t_index+1]) +
                    criterion(predictions["U_x"], y_batch[:, ux_index:ux_index+1]) +
                    criterion(predictions["U_y"], y_batch[:, uy_index:uy_index+1])
                )
                loss.backward()
                optimizer.step()

            # Compute validation loss after each epoch
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    predictions = model(x_val)
                    loss = (
                        criterion(predictions["T"], y_val[:, t_index:t_index+1]) +
                        criterion(predictions["U_x"], y_val[:, ux_index:ux_index+1]) +
                        criterion(predictions["U_y"], y_val[:, uy_index:uy_index+1])
                    )
                    val_loss += loss.item() * x_val.size(0)

            val_loss /= len(val_loader.dataset)

            training_config.logger.info(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}")
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Early stopping check
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

            trial.report(val_loss, epoch)
            # if trial.should_prune():
            #     raise optuna.exceptions.TrialPruned()
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}")
        return val_loss

    # Create a study object
    study = optuna.create_study(
        direction="minimize",
        study_name="FVMN_FirstTransferLearning",
        storage="sqlite:///fvmn_FirstTransferLearning.db",
        load_if_exists=True
    )
    # Optimize the objective function
    study.optimize(objective, n_trials=10000)

    # Print the best hyperparameters
    print("Best hyperparameters: ", study.best_params)
