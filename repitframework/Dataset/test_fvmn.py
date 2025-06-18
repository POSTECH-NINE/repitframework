from fvmn import FVMNDataset
from repitframework.config import TrainingConfig
import numpy as np
import torch
import json

if __name__ == "__main__":
    training_config = TrainingConfig()
    data_path = training_config.assets_path
    start_time = 10.0
    end_time = 10.02
    time_step = 0.01

    data = FVMNDataset(training_config,
                       True,
                       data_path,
                       start_time,
                       end_time,
                       time_step,
                       )
    inputs, labels = data._prepare_inputs_and_labels()

    metrics_save_path = training_config.model_dir/ "denorm_metrics.json"
    with open(metrics_save_path, "r") as f: 
        metrics = json.load(f)

    input_mean = metrics["input_MEAN"]
    input_std = metrics["input_STD"]
    label_mean = metrics["label_MEAN"]
    label_std = metrics["label_STD"]
    actual_inputs = FVMNDataset.denormalize(inputs, input_mean, input_std)[:,::5] + \
                    FVMNDataset.denormalize(labels, label_mean, label_std)

    data_test = FVMNDataset(training_config,
                       False,
                       data_path,
                       10.01,
                       10.03,
                       time_step,
                       )
    inputs_, _ = data_test._prepare_inputs_and_labels()
    test_inputs = FVMNDataset.denormalize(inputs_, input_mean, input_std)[:,::5]

    print(np.allclose(actual_inputs, test_inputs, rtol=1e-1))