import torch
import numpy as np

from repitframework.config import NaturalConvectionConfig, OpenfoamConfig
from repitframework.OpenFOAM import OpenfoamUtils, numpyToFoam
from repitframework.plot_utils import save_loss
from repitframework.utils import save_to_state_dict, Timer
from repitframework.trainer import BaseHybridTrainer
from repitframework.predictor import BaseHybridPredictor
from repitframework.Dataset import FVMNDataset
from repitframework.DataLoader import train_val_split

torch.set_default_dtype(torch.float64)
torch.manual_seed(1004)
torch.cuda.manual_seed_all(1004)
np.random.seed(1004)

def hybrid_train_predict(training_config:NaturalConvectionConfig,
						 openfoam_config:OpenfoamConfig,
						 saved_model_name:str=None,
						 transfer_learning_epochs:int=2) -> None:
	"""
	Function to train and predict using the BaseHybridTrainer and BaseHybridPredictor classes.
	"""
	training_start_time = training_config.training_start_time
	training_end_time = training_config.training_end_time
	running_time = training_start_time
	trainer = BaseHybridTrainer(training_config, saved_model_name=saved_model_name)
	predictor = BaseHybridPredictor(training_config=training_config)
	# trainer = BaseHybridTrainer(training_config=training_config)
	print("BaseHybridTrainer initialized successfully.")

	openfoam_utils = OpenfoamUtils(openfoam_config)
	first_training = True
	# Storing times 
	cfd_times = 0.0
	ml_times = 0.0
	update_times = 0.0

	switch_count = 0
	ml_timesteps = 0
	cfd_timesteps = 0
	while running_time < training_config.prediction_end_time:
		# Run CFD first:
		with Timer() as cfd_timer:
			openfoam_utils.run_solver(
				start_time=running_time, 
				end_time=training_end_time,
				save_to_numpy=True
			)
		if training_end_time >= trainer.training_config.prediction_end_time: break
		# Create dataset instance
		dataset = FVMNDataset(
			start_time=training_start_time,
			end_time=training_end_time,
			time_step=0.01,
			dataset_dir=training_config.assets_dir,
			first_training=first_training,
			vars_list=training_config.get_variables(),
			extended_vars_list=training_config.extend_variables(),
			do_normalize=training_config.do_normalize,
			left_wall_temperature=training_config.left_wall_temperature,
			right_wall_temperature=training_config.right_wall_temperature,
			do_feature_selection= training_config.do_feature_selection
		)
		train_loader, val_loader = train_val_split(
			dataset, 
			batch_size=trainer.training_config.batch_size,
			train_size=2/3
		)

		# Train the model
		with Timer() as update_timer:
			trainer.fit(
				train_loader, 
				val_loader,
				freeze_layers= not first_training
			)
		trainer.best_validation_loss = float("inf") # Reset the best validation accuracy for transfer learning
		# Before prediction, load the best model: because we are using the same instance of self.model for prediction, hence last trained parameters will be used.
		# trainer.model, trainer.optimizer = load_from_state_dict(
		# 	model=trainer.model,
		# 	model_save_path=trainer.training_config.model_dir,
		# 	model_name="best_model.pth",
		# 	optimizer=trainer.optimizer
		# )
		trainer.model, trainer.optimizer, trainer.scheduler = trainer.from_checkpoint(f"best_model_{training_config.model_type}.pth")

		if trainer.training_config.epochs == 5000: 
			save_to_state_dict(
				trainer.model,
				trainer.training_config.model_dump_dir,
				f"init_model_{trainer.training_config.model_type}.pth",
				trainer.optimizer,
				trainer.scheduler
			)
			save_loss(training_config=training_config,save_initial_losses=True)
		
		print("\nStarting prediction from: ", 
			round(training_end_time+trainer.training_config.write_interval,2)
		)
		# Store times 
		if running_time > training_config.training_start_time:
			cfd_times += cfd_timer.elapsed.total_seconds()
			update_times += update_timer.elapsed.total_seconds()

		with Timer() as ml_timer:
			running_time = predictor.predict(prediction_start_time=training_end_time,
								   model=trainer.model)
		ml_times += ml_timer.elapsed.total_seconds()

		# ML timesteps per cross-computation
		ml_timesteps += round((running_time - training_end_time)/trainer.training_config.write_interval)
		print("ML timesteps: ", round((running_time - training_end_time)/trainer.training_config.write_interval))
		print("Switch count: ", switch_count)
		switch_count += 1
		print(f"Prediction ended at:{running_time}\n")

		# Convert predicted numpy to foam
		numpyToFoam_string = numpyToFoam(
			openfoam_config=openfoam_config, 
			latestML_time=float(running_time), 
			latestCFD_time=training_end_time,
		)

		openfoam_config.logger.info(f"Converted numpy to foam: {numpyToFoam_string}")

		# Transfer learning
		# trainer.training_config.epochs, cfd_runs = dynamic_parameters(switch_count)
		trainer.training_config.epochs = transfer_learning_epochs
		cfd_runs = 10
		# if switch_count == 2: break
		cfd_timesteps += cfd_runs
		training_end_time = round(running_time + cfd_runs*trainer.training_config.write_interval,
								  2)
		# Just using last three time steps for transfer learning: 
		training_start_time = round(training_end_time - 3*trainer.training_config.write_interval, 2)
		first_training = False


	training_config.logger.info(f"Transfer learning epochs: {trainer.training_config.epochs}")
	training_config.logger.info(f"Relative Residual Mass: {trainer.training_config.residual_threshold}\n")
	training_config.logger.info(f"Total CFD Time: {cfd_times}")
	training_config.logger.info(f"Total ML Time: {ml_times}")
	training_config.logger.info(f"Total Update Time: {update_times}")
	training_config.logger.info(f"Total ML timesteps: {ml_timesteps}")
	training_config.logger.info(f"Total CFD runs: {cfd_timesteps}")

	if_CFD_alone = (ml_timesteps + cfd_timesteps)*(cfd_times/cfd_timesteps)
	training_config.logger.info("###############################################")
	training_config.logger.info(f"CFD alone time: {if_CFD_alone}")
	training_config.logger.info(f"CFD+ML+update times: {cfd_times+ml_times+update_times}")
	training_config.logger.info(f"Acceleration: {if_CFD_alone/(ml_times + cfd_times + update_times)}")
	training_config.logger.info(f"ML timesteps per cross-computation: {ml_timesteps/switch_count}")
	training_config.logger.info(f"t_ML: {ml_times/ml_timesteps}")
	training_config.logger.info(f"t_CFD: {cfd_times/cfd_timesteps}")
	training_config.logger.info("###############################################")
	
	return if_CFD_alone
	

if __name__ == "__main__":
	
	training_config = NaturalConvectionConfig()
	openfoam_config = OpenfoamConfig()
	training_config.logger.info("Starting the hybrid training and prediction process...")
	with Timer() as total_timer:
		if_cfd_alone_time = hybrid_train_predict(training_config, 
							openfoam_config,
							saved_model_name=f"init_model_{training_config.model_type}.pth",
							transfer_learning_epochs=2)
	training_config.logger.info("Hybrid training and prediction process completed.")
	training_config.logger.info(f"Total time taken: {total_timer.elapsed.total_seconds()} seconds")
	training_config.logger.info(f"Real acceleration: {if_cfd_alone_time/total_timer.elapsed.total_seconds()}")
	# save_loss(training_config=training_config, merge_initial_losses=False)