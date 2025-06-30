# Description: This script contains the code to perform sensitivity analysis on the hyperparameters of the KANG model.
import time
import argparse
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
from torch_geometric.datasets import Planetoid

from src.KANG import KANG
from src.utils import train, validate, test, set_seed

set_seed(seed=42)
torch.backends.cudnn.deterministic	= True
torch.backends.cudnn.benchmark = False

# Global device variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global dictionary of default hyperparameters.
PARAMS = {
	'hidden_channels': 32,
	'dropout': 0.1,
	'lr': 0.001,
	'wd': 4e-4,
	'grid_min': 20,
	'grid_max': -15,
	'num_grids': 4,
	'num_layers': 2
}

RUNS = 10

def node_classification(num_grids=None, grid_min=None, grid_max=None):
	# Load the Cora dataset (Planetoid)
	dataset = Planetoid(root=f"./dataset/Cora", name="Cora")
	data = dataset[0].to(device)

	epochs    = 1000
	patience  = 300

	# Initialize the model.
	model = KANG(
		dataset.num_features,
		PARAMS['hidden_channels'],
		dataset.num_classes,
		2,
		PARAMS['grid_min'] if grid_min is None else grid_min,
		PARAMS['grid_max'] if grid_max is None else grid_max,
		PARAMS['num_grids'] if num_grids is None else num_grids,
		PARAMS['dropout'],
		device=device,
	).to(device)

	lr = PARAMS['lr']
	weight_decay = PARAMS['wd']
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

	best_val_acc = 0
	patience_trigger = 0
	best_model_path = "./experiments/sensitivity/gkan.pth"

	epochs_times = []

	# Lists for logging training progress (if needed)
	train_accs, train_losses, val_accs, val_losses = [], [], [], []

	for epoch in range(epochs):
		time_start = time.time()
		train_acc, train_loss = train(model, data, optimizer)
		val_acc, val_loss = validate(model, data)
		time_end = time.time()
		epochs_times.append(time_end - time_start)

		train_accs.append(train_acc)
		train_losses.append(train_loss)
		val_accs.append(val_acc)
		val_losses.append(val_loss)

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			best_epoch = epoch
			patience_trigger = 0
			torch.save(model.state_dict(), best_model_path)
		else:
			patience_trigger += 1

		if patience_trigger > patience:
			break
		
	model.load_state_dict(torch.load(best_model_path))
	test_acc = test(model, data)

	return test_acc, np.mean(epochs_times)

def local_analysis_num_grids(results=False):
	results_file = "./experiments/sensitivity/local_analysis_num_grids_results.npz"
	"""
	Perform local sensitivity analysis by varying num_grids while keeping grid_min and grid_max fixed.
	Saves a line plot showing test accuracy and training time.
	"""
	# Define the range of num_grids values to test.
	num_grids_values = [2, 4, 6, 8, 10, 15, 20]
	test_accs = []
	test_stds = []
	training_times = []
	training_times_stds = []

	if not results:
		for num in num_grids_values: 
			print(f"\n[Local Analysis] Running experiment with num_grids = {num}")
			run_test_accs = []
			run_training_times = []
			for _ in tqdm(range(RUNS), leave=False):
				test_acc, train_time = node_classification(num)
				run_test_accs.append(test_acc)
				run_training_times.append(train_time)
			test_accs.append(np.mean(run_test_accs))
			test_stds.append(np.std(run_test_accs))
			training_times.append(np.mean(run_training_times))
			training_times_stds.append(np.std(run_training_times))

		np.savez(results_file,
							num_grids_values=num_grids_values,
							test_accs=test_accs,
							test_stds=test_stds,
							training_times=training_times,
							training_times_stds=training_times_stds)
		print(f"Results saved to {results_file}")

	# Load the saved results.
	loaded 							= np.load(results_file)
	num_grids_values		= loaded['num_grids_values']
	test_accs 					= loaded['test_accs']
	test_stds 					= loaded['test_stds']
	training_times 			= loaded['training_times']
	training_times_stds	= loaded['training_times_stds']

	# Plot per Test Accuracy
	_, ax1 = plt.subplots()
	ax1.set_xlabel('Number of control points', fontsize=12)
	ax1.set_ylabel('Test Accuracy', fontsize=12)
	ax1.plot(num_grids_values, test_accs, marker='o', color='tab:blue', label='Test Accuracy')
	ax1.fill_between(num_grids_values,
					np.array(test_accs) - np.array(test_stds),
					np.array(test_accs) + np.array(test_stds),
					color='tab:blue', alpha=0.2)
	ax1.tick_params(axis='x', labelsize=12) 
	ax1.tick_params(axis='y', labelsize=12) 
	ax1.legend(loc='upper left')
	# ax1.set_title("Effect of num_grids on Test Accuracy")
	plt.tight_layout()
	plt.savefig("./experiments/sensitivity/local_analysis_test_accuracy.pdf")
	plt.close()

	# Plot per Average Epoch Time
	_, ax2 = plt.subplots()
	ax2.set_xlabel('Number of control points', fontsize=12)
	ax2.set_ylabel('Average Epoch Time (s)', fontsize=12)
	ax2.plot(num_grids_values, training_times, marker='s', linestyle='--', color='tab:red', label='Training Time')
	ax2.fill_between(num_grids_values,
					np.array(training_times) - np.array(training_times_stds),
					np.array(training_times) + np.array(training_times_stds),
					color='tab:red', alpha=0.2)
	ax2.tick_params(axis='x', labelsize=12) 
	ax2.tick_params(axis='y', labelsize=12) 
	ax2.legend(loc='upper left')
	# ax2.set_title("Effect of num_grids on Epoch Time")
	plt.tight_layout()
	plt.savefig("./experiments/sensitivity/local_analysis_epoch_time.pdf")
	plt.close()

	print("Local analysis plots for Test Accuracy and Epoch Time saved.")

def grid_search_grid_min_max():
	"""
	Perform a grid search over grid_min and grid_max (keeping num_grids fixed) and record test accuracy.
	Saves a heatmap plot (confusion matrix style) showing the test accuracy for each combination.
	"""
	plot_path = "./experiments/sensitivity/grid_search_grid_min_max.pdf"
	# Define grid ranges. Only consider valid combinations where grid_min < grid_max.
	grid_min_values = [-30, -28, -25, -23, -20, -18, -15, -12, -10, -8, -5, -3, -1]
	# grid_min_values = [-3, -1]
	grid_max_values = sorted([-1*val for val in grid_min_values])

	print(grid_min_values)
	print(grid_max_values)

	test_acc_matrix = np.zeros((len(grid_min_values), len(grid_max_values)))
	test_std_matrix = np.zeros((len(grid_min_values), len(grid_max_values)))

	for i, grid_min in enumerate(grid_min_values):
		for j, grid_max in enumerate(grid_max_values):
			print(f"\n[Grid Search] Running experiment with grid_min = {grid_min}, grid_max = {grid_max}")
			run_accs = []
			for _ in tqdm(range(RUNS), leave=False):
				test_acc, _ = node_classification(grid_min=grid_min, grid_max=grid_max)
				run_accs.append(test_acc)
			test_acc_matrix[i, j] = np.mean(run_accs)
			test_std_matrix[i, j] = np.std(run_accs)

	fig, ax = plt.subplots()
	# Usa la mappa "Purples" per avere sfumature di viola
	cax = ax.imshow(test_acc_matrix, interpolation='nearest', cmap='Purples')
	ax.set_xlabel("Max grid value")
	ax.set_ylabel("Min grid value")
	ax.set_xticks(np.arange(len(grid_max_values)))
	ax.set_xticklabels(grid_max_values)
	ax.set_yticks(np.arange(len(grid_min_values)))
	ax.set_yticklabels(grid_min_values)

	# Annotazione dei valori all'interno di ogni cella: media ± std.
	# Per ciascuna cella, calcoliamo il colore corrispondente e decidiamo il colore del font in base alla luminosità.
	# for i in range(len(grid_min_values)):
	# 	for j in range(len(grid_max_values)):
	# 		mean = test_acc_matrix[i, j]
	# 		std = test_std_matrix[i, j]
	# 		# Otteniamo il colore RGBA corrispondente al valore medio normalizzato
	# 		rgba = cax.cmap(cax.norm(mean))
	# 		# Calcoliamo la luminanza usando la formula standard
	# 		luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
	# 		font_color = "white" if luminance < 0.5 else "black"
	# 		ax.text(j, i, f"{mean:.2f}\n± {std:.2f}", ha="center", va="center", color=font_color)

	fig.colorbar(cax)
	plt.tight_layout()
	plt.savefig(plot_path)
	plt.close()
	print(f"Grid search confusion matrix saved as {plot_path}.")

def main():
	print(f"Running on device: {device}")

	parser = argparse.ArgumentParser(description='Sensitivity Analysis of KANG Model')
	parser.add_argument('--num_grids', action='store_true', help='Run local analysis on num_grids.')
	parser.add_argument('--grid_range', action='store_true', help='Run grid search on grid_min and grid_max.')
	parser.add_argument("--results", action="store_true", help="Plot already computed results")
	args = parser.parse_args()

	if args.num_grids:
		print("Running local analysis on num_grids...")
		local_analysis_num_grids(args.results)
	elif args.grid_range:
		print("Running grid search on grid_min and grid_max...")
		grid_search_grid_min_max()
	elif not args.num_grids and not args.grid_range:
		print("Running local analysis on num_grids...")
		local_analysis_num_grids()
		print("Running grid search on grid_min and grid_max...")
		grid_search_grid_min_max()
	else:
		print('ERR')

if __name__ == '__main__':
	main()