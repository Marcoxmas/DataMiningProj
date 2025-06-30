import time
import wandb
import argparse
import numpy as np

import torch
from torch import cat
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import Planetoid, TUDataset

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from src.KANG import KANG
from src.utils import train, validate, test, set_seed, eval_link_predictor

torch.backends.cudnn.deterministic	= True
torch.backends.cudnn.benchmark			= False

def validate_config(config):
	"""Validate configuration parameters."""
	assert config.grid_min < config.grid_max, "grid_min must be less than grid_max"
	assert 0 <= config.dropout <= 1, "dropout must be between 0 and 1"
	assert config.lr > 0, "learning rate must be positive"
	assert config.wd >= 0, "weight decay must be non-negative"
	assert config.num_grids > 0, "num_grids must be positive"

def nc(config, dataset, epochs, patience):
	"""Training objective function with improved error handling and monitoring."""
	try:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		data = dataset[0].to(device)
		best_model_path = "./experiments/node_classification/gkan.pth"
		# Validate configuration
		validate_config(config)
		
		# Initialize model
		model = KANG(
			dataset.num_features,
			config.hidden_channels,
			dataset.num_classes,
			config.layers, # layers
			config.grid_min,
			config.grid_max,
			config.num_grids,
			config.dropout,
			device=device,
			residuals=config.residuals,
			trainable_grid=config.trainable_grid,
			linspace=config.linspace
		).to(device)
		
		optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)
		best_val_acc = 0
		patience_trigger = 0
		
		for epoch in range(epochs):
			# Training step
			_, train_loss = train(model, data, optimizer)
			val_acc, val_loss = validate(model, data)
			
			# Log metrics
			wandb.log({
				"epoch": epoch,
				"train_loss": train_loss,
				"val_loss": val_loss,
				"val_acc": val_acc,
				"best_val_acc": best_val_acc,
				"learning_rate": config.lr,
				"dropout": config.dropout,
				"weight_decay": config.wd,
			})
			
			# Check for improvement
			if val_acc > best_val_acc:
				best_val_acc = val_acc
				patience_trigger = 0
				wandb.run.summary["best_val_acc"] = best_val_acc
				wandb.run.summary["best_epoch"] = epoch
				torch.save(model.state_dict(), best_model_path)
			else:
				patience_trigger += 1
				
			if patience_trigger >= patience:
				break
		
		model.load_state_dict(torch.load(best_model_path))
		test_acc = test(model, data)
		wandb.log({"test_acc": test_acc})

		return best_val_acc
		
	except Exception as e:
		wandb.log({"error": str(e)})
		raise

def lp(config, dataset, epochs, patience):
	try:
		validate_config(config)
		best_model_path = "./experiments/link_prediction/gkan.pth"

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		data = dataset[0].to(device)
		split = T.RandomLinkSplit(
			num_val=0.1, 
			num_test=0.1, 
			is_undirected=True, 
			add_negative_train_samples=False,
			neg_sampling_ratio=1.0
		)

		train_data, val_data, test_data = split(data)

		model = KANG(
			dataset.num_features,
			config.hidden_channels,
			1, # 1 output channel for link prediction
			config.layers, # layers
			config.grid_min,
			config.grid_max,
			config.num_grids,
			config.dropout,
			device=device,
			residuals=config.residuals,
			trainable_grid=config.trainable_grid,
			linspace=config.linspace
		).to(device)

		optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)
		criterion = torch.nn.BCEWithLogitsLoss()

		best_val_auc = 0
		patience_trigger = 0

		for epoch in range(epochs):
			# Training step
			model.train()
			optimizer.zero_grad()
					# Compute node embeddings using the encode function of the model
			z = model.encode(train_data.x, train_data.edge_index)

			# Dynamically sample negative edges (edges that are not present in the graph)
			neg_edge_index = negative_sampling(
					edge_index = train_data.edge_index,
					num_nodes = train_data.num_nodes,
					num_neg_samples = train_data.edge_label_index.size(1),
					method='sparse'
				)
			
			# Combine real and negative edges
			edge_label_index = cat(
					[train_data.edge_label_index, neg_edge_index],
					dim=-1,
				)
			# Create labels for real edges (1) and negative edges (0)
			edge_label = cat(
				[
					train_data.edge_label,
					train_data.edge_label.new_zeros(neg_edge_index.size(1))
				],
				dim=0
			)

			# Decode the embeddings to predict the presence of edges
			out	= model.decode(z, edge_label_index).view(-1)

			train_loss = criterion(out, edge_label)

			train_loss.backward()
			optimizer.step()

			val_auc = eval_link_predictor(model, val_data)
			if val_auc > best_val_auc:
				best_val_auc = val_auc
				patience_trigger = 0
				wandb.run.summary["best_val_auc"] = best_val_auc
				wandb.run.summary["best_epoch"]		= epoch
				torch.save(model.state_dict(), best_model_path)
			else:
				patience_trigger += 1

			if patience_trigger >= patience:
				break

			# Log metrics
			wandb.log({
				"epoch": epoch,
				"train_loss": train_loss,
				"val_auc": val_auc,
				"best_val_auc": best_val_auc,
				"learning_rate": config.lr,
				"dropout": config.dropout,
				"weight_decay": config.wd,
			})

		model.load_state_dict(torch.load(best_model_path))
		test_auc = eval_link_predictor(model, test_data)

		wandb.log({"test_auc": test_auc})
		return best_val_auc
		
	except Exception as e:
		wandb.log({"error": str(e)})
		raise

def gc(config, dataset, epochs, patience):
	validate_config(config)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	best_model_path = "./experiments/graph_classification/gkan.pth"

	num_features = dataset.num_node_features
	out_channels = dataset.num_classes
	batch_size = 64
	shuffled_dataset = dataset.shuffle()
	train_size 		= int(0.8 * len(dataset))
	val_size 			= int(0.1 * len(dataset))
	train_dataset = shuffled_dataset[:train_size]
	val_dataset 	= shuffled_dataset[train_size:train_size + val_size]
	test_dataset 	= shuffled_dataset[train_size + val_size:]
	train_loader 	= DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader 		= DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	test_loader 	= DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	model = KANG(
		num_features,
		config.hidden_channels,
		out_channels,
		config.layers, # layers
		config.grid_min,
		config.grid_max,
		config.num_grids,
		config.dropout,
		device=device,
	).to(device)

	optimizer				 	= torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)
	criterion 				= torch.nn.CrossEntropyLoss()
	best_val_acc			= 0
	patience_trigger	= 0

	for epoch in range(epochs):
		model.train()
		train_acc 	= 0
		train_loss	= 0
		for data in train_loader:
			data = data.to(device)
			optimizer.zero_grad()
			out 				= model(data.x, data.edge_index, data.batch)
			loss				= criterion(out, data.y)
			pred 				= out.argmax(dim=1)
			train_acc 	+= int((pred == data.y).sum())
			train_loss	+= loss.item()
			loss.backward()
			optimizer.step()
		train_acc 	/= len(train_loader.dataset)
		train_loss	/= len(train_loader.dataset)

		model.eval()
		val_acc = 0
		val_loss = 0
		with torch.no_grad():
			for data in val_loader:
				data = data.to(device)
				out = model(data.x, data.edge_index, data.batch)
				loss = criterion(out, data.y)
				val_loss += loss.item()
				pred = out.argmax(dim=1)
				val_acc += int((pred == data.y).sum())

		val_acc		/= len(val_loader.dataset)
		val_loss	/= len(val_loader.dataset)

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			patience_trigger = 0
			wandb.run.summary["best_val_acc"] = best_val_acc
			wandb.run.summary["best_epoch"] = epoch
			torch.save(model.state_dict(), best_model_path)
		else:
			patience_trigger += 1
		if patience_trigger >= patience:
			break

		wandb.log({
			"epoch": epoch,
			"train_acc": train_acc,
			"train_loss": train_loss,
			"val_acc": val_acc,
			"val_loss": val_loss,
			"best_val_acc": best_val_acc,
			"learning_rate": config.lr,
			"dropout": config.dropout,
			"weight_decay": config.wd,
		})
	
	model.load_state_dict(torch.load(best_model_path))
	model.eval()
	test_acc = 0
	with torch.no_grad():
		for data in test_loader:
			data = data.to(device)
			out = model(data.x, data.edge_index, data.batch)
			pred = out.argmax(dim=1)
			test_acc += int((pred == data.y).sum())
	test_acc /= len(test_loader.dataset)

	wandb.log({"test_acc": test_acc})
	return best_val_acc
	
def main(task):
	run = wandb.init(
		config={
			"epochs": 1000,
			"patience": 300,
			"seed": 42
		}
	)

	set_seed(run.config.seed)

	dataset = Planetoid(
		root=f"./dataset/Cora", 
		name="Cora", 
	)
	
	if task == 'nc':
		val_acc = nc(run.config, dataset, run.config.epochs, run.config.patience)
		wandb.log({"final_val_acc": val_acc})
	if task == 'lp':
		val_auc = lp(run.config, dataset, run.config.epochs, run.config.patience)
		wandb.log({"final_auc": val_auc})
	if task == 'gc':
		dataset = TUDataset(root='./dataset/MUTAG', name='MUTAG')
		val_acc = gc(run.config, dataset, run.config.epochs, run.config.patience)
		wandb.log({"final_val_acc": val_acc})

sweep_configuration = {
	"method": "bayes",
	"metric": {"goal": "maximize", "name": "val_acc"},
	"parameters": {
		"hidden_channels": {"values": [8, 16, 32, 64]},
		"layers": {"values": [2, 3, 4]},
		"dropout": {"min": 0.0, "max": 0.7},
		"lr": {"min": 0.0001, "max": 0.05}, 
		"wd": {"min": 0.0001, "max": 0.05},
		"grid_min": {"min": -20, "max": -1},
		"grid_max": {"min": 1, "max": 20},
		"num_grids": {"min": 2, "max": 10},
	},
}

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Finetune KANG on Cora and MUTAG datasets for Node Classification, Link Prediction, and Graph Classification tasks.")
	parser.add_argument("--runs", type=int, default=200, help="Number of runs for each configuration")
	parser.add_argument("--nc", action="store_true", help="Perform Node Classification finetune")
	parser.add_argument("--lp", action="store_true", help="Perform Link Prediction finetune")
	parser.add_argument("--gc", action="store_true", help="Perform Graph Classification finetune")
	args = parser.parse_args()
	assert args.runs > 0, "Number of runs must be greater than zero"

	if args.nc:
		sweep_id = wandb.sweep(sweep=sweep_configuration, project="NodeClassification-FastKAN-Cora-sweep_v3_MovingCP")
		wandb.agent(sweep_id, lambda: main('nc'), count=args.runs)
		time.sleep(5) # <- wait before starting the next sweep
	if args.lp:
		sweep_configuration['metric']['name'] = 'val_auc'
		sweep_id = wandb.sweep(sweep=sweep_configuration, project="LinkPrediction-FastKAN-Cora-sweep_v3_MovingCP")
		wandb.agent(sweep_id, lambda: main('lp'), count=args.runs)
		time.sleep(5)
	if args.gc:
		sweep_configuration['metric']['name'] = 'val_acc'
		sweep_id = wandb.sweep(sweep=sweep_configuration, project="GraphClassification-FastKAN-Cora-sweep_v3_MovingCP")
		wandb.agent(sweep_id, lambda: main('gc'), count=args.runs)
