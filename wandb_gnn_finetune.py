import time
import wandb
import argparse
import numpy as np

import torch
from torch import cat
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import MLP
from torch_geometric.nn import global_mean_pool
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.utils import add_self_loops, negative_sampling
from torch_geometric.nn.conv import GCNConv, SAGEConv, GATv2Conv, GINConv

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from src.utils import set_seed, train, validate, test, eval_link_predictor

torch.backends.cudnn.deterministic	= True
torch.backends.cudnn.benchmark			= False
set_seed(seed=42)

gnn_mapping = {
		"gcn": GCNConv,
		"gat": GATv2Conv,
		"gin": GINConv,
		"sage": SAGEConv
}

# *-------------------------*
# | Genering GNN Definition |
# *-------------------------*
class GNN(nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, conv_layer, residuals=False):
		super(GNN, self).__init__()
		self.dropout = dropout
		self.convs = nn.ModuleList()
		self.residuals = residuals

		if conv_layer == GINConv:
			mlp = MLP([in_channels, hidden_channels])
			self.convs.append(GINConv(nn=mlp))
			for _ in range(num_layers - 1):
				mlp = MLP([hidden_channels, hidden_channels])
				self.convs.append(GINConv(nn=mlp))
			self.out_layer = MLP([hidden_channels, out_channels])
		else:
			self.convs.append(conv_layer(in_channels, hidden_channels))
			for _ in range(num_layers - 1):
				self.convs.append(conv_layer(hidden_channels, hidden_channels))
			self.out_layer = nn.Linear(hidden_channels, out_channels)

		self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])

	def forward(self, x, edge_index, batch=None):
		edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
		x = F.dropout(x, p=self.dropout, training=self.training)
		res = None
		for i, conv in enumerate(self.convs):
			if i > 0 and self.residuals: res = x
			x = conv(x, edge_index)
			x = F.relu(x)
			x = self.layer_norms[i](x)
			x = F.dropout(x, p=self.dropout, training=self.training)
			if i > 0 and self.residuals: x += res
		
		if batch != None:
			x = global_mean_pool(x, batch) 
		x = self.out_layer(x)
		return F.log_softmax(x, dim=1)
	
	def encode(self, x, edge_index, batch=None):
		edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
		x = F.dropout(x, p=self.dropout, training=self.training)
		res = None
		for i, conv in enumerate(self.convs):
			if i > 0 and self.residuals: res = x
			x = conv(x, edge_index)
			x = F.relu(x)
			x = self.layer_norms[i](x)
			x = F.dropout(x, p=self.dropout, training=self.training)
			if i > 0 and self.residuals: x += res
		
		if batch != None:
			x = global_mean_pool(x, batch)

		return x
	
	def decode(self, z, edge_label_index):
		return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)  # product of a pair of nodes on each edge

	def decode_all(self, z):
		prob_adj = z @ z.t()
		return (prob_adj > 0).nonzero(as_tuple=False).t()

def validate_config(config):
	"""Validate configuration parameters."""
	assert 0 <= config.dropout <= 1, "dropout must be between 0 and 1"
	assert config.lr > 0, "learning rate must be positive"
	assert config.wd >= 0, "weight decay must be non-negative"

def nc(gnn_conv, config, dataset, epochs, patience):
	"""Training objective function with improved error handling and monitoring."""
	try:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		data = dataset[0].to(device)
		best_model_path = "./experiments/node_classification/gkan.pth"
		# Validate configuration
		validate_config(config)
		
		# Initialize model
		model = GNN(
			dataset.num_features,
			config.hidden_channels,
			dataset.num_classes,
			2, # layers
			config.dropout,
			gnn_conv,
			residuals=config.residuals,
		).to(device)
		
		optimizer 				= torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)
		best_val_acc 			= 0
		patience_trigger	= 0
		
		for epoch in range(epochs):
			# Training step
			_, train_loss 		= train(model, data, optimizer)
			val_acc, val_loss	= validate(model, data)
			
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

def lp(gnn_conv, config, dataset, epochs, patience):
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

		model = GNN(
			dataset.num_features,
			config.hidden_channels,
			1, # 1 output channel for link prediction
			2, # layers
			config.dropout,
			gnn_conv,
			residuals=config.residuals,
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
				wandb.run.summary["best_epoch"] = epoch
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

# def gc(gnn_conv, config, dataset, epochs, patience):
# 	validate_config(config)
# 	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 	best_model_path = "./experiments/graph_classification/gkan.pth"

# 	num_features 			= dataset.num_node_features
# 	out_channels 			= dataset.num_classes
# 	batch_size 				= 64
# 	shuffled_dataset	= dataset.shuffle()
# 	train_size 				= int(0.8 * len(dataset))
# 	val_size 					= int(0.1 * len(dataset))
# 	train_dataset 		= shuffled_dataset[:train_size]
# 	val_dataset 			= shuffled_dataset[train_size:train_size + val_size]
# 	test_dataset 			= shuffled_dataset[train_size + val_size:]
# 	train_loader 			= DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# 	val_loader 				= DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# 	test_loader 			= DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 	model = GNN(
# 		num_features,
# 		config.hidden_channels,
# 		out_channels,
# 		2, # layers
# 		config.dropout,
# 		gnn_conv,
# 		residuals=config.residuals,
# 	).to(device)

# 	optimizer				 	= torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)
# 	criterion 				= torch.nn.CrossEntropyLoss()
# 	best_val_acc			= 0
# 	patience_trigger	= 0

# 	for epoch in range(epochs):
# 		model.train()
# 		train_acc 	= 0
# 		train_loss	= 0
# 		for data in train_loader:
# 			data = data.to(device)
# 			optimizer.zero_grad()
# 			out 				= model(data.x, data.edge_index, data.batch)
# 			loss				= criterion(out, data.y)
# 			pred 				= out.argmax(dim=1)
# 			train_acc 	+= int((pred == data.y).sum())
# 			train_loss	+= loss.item()
# 			loss.backward()
# 			optimizer.step()
# 		train_acc 	/= len(train_loader.dataset)
# 		train_loss	/= len(train_loader.dataset)

# 		model.eval()
# 		val_acc = 0
# 		val_loss = 0
# 		with torch.no_grad():
# 			for data in val_loader:
# 				data = data.to(device)
# 				out = model(data.x, data.edge_index, data.batch)
# 				loss = criterion(out, data.y)
# 				val_loss += loss.item()
# 				pred = out.argmax(dim=1)
# 				val_acc += int((pred == data.y).sum())

# 		val_acc		/= len(val_loader.dataset)
# 		val_loss	/= len(val_loader.dataset)

# 		if val_acc > best_val_acc:
# 			best_val_acc = val_acc
# 			patience_trigger = 0
# 			wandb.run.summary["best_val_acc"] = best_val_acc
# 			wandb.run.summary["best_epoch"] = epoch
# 			torch.save(model.state_dict(), best_model_path)
# 		else:
# 			patience_trigger += 1
# 		if patience_trigger >= patience:
# 			break

# 		wandb.log({
# 			"epoch": epoch,
# 			"train_acc": train_acc,
# 			"train_loss": train_loss,
# 			"val_acc": val_acc,
# 			"val_loss": val_loss,
# 			"best_val_acc": best_val_acc,
# 			"learning_rate": config.lr,
# 			"dropout": config.dropout,
# 			"weight_decay": config.wd,
# 		})
	
# 	model.load_state_dict(torch.load(best_model_path))
# 	model.eval()
# 	test_acc = 0
# 	with torch.no_grad():
# 		for data in test_loader:
# 			data = data.to(device)
# 			out = model(data.x, data.edge_index, data.batch)
# 			pred = out.argmax(dim=1)
# 			test_acc += int((pred == data.y).sum())
# 	test_acc /= len(test_loader.dataset)

# 	wandb.log({"test_acc": test_acc})
# 	return best_val_acc		

def gc(gnn_conv, config, dataset, epochs, patience):
		validate_config(config)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		num_features = dataset.num_node_features
		out_channels = dataset.num_classes
		batch_size = 32

		# Extract labels from each graph (assumes each data.y is a tensor with one integer)
		labels = np.array([data.y.item() for data in dataset])
		
		n_splits = 5  # number of folds
		skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
		
		fold_test_accuracies = []
		fold_val_accs = []
		
		# Loop over folds
		for fold, (train_val_idx, test_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels)):
			# Further split train_val_idx into training and validation sets.
			# We want ~10% of the overall dataset as validation.
			train_val_labels = labels[train_val_idx]
			sss = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=config.seed)
			train_idx_rel, val_idx_rel = next(iter(sss.split(np.zeros(len(train_val_idx)), train_val_labels)))
			actual_train_idx = train_val_idx[train_idx_rel]
			actual_val_idx   = train_val_idx[val_idx_rel]
			
			# Create the respective subsets
			train_dataset = [dataset[i] for i in actual_train_idx]
			val_dataset   = [dataset[i] for i in actual_val_idx]
			test_dataset  = [dataset[i] for i in test_idx]
			
			train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
			val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
			test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
			
			# Initialize model for this fold
			model = GNN(
				num_features,
				config.hidden_channels,
				out_channels,
				config.layers, # layers
				config.dropout,
				gnn_conv,
			).to(device)
			
			optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)
			criterion = torch.nn.CrossEntropyLoss()
			best_val_acc = 0
			patience_trigger = 0
			
			# Save best model separately for each fold
			best_fold_model_path = f"./experiments/graph_classification/gkan_fold{fold}.pth"
			
			for epoch in range(epochs):
				model.train()
				train_acc = 0
				train_loss = 0
				for data in train_loader:
					data = data.to(device)
					optimizer.zero_grad()
					out = model(data.x, data.edge_index, data.batch)
					loss = criterion(out, data.y)
					pred = out.argmax(dim=1)
					train_acc += int((pred == data.y).sum())
					train_loss += loss.item()
					loss.backward()
					optimizer.step()
				train_acc /= len(train_loader.dataset)
				train_loss /= len(train_loader.dataset)
				
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
				val_acc /= len(val_loader.dataset)
				val_loss /= len(val_loader.dataset)
				
				if val_acc > best_val_acc:
					best_val_acc = val_acc
					patience_trigger = 0
					wandb.run.summary[f"best_val_acc_fold_{fold}"] = best_val_acc
					wandb.run.summary[f"best_epoch_fold_{fold}"] = epoch
					torch.save(model.state_dict(), best_fold_model_path)
				else:
					patience_trigger += 1
				if patience_trigger >= patience:
					break
				
				wandb.log({
					"fold": fold,
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
			
			# Load the best model for this fold and evaluate on the test set.
			model.load_state_dict(torch.load(best_fold_model_path))
			model.eval()
			test_acc = 0
			with torch.no_grad():
				for data in test_loader:
					data = data.to(device)
					out = model(data.x, data.edge_index, data.batch)
					pred = out.argmax(dim=1)
					test_acc += int((pred == data.y).sum())
			test_acc /= len(test_loader.dataset)
			wandb.log({f"test_acc_fold_{fold}": test_acc})
			# print(f"Fold {fold+1} Test Accuracy: {test_acc:.4f}")
			
			fold_test_accuracies.append(test_acc)
			fold_val_accs.append(best_val_acc)
		
		avg_test_acc = np.mean(fold_test_accuracies)
		avg_val_acc  = np.mean(fold_val_accs)
		wandb.log({"avg_test_acc": avg_test_acc, "avg_val_acc": avg_val_acc})
		# print(f"Average Test Accuracy over {n_splits} folds: {avg_test_acc:.4f}")
		return avg_val_acc
	
def main(task, gnn_name):
	"""Main function with improved resource management."""
	run = wandb.init(
		config={
			"epochs": 500,
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
		val_acc = nc(gnn_mapping[gnn_name], run.config, dataset, run.config.epochs, run.config.patience)
		wandb.log({"final_val_acc": val_acc})
	if task == 'lp':
		val_auc = lp(gnn_mapping[gnn_name], run.config, dataset, run.config.epochs, run.config.patience)
		wandb.log({"final_auc": val_auc})
	if task == 'gc':
		dataset = TUDataset(root='./dataset/MUTAG', name='MUTAG')
		val_acc = gc(gnn_mapping[gnn_name], run.config, dataset, run.config.epochs, run.config.patience)
		wandb.log({"final_val_acc": val_acc})

# Start the sweep
sweep_configuration = {
	"method": "bayes",
	"metric": {"goal": "maximize", "name": "val_acc"},
	"parameters": {
		"hidden_channels": {"values": [8, 16, 32, 64]},
		"layers": {"values": [2, 3, 4]},
		"dropout": {"min": 0.0, "max": 0.7},
		"lr": {"min": 0.0001, "max": 0.05}, 
		"wd": {"min": 0.0001, "max": 0.05},
	},
}

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Finetune GNNs on Cora and MUTAG datasets for Node Classification, Link Prediction, and Graph Classification tasks.")
	parser.add_argument("--runs", type=int, default=200, help="Number of runs for each configuration")
	parser.add_argument("--gnn", type=str, default="GCN", choices=["GCN", "GAT", "GIN", "SAGE"], help="Graph convolution layer name")
	parser.add_argument("--nc", action="store_true", help="Perform Node Classification finetune")
	parser.add_argument("--lp", action="store_true", help="Perform Link Prediction finetune")
	parser.add_argument("--gc", action="store_true", help="Perform Graph Classification finetune")
	args = parser.parse_args()
	assert args.runs > 0, "Number of runs must be greater than zero"

	gnn_name = args.gnn.lower()

	if args.nc:
		sweep_id = wandb.sweep(sweep=sweep_configuration, project=f"NodeClassification-{gnn_name}-Cora")
		wandb.agent(sweep_id, lambda: main('nc', gnn_name), count=args.runs)
		time.sleep(5) # <- wait before starting the next sweep
	if args.lp:
		sweep_configuration['metric']['name'] = 'val_auc'
		sweep_id = wandb.sweep(sweep=sweep_configuration, project=f"LinkPrediction-{gnn_name}-Cora")
		wandb.agent(sweep_id, lambda: main('lp', gnn_name), count=args.runs)
		time.sleep(5)
	if args.gc:
		sweep_configuration['metric']['name'] = 'val_acc'
		sweep_id = wandb.sweep(sweep=sweep_configuration, project=f"GraphClassification-{gnn_name}-MUTAG")
		wandb.agent(sweep_id, lambda: main('gc', gnn_name), count=args.runs)
