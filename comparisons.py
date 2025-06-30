import warnings
warnings.filterwarnings('ignore')

import gc
import json
import argparse
import numpy as np
from time import time
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn.models import MLP
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Planetoid, TUDataset

torch.backends.cudnn.deterministic	= True
torch.backends.cudnn.benchmark			= False

# === GNN Layers ===
from src.KANG import KANG
from torch_geometric.nn.conv import GCNConv, SAGEConv, GATv2Conv, GINConv

# === Util Functions ===
from src.utils import profile_model, pretty_print_model_profile,\
	 	 									save_list_to_file, confidence_interval, \
											set_seed, train_link_predictor, eval_link_predictor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[i] Using device {device}')

epochs = 1000
patience = 300
# *-------------------------*
# | Genering GNN Definition |
# *-------------------------*
class GNN(nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels, num_layers, conv_layer, dropout):
		super(GNN, self).__init__()
		self.dropout = dropout
		self.convs = nn.ModuleList()

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

		for i, conv in enumerate(self.convs):
			x = conv(x, edge_index)
			x = F.relu(x)
			x = F.dropout(x, p=self.dropout, training=self.training)
			x = self.layer_norms[i](x)

		if batch != None:
			x = global_mean_pool(x, batch) 
		x = self.out_layer(x)
		return F.log_softmax(x, dim=1)
	
	def encode(self, x, edge_index, batch=None):
		edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
		x = F.dropout(x, p=self.dropout, training=self.training)

		for i, conv in enumerate(self.convs):
			x = conv(x, edge_index)
			x = F.relu(x)
			x = F.dropout(x, p=self.dropout, training=self.training)
			x = self.layer_norms[i](x)

		if batch != None:
			x = global_mean_pool(x, batch) 

		return x
	
	def decode(self, z, edge_label_index):
		return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

	def decode_all(self, z):
		prob_adj = z @ z.t()
		return (prob_adj > 0).nonzero(as_tuple=False).t()

# *---------------------*
# | Node Classification |
# *---------------------*
def node_classification(n_runs, seed):
	print('==> NODE CLASSIFICATION <==')
	gnn_convs = ['gcn', 'gat', 'sage', 'gin', 'kang']
	datasets	= ['Cora', 'PubMed', 'CiteSeer']

	# Hyperparameter configurations
	node_config = {
		'gcn':  {'conv': GCNConv,   'hidden_channels': 64, 'dropout': 0.7, 'lr': 0.006,  'wd': 0.004, 'num_layers': 2},
		'gat':  {'conv': GATv2Conv, 'hidden_channels': 64, 'dropout': 0.6, 'lr': 0.01, 'wd': 0.0003,  'num_layers': 2},
		'sage': {'conv': SAGEConv,  'hidden_channels': 64, 'dropout': 0.3, 'lr': 0.01,  'wd': 0.002, 'num_layers': 2},
		'gin':  {'conv': GINConv,   'hidden_channels': 64, 'dropout': 0.2, 'lr': 0.006, 'wd': 0.0007,  'num_layers': 2},
		'kang': {'conv': None,      'hidden_channels': 32, 'dropout': 0.1, 'lr': 0.001, 'wd': 4e-4,  'grid_min': -15, 'grid_max': 20, 'num_grids': 4, 'num_layers': 2}
	}

	for dataset_name in datasets:
		print(f'[i] Running experiments on {dataset_name}', flush=True)
		dataset	= Planetoid(root=f'./dataset/{dataset_name}', name=dataset_name)
		data 		= dataset[0].to(device)

		in_channels		= dataset.num_features
		out_channels 	= dataset.num_classes

		for gnn in gnn_convs:
			print(f'\t[+] Training {gnn.upper()}', flush=True)
			best_model_path = f'./comparisons/node_classification/models/{gnn}_{dataset_name}.pth'
			val_accuracies 	= []
			test_accuracies = []
			run_epoch_times = []

			for run in range(n_runs):
				set_seed(seed + run)

				# Initialise model based on configuration
				if gnn != 'kang':
					cfg = node_config[gnn]
					model = GNN(in_channels, cfg['hidden_channels'], out_channels,
												cfg['num_layers'], cfg['conv'], cfg['dropout'])
					lr, wd = cfg['lr'], cfg['wd']
				else:
					cfg			= node_config[gnn]
					lr, wd	= cfg['lr'], cfg['wd']
					model		= KANG(
						in_channels,
						cfg['hidden_channels'],
						out_channels,
						cfg['num_layers'],
						cfg['grid_min'],
						cfg['grid_max'],
						cfg['num_grids'],
						cfg['dropout'],
						device=device,
					)

				model 							= model.to(device)
				optimizer 					= torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
				best_val_acc 				= 0
				early_stop_counter	= 0
				epoch_times 				= []

				for _ in tqdm(range(epochs), desc=f'{gnn.upper()} on {dataset_name} Run {run+1}', leave=False):
					start_time = time()
					model.train()
					optimizer.zero_grad()
					out 	= model(data.x, data.edge_index)
					loss	= F.nll_loss(out[data.train_mask], data.y[data.train_mask])
					loss.backward()
					optimizer.step()
					epoch_times.append(time() - start_time)

					# Validation evaluation
					model.eval()
					val_acc = None
					with torch.no_grad():
						val_out		= model(data.x, data.edge_index)
						val_pred	= val_out.argmax(dim=1)
						correct		= (val_pred[data.val_mask] == data.y[data.val_mask]).sum().item()
						val_acc 	= correct / int(data.val_mask.sum())
					if val_acc > best_val_acc:
						best_val_acc = val_acc
						torch.save(model.state_dict(), best_model_path)
						early_stop_counter = 0
					else:
						early_stop_counter += 1
					if early_stop_counter >= patience:
						break

					# if epoch%20==0: print(f'|Run {run}|{epoch}/{epochs}| val_acc: {val_acc}, train_loss: {loss}')

				run_epoch_times.append(np.mean(epoch_times))
				# Load best model (selected on validation set) and evaluate on test set
				model.load_state_dict(torch.load(best_model_path))
				model.eval()
				test_acc = None
				with torch.no_grad():
					test_out 			= model(data.x, data.edge_index)
					test_pred 		= test_out.argmax(dim=1)
					correct_test	= (test_pred[data.test_mask] == data.y[data.test_mask]).sum().item()
					test_acc			= correct_test / int(data.test_mask.sum())
				test_accuracies.append(test_acc)
				val_accuracies.append(best_val_acc)

				# Memory management: clear GPU memory and garbage collect
				del model, optimizer
				torch.cuda.empty_cache()
				gc.collect()

			# Compute metrics and confidence intervals
			avg_val					= np.mean(val_accuracies)
			std_val 				= np.std(val_accuracies)
			ci_val					= confidence_interval(std_val, n_runs)
			avg_test				= np.mean(test_accuracies)
			std_test				= np.std(test_accuracies)
			ci_test					= confidence_interval(std_test, n_runs)
			avg_epoch_time 	= np.mean(run_epoch_times)
			std_epoch_time 	= np.std(run_epoch_times)
			ci_epoch				= confidence_interval(std_epoch_time, n_runs)

			# Save metrics to files
			save_list_to_file(test_accuracies, f'./comparisons/node_classification/results/{gnn}_{dataset_name}_test_accs.txt')
			save_list_to_file(run_epoch_times, f'./comparisons/node_classification/results/{gnn}_{dataset_name}_node_epoch_times.txt')

			print(f'\t[i] Validation acc: {avg_val:.3f}±{ci_val:.3f} (std: {std_val:.3f})')
			print(f'\t[i] Test acc: {avg_test:.3f}±{ci_test:.3f} (std: {std_test:.3f})')
			print(f'\t[i] Epoch time: {avg_epoch_time:.3f}±{ci_epoch:.3f} seconds (std: {std_epoch_time:.3f})')
			# Re-instantiate a dummy model for parameter counting
			if gnn != 'kang':
				cfg = node_config[gnn]
				dummy_model = GNN(in_channels, cfg['hidden_channels'], out_channels,
								  					cfg['num_layers'], cfg['conv'], cfg['dropout'])
			else:
				cfg = node_config[gnn]
				dummy_model = KANG(
					in_channels,
					cfg['hidden_channels'],
					out_channels,
					cfg['num_layers'],
					cfg['grid_min'],
					cfg['grid_max'],
					cfg['num_grids'],
					cfg['dropout'],
					device=device,
				)
			
			model_profile = profile_model(dummy_model.to(device), data.x, data.edge_index)
			# pretty_print_model_profile(model_profile)
			with open(f'./comparisons/node_classification/results/{gnn}_model_profile_{dataset_name}.json', 'w') as fout:
				json.dump(model_profile, fout)
			del dummy_model

# *-----------------*
# | Link Prediction |
# *-----------------*
def link_prediction(n_runs, seed):
	print('==> LINK PREDICTION <==')
	gnn_convs = ['gcn', 'gat', 'sage', 'gin', 'kang']
	datasets  = ['Cora', 'PubMed', 'CiteSeer']

	# Hyperparameter configuration for link prediction
	link_config = {
		'gcn':  {'conv': GCNConv,   'hidden_channels': 32, 'dropout': 0.6, 'lr': 0.01,  'wd': 0.002, 'num_layers': 2},
		'gat':  {'conv': GATv2Conv, 'hidden_channels': 32, 'dropout': 0.5, 'lr': 0.01, 'wd': 3e-4,  'num_layers': 2},
		'sage': {'conv': SAGEConv,  'hidden_channels': 64, 'dropout': 0.3, 'lr': 0.01,  'wd': 0.002, 'num_layers': 2},
		'gin':  {'conv': GINConv,   'hidden_channels': 64, 'dropout': 0.5, 'lr': 0.002, 'wd': 5e-4,  'num_layers': 2},
		'kang': {
			'conv': None, 'hidden_channels': 8, 'dropout': 0.1, 'lr': 0.008, 'wd': 0.0005,  'grid_min': -12, 'grid_max': 11, 'num_grids': 7, "num_layers": 2
			}
	}
	for dataset_name in datasets:
		print(f'[i] Running experiments on {dataset_name}', flush=True)
		dataset	= Planetoid(root=f'./dataset/{dataset_name}', name=dataset_name, transform=NormalizeFeatures())
		data		= dataset[0].to(device)

		in_channels		= dataset.num_features
		out_channels	= dataset.num_classes

		for gnn in gnn_convs:
			print(f'\t[+] Training {gnn.upper()}', flush=True)
			best_model_path = f'./comparisons/link_prediction/models/{gnn}_{dataset_name}.pth'
			test_aucs = []
			epoch_times_list = []

			for run in range(n_runs):
				set_seed(seed + run)

				if gnn != 'kang':
					cfg = link_config[gnn]
					model = GNN(in_channels, cfg['hidden_channels'], out_channels,
												cfg['num_layers'], cfg['conv'], cfg['dropout'])
					lr, wd = cfg['lr'], cfg['wd']
				else:
					cfg			= link_config[gnn]
					lr, wd	= cfg['lr'], cfg['wd']
					model		= KANG(
						in_channels,
						cfg['hidden_channels'],
						out_channels,
						cfg['num_layers'],
						cfg['grid_min'],
						cfg['grid_max'],
						cfg['num_grids'],
						cfg['dropout'],
						device = device,
					)
		
				model = model.to(device)
				optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
				criterion = nn.BCEWithLogitsLoss()

				# Split data for link prediction
				split = T.RandomLinkSplit(
					num_val=0.1,
					num_test=0.1,
					is_undirected=True,
					add_negative_train_samples=False,
					neg_sampling_ratio=1.0
				)
				train_data, val_data, test_data = split(data)

				start_time = time()
				info = {
					'gnn': gnn,
					'dataset_name': dataset_name,
					'run': run
				}
				# Train link predictor with early stopping based on validation AUC
				train_link_predictor(model, train_data, val_data, optimizer, criterion,
										best_model_path, epochs, patience, info=info)
				
				total_time = time() - start_time
				avg_epoch_time = total_time / epochs
				epoch_times_list.append(avg_epoch_time)

				model.load_state_dict(torch.load(best_model_path))
				test_auc = eval_link_predictor(model, test_data)
				test_aucs.append(test_auc)

				del model, optimizer
				torch.cuda.empty_cache()
				gc.collect()

			avg_test_auc 		= np.mean(test_aucs)
			std_test_auc 		= np.std(test_aucs)
			ci_test_auc			= confidence_interval(std_test_auc, n_runs)
			avg_epoch_time	= np.mean(epoch_times_list)
			std_epoch_time	= np.std(epoch_times_list)
			ci_epoch = confidence_interval(std_epoch_time, n_runs)

			save_list_to_file(test_aucs, f'./comparisons/link_prediction/results/{gnn}_{dataset_name}_test_aucs.txt')
			save_list_to_file(epoch_times_list, f'./comparisons/link_prediction/results/{gnn}_{dataset_name}_epoch_times.txt')

			print(f'\t\t[i] Test AUC: {avg_test_auc:.3f}±{ci_test_auc:.3f} (std: {std_test_auc:.3f})')
			print(f'\t\t[i] Epoch time: {avg_epoch_time:.3f}±{ci_epoch:.3f} seconds (std: {std_epoch_time:.3f})')
			# Dummy model for parameter count
			if gnn != 'kang':
				cfg = link_config[gnn]
				dummy_model = GNN(in_channels, cfg['hidden_channels'], out_channels,
														cfg['num_layers'], cfg['conv'], cfg['dropout'])
			else:
				cfg = link_config[gnn]
				dummy_model	= KANG(
							in_channels,
							cfg['hidden_channels'],
							out_channels,
							cfg['num_layers'],
							cfg['grid_min'],
							cfg['grid_max'],
							cfg['num_grids'],
							cfg['dropout'],
							device=device,
						)
				
			model_profile = profile_model(dummy_model.to(device), data.x, data.edge_index)
			# pretty_print_model_profile(model_profile)
			with open(f'./comparisons/link_prediction/results/{gnn}_model_profile_{dataset_name}.json', 'w') as fout:
					json.dump(model_profile, fout)
			del dummy_model

# *----------------------*
# | Graph Classification |
# *----------------------*
def graph_classification(n_runs, seed):
	print('==> GRAPH CLASSIFICATION <==')
	batch_size	= 64

	gnn_convs = ['gcn', 'gat', 'sage', 'gin', 'kang']
	gnn_convs = ['kang']
	datasets = ['MUTAG', 'PROTEINS']
	datasets = ['MUTAG']

	# Hyperparameter configurations for graph classification (example values)
	graph_config = {
		'gcn':  {'conv': GCNConv,   'hidden_channels': 64, 'dropout': 0.2, 'lr': 0.002,  'wd': 0.002, 'num_layers': 2},
		'gat':  {'conv': GATv2Conv, 'hidden_channels': 32, 'dropout': 0.1, 'lr': 0.004, 'wd': 0.004,  'num_layers': 2},
		'sage': {'conv': SAGEConv,  'hidden_channels': 16, 'dropout': 0.2, 'lr': 0.001,  'wd': 0.004, 'num_layers': 2},
		'gin':  {'conv': GINConv,   'hidden_channels': 16, 'dropout': 0.0, 'lr': 0.004, 'wd': 0.003,  'num_layers': 2},
		'kang': {
			'conv': None, 'hidden_channels': 32, 'dropout': 0.1, 'lr': 0.004, 'wd': 0.005,  'grid_min': -10, 'grid_max': 3, 'num_grids': 6, "num_layers": 2
		}
	}

	for dataset_name in datasets:
		print(f'[i] Running experiments on {dataset_name}', flush=True)
		dataset = TUDataset(root=f'./dataset/{dataset_name}', name=dataset_name)

		in_channels 	= dataset.num_node_features
		out_channels	= dataset.num_classes

		for gnn in gnn_convs:
			print(f'\t[+] Training {gnn.upper()}', flush=True)
			best_model_path = f'./comparisons/graph_classification/models/{gnn}_{dataset_name}.pth'
			test_accuracies = []
			run_epoch_times = []

			for run in range(n_runs):
				set_seed(seed + run)

				# Shuffle dataset after setting the seed for reproducibility
				shuffled_dataset = dataset.shuffle()
				train_size 		= int(0.8 * len(dataset))
				val_size 			= int(0.1 * len(dataset))
				train_dataset = shuffled_dataset[:train_size]
				val_dataset 	= shuffled_dataset[train_size:train_size + val_size]
				test_dataset 	= shuffled_dataset[train_size + val_size:]
				train_loader 	= DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
				val_loader 		= DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
				test_loader 	= DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

				if gnn != 'kang':
					cfg = graph_config[gnn]
					model = GNN(in_channels, cfg['hidden_channels'], out_channels,
												cfg['num_layers'], cfg['conv'], cfg['dropout'])
					lr, wd = cfg['lr'], cfg['wd']
				else:
					cfg			= graph_config[gnn]
					lr, wd	= cfg['lr'], cfg['wd']
					model		= KANG(
						in_channels,
						cfg['hidden_channels'],
						out_channels,
						cfg['num_layers'],
						cfg['grid_min'],
						cfg['grid_max'],
						cfg['num_grids'],
						cfg['dropout'],
						device=device,
					)

				model 							= model.to(device)
				optimizer	 					= torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
				criterion 					= nn.CrossEntropyLoss()
				best_val_acc 				= 0
				early_stop_counter 	= 0
				epoch_times					= []

				for _ in tqdm(range(epochs), desc=f'{gnn.upper()} on {dataset_name} Run {run+1}', leave=False):
					start_time = time()
					model.train()
					epoch_loss = 0
					for data in train_loader:
						optimizer.zero_grad()
						data 				= data.to(device)
						out 				= model(data.x, data.edge_index, data.batch)
						loss				= criterion(out, data.y)
						epoch_loss	+= loss.item()
						loss.backward()
						optimizer.step()
					epoch_times.append(time() - start_time)

					# Validation evaluation
					model.eval()
					correct = 0
					total 	= 0
					with torch.no_grad():
						for data in val_loader:
							data 		= data.to(device)
							out 		= model(data.x, data.edge_index, data.batch)
							pred 		= out.argmax(dim=1)
							correct	+= (pred == data.y).sum().item()
							total 	+= data.y.size(0)
					val_acc = correct / total if total > 0 else 0
					if val_acc > best_val_acc:
						best_val_acc = val_acc
						torch.save(model.state_dict(), best_model_path)
						early_stop_counter = 0
					else:
						early_stop_counter += 1
					if early_stop_counter >= patience:
						break

				run_epoch_times.append(np.mean(epoch_times))
				# Testing phase
				model.load_state_dict(torch.load(best_model_path))
				model.eval()
				correct = 0
				total		= 0
				with torch.no_grad():
					for data in test_loader:
						data 		= data.to(device)
						out			= model(data.x, data.edge_index, data.batch)
						pred 		= out.argmax(dim=1)
						correct	+= (pred == data.y).sum().item()
						total		+= data.y.size(0)
				test_acc = correct / total if total > 0 else 0
				test_accuracies.append(test_acc)

				del model, optimizer
				torch.cuda.empty_cache()
				gc.collect()

			avg_test 	= np.mean(test_accuracies)
			std_test 	= np.std(test_accuracies)
			ci_test		= confidence_interval(std_test, n_runs)
			avg_epoch_time = np.mean(run_epoch_times)
			std_epoch_time = np.std(run_epoch_times)
			ci_epoch = confidence_interval(std_epoch_time, n_runs)

			save_list_to_file(test_accuracies, f'./comparisons/graph_classification/results/{gnn}_{dataset_name}_test_accs.txt')
			save_list_to_file(run_epoch_times, f'./comparisons/graph_classification/results/{gnn}_{dataset_name}_epoch_times.txt')

			print(f'\t\t[i] Test acc: {avg_test:.3f}±{ci_test:.3f} (std: {std_test:.3f})')
			print(f'\t\t[i] Epoch time: {avg_epoch_time:.3f}±{ci_epoch:.3f} seconds (std: {std_epoch_time:.3f})')
			# Dummy model for parameter counting
			if gnn != 'kang':
				cfg = graph_config[gnn]
				dummy_model = GNN(in_channels, cfg['hidden_channels'], out_channels,
								  					cfg['num_layers'], cfg['conv'], cfg['dropout'])
			else:
				cfg = graph_config[gnn]
				dummy_model = KANG(
						in_channels,
						cfg['hidden_channels'],
						out_channels,
						cfg['num_layers'],
						cfg['grid_min'],
						cfg['grid_max'],
						cfg['num_grids'],
						cfg['dropout'],
						device=device,
					)
				
			model_profile = profile_model(dummy_model.to(device), data.x, data.edge_index)
			# pretty_print_model_profile(model_profile)
			with open(f'./comparisons/graph_classification/results/{gnn}_model_profile_{dataset_name}.json', 'w') as fout:
				json.dump(model_profile, fout)
			del dummy_model

def get_results(task):
	datasets = None
	folder_path	= './comparisons/'
	gnn_convs		= ['gcn', 'gat', 'sage', 'gin', 'kang']
	datasets		= ['CiteSeer', 'PubMed', 'Cora']
	if task == 'gc':
		folder_path += '/graph_classification/'
		datasets = ['MUTAG', 'PROTEINS']
	elif task == 'lp':
		folder_path += '/link_prediction/'
	else:
		folder_path += '/node_classification/'
	folder_path += 'results/'
		
	for dataset in datasets:
		for gnn in gnn_convs:
			model_path 	= folder_path + f'{gnn}_{dataset}_'
			epoch_times	= model_path + 'epoch_times.txt'
			test_accs 	= model_path + 'test_accs.txt'
			times = test = None
			with open(epoch_times, 'r') as f:
				times = [float(line.strip()) for line in f.readlines()]
			with open(test_accs, 'r') as f:
				test = [float(line.strip()) for line in f.readlines()]
			avg_test 	= np.mean(test)
			std_test 	= np.std(test)
			avg_times	= np.mean(times)
			std_times = np.std(times)
			print(f'{gnn} \t| {dataset} \t| Test Acc = {avg_test:.3f}+-{std_test:.3f} \t| Epoch Times = {avg_times:.3f}+-{std_times:.3f} (s) \t| on {len(test)} runs')

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Benchmark Datasets Experiments")
	parser.add_argument("--runs", type=int, default=30, help="Number of runs for each configuration")
	parser.add_argument("--nc", action="store_true", help="Perform Node Classification Comparisons")
	parser.add_argument("--lp", action="store_true", help="Perform Link Prediction Comparisons")
	parser.add_argument("--gc", action="store_true", help="Perform Graph Classification Comparisons")
	parser.add_argument("--results", action="store_true", help="Print already computed results")

	args = parser.parse_args()

	seed = 42

	if args.results:
		if args.gc: get_results('gc')
		if args.nc: get_results('nc')
		if args.lp: get_results('lp')
	else:
		total_time_start = time()
		if args.nc:
			start	= time()
			node_classification(args.runs, seed)
			end		= round(time() - start, 2)
			print(f'[t] Completed NODE CLASSIFICATION in {end} seconds')
		if args.lp:
			start	= time()
			link_prediction(args.runs, seed)
			end		= round(time() - start, 2)
			print(f'[t] Completed LINK PREDICTION in {end} seconds')
		if args.gc:
			start	= time()
			graph_classification(args.runs, seed)
			end		= round(time() - start, 2)
			print(f'[t] Completed GRAPH CLASSIFICATION in {end} seconds')

		total_time_end = round(time() - total_time_start, 2)
		print(f'[t] Completed all experiments in {total_time_end} seconds')