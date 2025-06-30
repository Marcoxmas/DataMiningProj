import time
import json
import argparse
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn.conv import GCNConv, SAGEConv, GATv2Conv, GINConv

from src.KANG import KANG
from src.GenericGNN import GNN
from src.utils import train, validate, test, set_seed

torch.backends.cudnn.deterministic	= True
torch.backends.cudnn.benchmark			= False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
starting_seed = 42

epochs = 1000
patience = 300

config = {
  'gcn':  {'conv': GCNConv,   'hidden_channels': 64, 'dropout': 0.7, 'lr': 0.006, 'wd': 0.004,},
  'gat':  {'conv': GATv2Conv, 'hidden_channels': 64, 'dropout': 0.6, 'lr': 0.01, 'wd': 0.0003,},
  'sage': {'conv': SAGEConv,  'hidden_channels': 64, 'dropout': 0.3, 'lr': 0.01, 'wd': 0.002,},
  'gin':  {'conv': GINConv,   'hidden_channels': 64, 'dropout': 0.2, 'lr': 0.006, 'wd': 0.0007,},
  'kang': {'conv': None,      'hidden_channels': 32, 'dropout': 0.1, 'lr': 0.001, 'wd': 4e-4,  'grid_min': -15, 'grid_max': 20, 'num_grids': 4, 'num_layers': 2}
}

path_to_results = './experiments/oversmoothing/'

def dirichlet_energy(model, data):
  model.eval() 
  
  with torch.no_grad(): 
      embeddings = model.encode(data.x, data.edge_index)  
  
  row, col	= data.edge_index  
  energy		= 0.5*torch.norm(embeddings[row] - embeddings[col], dim=1).pow(2).sum()
  energy		/= row.size(0)
  
  return energy.item()  

def oversmoothing(args):
  gnn_convs   = ['gcn', 'gat', 'sage', 'gin', 'kang']
  layers_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

  # Init results dict
  results = {}
  for gnn in gnn_convs:
    results[gnn] = {}
    for layers in layers_list:
      results[gnn][layers] = {}
      results[gnn][layers]["avg_test_acc"] = 0
      results[gnn][layers]["std_test_acc"] = 0
      results[gnn][layers]["avg_energies"] = 0
      results[gnn][layers]["std_energies"] = 0

  dataset	      = Planetoid(root=f'./dataset/{args.dataset}', name=args.dataset)
  data 		      = dataset[0].to(device)
  in_channels		= dataset.num_features
  out_channels 	= dataset.num_classes

  for gnn in tqdm(gnn_convs, desc="GNNs"):
    best_model_path = f'{path_to_results}{gnn}.pth'
    for layers in tqdm(layers_list, desc="Layers", leave=False):
      run_test_accs = []
      run_energies = []
      for run in tqdm(range(args.runs), desc="Runs", leave=False):
        set_seed(starting_seed+run)
        if gnn != 'kang':
          cfg = config[gnn]
          lr, wd = cfg['lr'], cfg['wd']
          model = GNN(
            in_channels,
            cfg['hidden_channels'],
            out_channels,
            layers,
            cfg['conv'],
            cfg['dropout'],
            residuals=args.residuals
          )
        else:
          cfg = config[gnn]
          lr, wd = cfg['lr'], cfg['wd']
          model = KANG(
            in_channels,
            cfg['hidden_channels'],
            out_channels,
            layers,
            cfg['grid_min'],
            cfg['grid_max'],
            cfg['num_grids'],
            cfg['dropout'],
            device=device,
            residuals=args.residuals
          )

        model 							= model.to(device)
        optimizer 					= torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        best_val_acc 				= 0
        patience_trigger  	= 0

        for _ in range(epochs):
          train(model, data, optimizer)
          val_acc, _ = validate(model, data)
          if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_trigger = 0
            torch.save(model.state_dict(), best_model_path)
          else:
            patience_trigger += 1
          if patience_trigger > patience:
            break
        
        # After training for the number of layers let's test and compute the dirichlet energy
        test_acc = test(model, data)
        energy = dirichlet_energy(model, data)
        run_test_accs.append(test_acc)
        run_energies.append(energy)
      # for gnn, for layers, store avg and std of test accs and energy
      results[gnn][layers]["avg_test_acc"] = np.mean(run_test_accs)
      results[gnn][layers]["std_test_acc"] = np.std(run_test_accs)
      results[gnn][layers]["avg_energies"] = np.mean(run_energies)
      results[gnn][layers]["std_energies"] = np.std(run_energies)

  with open(f'{path_to_results}results_{args.dataset}_{args.residuals}.json', 'w') as f:
    json.dump(results, f)
  return results

def plot(args, results=None):
  # Load results if not provided
  if not results:
    with open(f'{path_to_results}results_{args.dataset}_{args.residuals}.json', 'r') as f:
      results = json.load(f)

  # Define architectures and their corresponding colors
  architectures = ['gcn', 'gat', 'sage', 'gin', 'kang']
  colors = {
    'gcn': '#1f77b4',   # blue
    'gat': '#ff7f0e',   # orange
    'sage': '#2ca02c',  # green
    'gin': '#d62728',   # red
    'kang': '#9467bd'   # purple
  }

  # The JSON keys are stored as strings, so convert layer keys to integers and sort them.
  sample_arch = architectures[0]
  layers_list = sorted([int(k) for k in results[sample_arch].keys()])

  # Create a figure with two subplots (side by side)
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

  for arch in architectures:
    avg_energies, std_energies = [], []
    avg_test_acc, std_test_acc = [], []
    # Iterate over sorted layers
    for l in layers_list:
      key = str(l)
      avg_energies.append(results[arch][key]["avg_energies"])
      std_energies.append(results[arch][key]["std_energies"])
      avg_test_acc.append(results[arch][key]["avg_test_acc"])
      std_test_acc.append(results[arch][key]["std_test_acc"])

    # Plot Dirichlet energy (left subplot)
    ax1.plot(layers_list, avg_energies, label=arch.upper(),
              color=colors[arch], marker='o', linewidth=2)
    ax1.fill_between(layers_list,
                      np.array(avg_energies) - np.array(std_energies),
                      np.array(avg_energies) + np.array(std_energies),
                      color=colors[arch], alpha=0.2)

    # Plot test accuracy (right subplot)
    ax2.plot(layers_list, avg_test_acc, label=arch.upper(),
              color=colors[arch], marker='o', linewidth=2)
    ax2.fill_between(layers_list,
                      np.array(avg_test_acc) - np.array(std_test_acc),
                      np.array(avg_test_acc) + np.array(std_test_acc),
                      color=colors[arch], alpha=0.2)

  # Customize left subplot (Dirichlet Energy)
  # ax1.set_title('Dirichlet Energy', fontsize=14)
  ax1.set_xlabel('Number of Layers', fontsize=12)
  ax1.set_ylabel('Dirichlet Energy', fontsize=12)
  ax1.spines['top'].set_visible(False)
  ax1.spines['right'].set_visible(False)
  # ax1.grid(True, linestyle='--', alpha=0.5)
  ax1.grid(False)
  ax1.set_xticks(layers_list)

  # Customize right subplot (Test Accuracy)
  # ax2.set_title('Test Accuracy', fontsize=14)
  ax2.set_xlabel('Number of Layers', fontsize=12)
  ax2.set_ylabel('Test Accuracy', fontsize=12)
  ax2.spines['top'].set_visible(False)
  ax2.spines['right'].set_visible(False)
  # ax2.grid(True, linestyle='--', alpha=0.5)
  ax2.grid(False)
  ax2.set_xticks(layers_list)

  # Create a common legend above the plots
  handles, labels = ax1.get_legend_handles_labels()
  fig.legend(handles, labels, loc='upper center', ncol=len(architectures), frameon=False, fontsize=12)

  plt.tight_layout(rect=[0, 0, 1, 0.93])
  path_fig = f'{path_to_results}oversmoothing_{args.dataset}_{args.residuals}.pdf'
  plt.savefig(path_fig)
  print(f'[i] Plot saved to {path_fig}')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Perform oversmoothing studys")
  parser.add_argument("--runs", type=int, default=10, help="Number of runs for each configuration")
  parser.add_argument("--dataset", type=str, default="Cora", choices=["Cora", "CiteSeer", "PubMed"], help="Dataset name")
  parser.add_argument("--residuals", action="store_true", help="Use residuals")
  parser.add_argument("--plot", action="store_true", help="Plot results without performing training")
  args = parser.parse_args()

  print(f'[i] Running on {device}')
  print(f'[i] Oversmoothing study on {args.dataset}')
  print(f'[i] Residuals={args.residuals}')

  start = time.time()
  if args.plot:
    plot(args)
  else:
    res = oversmoothing(args)
    plot(args, results=res)
  
  total_time_end = round(time.time() - start, 2)
  print(f'[t] Completed in {total_time_end} seconds')