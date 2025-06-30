import os
import time
import json
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from torch_geometric.datasets import Planetoid
from torch_geometric.nn.conv import GCNConv, SAGEConv, GATv2Conv, GINConv

from src.KANG import KANG
from src.GenericGNN import GNN
from src.utils import set_seed, train, validate, test

# Set seed for reproducibility
starting_seed = 42

# ---------------------------
# Subgraph Creation Function
# ---------------------------
def create_subgraph(data, ratio):
    """
    Create a subgraph from the input data by randomly sampling a fraction (ratio) of nodes.
    It also reassigns train/val/test masks (60/20/20 split).
    """
    total_nodes = data.num_nodes
    num_keep = max(1, int(ratio * total_nodes))
    # Randomly select a subset of node indices
    perm = torch.randperm(total_nodes)[:num_keep]
    perm, _ = torch.sort(perm)  # sort indices for consistency

    # Create the subgraph (edge_index and edge_mask) with relabeling
    sub_edge_index, _ = subgraph(perm, data.edge_index, relabel_nodes=True)
    sub_x = data.x[perm]
    sub_y = data.y[perm]

    # Create new train/val/test masks for the subgraph (60/20/20 split)
    num_nodes_sub = num_keep
    train_ratio, val_ratio = 0.6, 0.2
    indices = torch.randperm(num_nodes_sub)
    train_end = int(train_ratio * num_nodes_sub)
    val_end = train_end + int(val_ratio * num_nodes_sub)

    train_mask = torch.zeros(num_nodes_sub, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes_sub, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes_sub, dtype=torch.bool)
    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True

    # Build new data object for the subgraph
    new_data = Data(x=sub_x, edge_index=sub_edge_index, y=sub_y)
    new_data.train_mask = train_mask
    new_data.val_mask = val_mask
    new_data.test_mask = test_mask
    new_data.num_nodes = num_nodes_sub  # update num_nodes
    return new_data

# ---------------------------
# Experiment Runner for a Single Architecture on a Subgraph
# ---------------------------
def run_experiment_subgraph(dataset, data, dataset_name, ratio, arch, epochs=200, patience=10, run=0):
    """
    Run training and evaluation for a given architecture (arch) on the provided subgraph.
    Returns a dictionary with the results.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    in_channels = dataset.num_features
    out_channels = dataset.num_classes

    # Configuration for each architecture
    node_config = {
      'gcn':  {'conv': GCNConv,   'hidden_channels': 64, 'dropout': 0.7, 'lr': 0.006,  'wd': 0.004,   'num_layers': 2},
      'gat':  {'conv': GATv2Conv, 'hidden_channels': 64, 'dropout': 0.6, 'lr': 0.01,   'wd': 0.0003,  'num_layers': 2},
      'sage': {'conv': SAGEConv,  'hidden_channels': 64, 'dropout': 0.3, 'lr': 0.01,   'wd': 0.002,   'num_layers': 2},
      'gin':  {'conv': GINConv,   'hidden_channels': 64, 'dropout': 0.2, 'lr': 0.006,  'wd': 0.0007,  'num_layers': 2},
      'kang': {'conv': None,      'hidden_channels': 32, 'dropout': 0.1, 'lr': 0.001,  'wd': 4e-4,    'grid_min': -15, 'grid_max': 20, 'num_grids': 4, 'num_layers': 2}
    }

    if arch not in node_config:
        raise ValueError(f"Architecture '{arch}' is not supported!")
    cfg = node_config[arch]
    
    if arch != 'kang':
        model = GNN(in_channels, cfg['hidden_channels'], out_channels,
                    cfg['num_layers'], cfg['conv'], cfg['dropout'])
        lr, wd = cfg['lr'], cfg['wd']
    else:
        model = KANG(
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
        lr, wd = cfg['lr'], cfg['wd']

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_val_acc = 0
    patience_trigger = 0
    epoch_times = []

    # Ensure output directory exists
    out_dir = './experiments/scalability'
    os.makedirs(out_dir, exist_ok=True)
    best_model_path = os.path.join(out_dir, f'{arch}.pt')

    for epoch in range(epochs):
        start_time = time.time()
        train(model, data, optimizer)
        val_acc, _ = validate(model, data)
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_trigger = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_trigger += 1

        if patience_trigger > patience:
            break

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(best_model_path))
    test_acc = test(model, data)

    return {
        'dataset': dataset_name,
        'ratio': ratio,
        'num_nodes': data.num_nodes,
        'architecture': arch,
        'epochs_run': len(epoch_times),
        'avg_epoch_time': sum(epoch_times) / len(epoch_times),
        'total_training_time': sum(epoch_times),
        'test_accuracy': test_acc
    }

# ---------------------------
# Main Experiment Loop
# ---------------------------
def main():
    # Define datasets, subgraph ratios, architectures, and number of runs per configuration
    dataset_names = ['Cora', 'CiteSeer', 'PubMed']
    ratios = [0.1, 0.3, 0.5, 0.7, 1.0]
    architectures = ['gcn', 'gat', 'sage', 'gin', 'kang']
    num_runs = 5  # number of runs per configuration
    aggregated_results = []  # to store aggregated metrics for each configuration

    for dataset_name in dataset_names:
        print(f"\nLoading dataset {dataset_name} ...")
        dataset = Planetoid(root=os.path.join('dataset', dataset_name), name=dataset_name)
        original_data = dataset[0]
        original_data.num_nodes = original_data.x.size(0)  # ensure num_nodes is defined

        for ratio in ratios:
            print(f"Creating subgraph for {dataset_name} with {int(ratio*100)}% of nodes ...")
            sub_data = create_subgraph(original_data, ratio)
            for arch in architectures:
                run_results = []
                print(f"Running configuration: {dataset_name} | Ratio: {int(ratio*100)}% | Architecture: {arch.upper()}")
                for run in tqdm(range(num_runs), desc='Runs', leave=False):
                    set_seed(starting_seed + run)
                    result = run_experiment_subgraph(dataset, sub_data, dataset_name, ratio, arch, epochs=1000, patience=300, run=run)
                    run_results.append(result)
                # Aggregate the results over runs
                test_accs = [r['test_accuracy'] for r in run_results]
                epoch_times = [r['avg_epoch_time'] for r in run_results]
                total_times = [r['total_training_time'] for r in run_results]
                epochs_runs = [r['epochs_run'] for r in run_results]
                num_nodes_list = [r['num_nodes'] for r in run_results]
                
                agg_result = {
                    'dataset': dataset_name,
                    'ratio': ratio,
                    'architecture': arch,
                    'num_nodes_avg': np.mean(num_nodes_list),
                    'test_accuracy_avg': np.mean(test_accs),
                    'test_accuracy_std': np.std(test_accs),
                    'avg_epoch_time_avg': np.mean(epoch_times),
                    'avg_epoch_time_std': np.std(epoch_times),
                    'total_training_time_avg': np.mean(total_times),
                    'total_training_time_std': np.std(total_times),
                    'epochs_run_avg': np.mean(epochs_runs),
                    'epochs_run_std': np.std(epochs_runs)
                }
                aggregated_results.append(agg_result)

    # Save aggregated results to a JSON file
    results_file = './experiments/scalability/scalability_results.json'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(aggregated_results, f, indent=4)

    # ---------------------------
    # Plotting the Results
    # ---------------------------
    # We regroup the aggregated results by dataset and by architecture.
    # X-axis: Subgraph Ratio; Y-axis: Metric (Test Accuracy or Average Epoch Time)
    # Each architecture is plotted with its fixed color.

    # Group results by dataset and architecture
    results_by_dataset_arch = {}
    for r in aggregated_results:
        ds = r['dataset']
        arch = r['architecture']
        if ds not in results_by_dataset_arch:
            results_by_dataset_arch[ds] = {}
        if arch not in results_by_dataset_arch[ds]:
            results_by_dataset_arch[ds][arch] = {
                'ratios': [],
                'test_accuracy_avg': [],
                'test_accuracy_std': [],
                'avg_epoch_time_avg': [],
                'avg_epoch_time_std': []
            }
        results_by_dataset_arch[ds][arch]['ratios'].append(r['ratio'])
        results_by_dataset_arch[ds][arch]['test_accuracy_avg'].append(r['test_accuracy_avg'])
        results_by_dataset_arch[ds][arch]['test_accuracy_std'].append(r['test_accuracy_std'])
        results_by_dataset_arch[ds][arch]['avg_epoch_time_avg'].append(r['avg_epoch_time_avg'])
        results_by_dataset_arch[ds][arch]['avg_epoch_time_std'].append(r['avg_epoch_time_std'])

    # Define a fixed order for architectures and their colors
    arch_order = ['gcn', 'gat', 'sage', 'gin', 'kang']
    colors = {
      'gcn': '#1f77b4',   # blue
      'gat': '#ff7f0e',   # orange
      'sage': '#2ca02c',  # green
      'gin': '#d62728',   # red
      'kang': '#9467bd'   # purple
    }

    # For each dataset, produce plots for test accuracy and average epoch time.
    for ds, arch_data in results_by_dataset_arch.items():
        # Plot: Test Accuracy vs. Subgraph Ratio
        plt.figure(figsize=(6, 4))
        for arch in arch_order:
            if arch in arch_data:
                # Get data and sort by ratio for a proper line plot
                ratios_arr = np.array(arch_data[arch]['ratios'])
                acc_arr = np.array(arch_data[arch]['test_accuracy_avg'])
                std_arr = np.array(arch_data[arch]['test_accuracy_std'])
                sort_idx = np.argsort(ratios_arr)
                ratios_sorted = ratios_arr[sort_idx]
                acc_sorted = acc_arr[sort_idx]
                std_sorted = std_arr[sort_idx]
                plt.plot(ratios_sorted, acc_sorted, marker='o', label=arch.upper(), color=colors[arch])
                plt.fill_between(ratios_sorted,
                                acc_sorted - std_sorted,
                                acc_sorted + std_sorted,
                                color=colors[arch],
                                alpha=0.2)
        plt.xlabel('Subgraph Ratio', fontsize=12)
        plt.ylabel('Test Accuracy', fontsize=12)
        # plt.title(f'{ds}: Test Accuracy vs. Subgraph Ratio', fontsize=14)
        plt.legend()
        plt.tight_layout()
        acc_plot_path = f'./experiments/scalability/{ds}_test_accuracy.pdf'
        plt.savefig(acc_plot_path, format='pdf')
        plt.close()

        # Plot: Average Epoch Time vs. Subgraph Ratio
        plt.figure(figsize=(6, 4))
        for arch in arch_order:
            if arch in arch_data:
                ratios_arr = np.array(arch_data[arch]['ratios'])
                time_arr = np.array(arch_data[arch]['avg_epoch_time_avg'])
                std_arr = np.array(arch_data[arch]['avg_epoch_time_std'])
                sort_idx = np.argsort(ratios_arr)
                ratios_sorted = ratios_arr[sort_idx]
                time_sorted = time_arr[sort_idx]
                std_sorted = std_arr[sort_idx]
                plt.plot(ratios_sorted, time_sorted, marker='o', label=arch.upper(), color=colors[arch])
                plt.fill_between(ratios_sorted,
                                time_sorted - std_sorted,
                                time_sorted + std_sorted,
                                color=colors[arch],
                                alpha=0.2)
        plt.xlabel('Subgraph Ratio', fontsize=12)
        plt.ylabel('Average Epoch Time (s)', fontsize=12)
        # plt.title(f'{ds}: Training Time vs. Subgraph Ratio', fontsize=14)
        plt.legend()
        plt.tight_layout()
        time_plot_path = f'./experiments/scalability/{ds}_epoch_time.pdf'
        plt.savefig(time_plot_path, format='pdf')
        plt.close()

    print(f"\nAll aggregated results saved to {results_file}.")
    print("Plots saved in './experiments/scalability/' for each dataset.")

if __name__ == '__main__':
    start_time = time.time()
    main()
    print(f'Ended all experiments in {(time.time()-start_time):.2f} (s)')