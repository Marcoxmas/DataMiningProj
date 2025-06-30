import time
import json
import argparse
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
from torch_geometric.datasets import Planetoid

from src.KANG import KANG
from src.utils import train, validate, test, set_seed

torch.backends.cudnn.deterministic	= True
torch.backends.cudnn.benchmark			= False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
starting_seed = 42

path_to_results = './experiments/ablation/'

def ablation(args):
  epochs = 1000
  patience = 300

  dataset	= Planetoid(root=f'./dataset/{args.dataset}', name=args.dataset)
  data 		= dataset[0].to(device)
  in_channels		= dataset.num_features
  out_channels 	= dataset.num_classes

  E = [True, False]
  T = [True, False]

  results = {
    "ET": {
      "avg_test_acc": 0,
      "std_test_acc": 0,
    },
    "E!T": {
      "avg_test_acc": 0,
      "std_test_acc": 0,
    },
    "G!T": {
      "avg_test_acc": 0,
      "std_test_acc": 0,
    },
    "GT": {
      "avg_test_acc": 0,
      "std_test_acc": 0,
    }
  }

  for e in E:
    for t in T:
      experiment_name = ("E" if e else "G") + ("T" if t else "!T")
      if experiment_name == "GT":
        continue # Skipping configuration GT, since the experiments have been already done
      print(F'Started experiment: {experiment_name}')
      best_model_path = f'./experiments/ablation/gkan_{experiment_name}.pth'
      test_accs = []
      epoch_times = []
      for run in tqdm(range(args.runs), desc="Runs", leave=False):
        set_seed(starting_seed + run)

        model = KANG(
          in_channels,
          32,
          out_channels,
          2,
          -15,
          20,
          4,
          0.1,
          device = device,
          linspace = e,
          trainable_grid = t
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=4e-4)
        patience_trigger = 0
        best_val_acc = 0

        for _ in range(epochs):
          epoch_start_time = time.time()
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
          epoch_end_time = time.time()
          epoch_total_time = round(epoch_end_time - epoch_start_time, 2)
          epoch_times.append(epoch_total_time)
        
        model.load_state_dict(torch.load(best_model_path))
        test_acc = test(model, data)
        test_accs.append(test_acc)
      
      results[experiment_name]["avg_test_acc"] = np.mean(test_accs)
      results[experiment_name]["std_test_acc"] = np.std(test_accs)
      results[experiment_name]["avg_epoch_time"] = np.mean(epoch_times)
      results[experiment_name]["std_epoch_time"] = np.std(epoch_times)
    
  # print(results)
  if args.dataset == 'Cora':
    results['GT']["avg_test_acc"] = 0.82
    results['GT']["std_test_acc"] = 0.007
  elif args.dataset == 'PubMed':
    results['GT']["avg_test_acc"] = 0.78
    results['GT']["std_test_acc"] = 0.008
  else:
    results['GT']["avg_test_acc"] = 0.704
    results['GT']["std_test_acc"] = 0.013
  results_fname = f'results_{args.dataset.lower()}.json'
  with open(path_to_results+results_fname, 'w') as f:
    json.dump(results, f)
  return results

def plot(dataset, results=None):
  if not results:
    results_fname = f'results_{dataset.lower()}.json'
    with open(path_to_results+results_fname, 'r') as f:
      results = json.load(f)

  configurations = ["ET", "E!T", "G!T", "GT"]

  test_accs = [results[config]['avg_test_acc'] for config in configurations]
  test_stds = [results[config]['std_test_acc'] for config in configurations]

  colors = {
      'ET': "#8ecae6",
      'E!T': "#219ebc",
      'G!T': "#ffb703",
      'GT': "#fb8500"
  }

  # Creazione della figura con un singolo subplot
  _, ax = plt.subplots(figsize=(10, 6))

  # Pannello: Test Accuracy
  ax.bar(
      configurations, test_accs, yerr=test_stds, capsize=5,
      color=[colors[config] for config in configurations]
  )
  ax.set_ylim(0, 1)
  ax.set_xlabel('Configuration')
  ax.set_ylabel('Test Accuracy')
  for i, v in enumerate(test_accs):
      ax.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=14)
  
  ax.set_xlabel('Configuration', fontsize=14) 
  ax.set_ylabel('Test Accuracy', fontsize=14)
  ax.tick_params(axis='x', which='major', labelsize=14)
  ax.tick_params(axis='y', which='major', labelsize=14)

  plt.tight_layout()
  plot_path = f'./experiments/ablation/ablation_{dataset.lower()}.pdf'
  plt.savefig(plot_path)
  print(f'[i] File saved in {plot_path}')
  return None

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Perform ablation study on the grid configuration")
  parser.add_argument("--runs", type=int, default=30, help="Number of runs for each configuration")
  parser.add_argument("--dataset", type=str, default="Cora", choices=["Cora", "CiteSeer", "PubMed"], help="Dataset name")
  parser.add_argument("--plot", action="store_true", help="Plot results without performing training")
  args = parser.parse_args()

  print(f'[i] Running on {device}')
  print(f'[i] Ablation study on {args.dataset}')

  start = time.time()
  if args.plot:
    plot(args.dataset)
  else:
    res = ablation(args)
    plot(args.dataset, results=res)
  
  total_time_end = round(time.time() - start, 2)
  print(f'[t] Completed in {total_time_end} seconds')