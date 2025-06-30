import time
import random
import numpy as np
from rich.table import Table
from rich.console import Console
from torchprofile import profile_macs
from sklearn.metrics import roc_auc_score

import torch
from torch import no_grad, cat
import torch.nn.functional as F

from torch_geometric.utils import negative_sampling

def set_seed(seed=42):
  np.random.seed(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  

  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

def pretty_print_model_profile(stats_dict):
    """
    Pretty print the model statistics dictionary.
    
    Args:
        stats_dict (dict): Dictionary containing model statistics.
    """

    console = Console()

    table = Table(title="Model Performance Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold magenta")
    table.add_column("Value", style="bold yellow")

    # Formatting the values for better readability
    formatted_values = {
        'total_parameters': f"{stats_dict['total_parameters']:,}",
        'trainable_parameters': f"{stats_dict['trainable_parameters']:,}",
        'flops': f"{stats_dict['flops']:,}",
        'macs': f"{stats_dict['macs']:,}",
        'avg_inference_time_ms': f"{stats_dict['avg_inference_time_ms']:.3f} ms",
        'max_memory_mb': f"{stats_dict['max_memory_mb']:.2f} MB"
    }

    for key, value in formatted_values.items():
        table.add_row(key.replace('_', ' ').title(), value)

    console.print(table)

def profile_model(model, x, edge_index, warm_up_iterations=5, measurement_iterations=10):
    """
    Profile a PyTorch model's FLOPs, MACs, parameters, inference time, and memory usage.
    Handles both CPU and GPU.
    """
    # Ensure model and inputs are on the same device
    device = next(model.parameters()).device if list(model.parameters()) else 'cpu'
    model.to(device)
    model.eval()
    
    # Warm-up with synchronization
    with torch.no_grad():
        for _ in range(warm_up_iterations):
            _ = model(x, edge_index)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    # Timing
    avg_time = 0.0
    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(measurement_iterations):
            _ = model(x, edge_index)
        end_event.record()
        torch.cuda.synchronize()
        avg_time = start_event.elapsed_time(end_event) / measurement_iterations
    else:
        start_time = time.perf_counter()
        for _ in range(measurement_iterations):
            _ = model(x, edge_index)
        avg_time = (time.perf_counter() - start_time) * 1000 / measurement_iterations  # ms
    
    # FLOPs/MACs (using ptflops)
    macs = profile_macs(model, (x, edge_index))
    flops = macs * 2
    
    # Memory (CUDA only)
    max_memory = None
    if device.type == 'cuda':
        max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    
    return {
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'flops': flops,
        'macs': macs,
        'avg_inference_time_ms': avg_time,
        'max_memory_mb': max_memory
    }

def save_list_to_file(lst, fname):
	with open(fname, 'w') as fout:
		for el in lst:
			fout.write(f'{str(el)}\n')

def confidence_interval(std, n):
	from math import sqrt
	return 1.96 * std / sqrt(n)

def train_link_predictor(model, train_data, val_data, optimizer, criterion, best_model_path, epochs=100, patience=10, verbose=False, info=None, log_freq=20):
  best_val_auc            = 0
  early_stopping_trigger  = 0

  # for epoch in tqdm(range(epochs), desc=f'{info['gnn'].upper()} on {info['dataset_name']} Run {info['run']+1}' if info != None else '', leave=False):
  for epoch in range(epochs):
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

    loss = criterion(out, edge_label)

    loss.backward()
    optimizer.step()

    val_auc = eval_link_predictor(model, val_data)
    if val_auc > best_val_auc:
      best_val_auc = val_auc
      early_stopping_trigger = 0
      torch.save(model.state_dict(), best_model_path)
    else:
      early_stopping_trigger += 1

    if early_stopping_trigger >= patience:
      break
    
    if verbose and (epoch % log_freq == 0 or epoch == epochs-1):
      print(f"Epoch: {epoch}, train Loss: {loss:.3f}, val AUC: {val_auc:.3f}")

  return best_val_auc

@no_grad()
def eval_link_predictor(model, data):
  model.eval()
  z 	= model.encode(data.x, data.edge_index)
  out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
  return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

def train(model, data, optimizer):
  model.train()
  optimizer.zero_grad()
  out     = model(data.x, data.edge_index)
  pred    = out.argmax(dim=1)
  correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
  acc     = int(correct) / int(data.train_mask.sum().item())
  loss    = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
  loss.backward()
  optimizer.step()
  return acc, loss.item()

@no_grad()
def test(model, data):
  model.eval()
  out     = model(data.x, data.edge_index)
  pred    = out.argmax(dim=1)
  correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
  acc     = int(correct) / int(data.test_mask.sum())
  return acc

@no_grad()
def validate(model, data):
  model.eval()
  out     = model(data.x, data.edge_index)
  pred    = out.argmax(dim=1)
  correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
  acc     = int(correct) / int(data.val_mask.sum())
  loss    = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
  return acc, loss.item()

def evaluate_node(model, data, idx, evaluator, device):
    model.eval()
    # Ensure data is on the correct device
    data = data.to(device)
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1, keepdim=True)
    # Ensure y_true is 2D
    y_true = data.y[idx]
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(1)
    input_dict = {"y_true": y_true.cpu(), "y_pred": pred[idx].cpu()}
    result = evaluator.eval(input_dict)
    return result["acc"]

def evaluate_link(model, data, split_edge, evaluator, device):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        pos_edge = split_edge['test']['edge'].to(device)
        if 'edge_neg' in split_edge['test']:
            neg_edge = split_edge['test']['edge_neg'].to(device)
        else:
            neg_edge = negative_sampling(edge_index=data.edge_index, num_nodes=data.x.size(0), num_neg_samples=pos_edge.size(1))
        pos_out = model.decode(z, pos_edge)
        neg_out = model.decode(z, neg_edge)
        pos_label = torch.ones(pos_out.size(0), device=device)
        neg_label = torch.zeros(neg_out.size(0), device=device)
        input_dict = {"y_true": torch.cat([pos_label, neg_label]).unsqueeze(1).cpu(),
                      "y_pred": torch.cat([pos_out, neg_out]).unsqueeze(1).cpu()}
        result = evaluator.eval(input_dict)
        return result["rocauc"] if "rocauc" in result else result

def evaluate_graph(model, loader, evaluator, device):
    model.eval()
    y_true = []
    y_pred = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.batch)
        y_true.append(data.y.cpu())
        y_pred.append(out.argmax(dim=1, keepdim=True).cpu())
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    result = evaluator.eval(input_dict)
    return result["acc"]

def compute_auc(preds, labels):
    """
    Computes the ROC-AUC score given prediction scores and true binary labels.

    Args:
        preds (torch.Tensor): Prediction scores (logits or probabilities).
        labels (torch.Tensor): Ground-truth binary labels.

    Returns:
        float: The computed ROC-AUC score.
    """
    # Convert tensors to NumPy arrays (detaching from the graph if necessary)
    preds_np = preds.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    return roc_auc_score(labels_np, preds_np)