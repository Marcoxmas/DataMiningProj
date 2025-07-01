import argparse
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from toxcast_dataset import ToxCastGraphDataset
from hiv_dataset import HIVGraphDataset
from qm9_dataset import QM9GraphDataset

from src.KANG_regression import KANG
from src.utils import set_seed

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed = 42
set_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
	parser = argparse.ArgumentParser(description="GKAN - Graph Regression Example")
	parser.add_argument("--dataset_name", type=str, default="QM9", help="Dataset name")
	parser.add_argument("--target_column", type=str, default="mu", help="Target column for QM9 dataset (mu, alpha, homo, lumo, etc.)")
	parser.add_argument("--epochs", type=int, default=1000, help="Training epochs")
	parser.add_argument("--patience", type=int, default=300, help="Early stopping patience")
	parser.add_argument("--lr", type=float, default=0.004, help="Learning rate")
	parser.add_argument("--wd", type=float, default=0.005, help="Weight decay")
	parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
	parser.add_argument("--hidden_channels", type=int, default=32, help="Hidden layer dimension")
	parser.add_argument("--layers", type=int, default=2, help="Number of GNN layers")
	parser.add_argument("--num_grids", type=int, default=6, help="Number of splines")
	parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
	parser.add_argument("--grid_min", type=int, default=-10, help="")
	parser.add_argument("--grid_max", type=int, default=3, help="")
	parser.add_argument("--log_freq", type=int, default=10, help="Logging frequency (epochs)")
	return parser.parse_args()

def graph_regression(args):
	if args.dataset_name == "QM9":
		dataset_path = f'./dataset/{args.dataset_name}_{args.target_column}'
		dataset = QM9GraphDataset(root=dataset_path, target_column=args.target_column)
		print(f"QM9 dataset loaded with target column: {args.target_column}")
		dataset.print_dataset_info()

	shuffled_dataset = dataset.shuffle()
	train_size = int(0.8 * len(dataset))
	val_size = int(0.1 * len(dataset))
	train_dataset = shuffled_dataset[:train_size]
	val_dataset = shuffled_dataset[train_size:train_size + val_size]
	test_dataset = shuffled_dataset[train_size + val_size:]
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

	criterion = nn.MSELoss()

	# Modify output_dim to 1 for regression
	model = KANG(
		dataset.num_node_features,
		args.hidden_channels,
		1,  # Regression output
		args.layers,
		args.grid_min,
		args.grid_max,
		args.num_grids,
		args.dropout,
		device=device,
	).to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

	best_val_score = float("inf")
	early_stop_counter = 0
	best_epoch = -1
	best_model_path = f"./experiments/graph_regression/gkan.pth"

	for epoch in range(args.epochs):
		model.train()
		epoch_loss = 0
		for data in train_loader:
			optimizer.zero_grad()
			data = data.to(device)
			out = model(data.x, data.edge_index, data.batch).view(-1)
			loss = criterion(out, data.y.view(-1).float())
			epoch_loss += loss.item()
			loss.backward()
			optimizer.step()

		# Validation
		model.eval()
		total_loss = 0
		with torch.no_grad():
			for data in val_loader:
				data = data.to(device)
				out = model(data.x, data.edge_index, data.batch).view(-1)
				loss = criterion(out, data.y.view(-1).float())
				total_loss += loss.item()
		avg_val_loss = total_loss / len(val_loader)

		if avg_val_loss < best_val_score:
			best_val_score = avg_val_loss
			best_epoch = epoch
			torch.save(model.state_dict(), best_model_path)
			early_stop_counter = 0
		else:
			early_stop_counter += 1
		if early_stop_counter >= args.patience:
			break

		if epoch % args.log_freq == 0 or epoch == args.epochs - 1:
			print(f"Epoch {epoch:03d}: Train Loss: {epoch_loss:.4f}, Val MSE: {avg_val_loss:.4f}, Val RMSE: {math.sqrt(avg_val_loss):.4f}")

	print(f"\nBest model was saved at epoch {best_epoch} with val RMSE: {math.sqrt(best_val_score):.4f}")
	model.load_state_dict(torch.load(best_model_path))
	model.eval()

	# Test Evaluation
	total_loss = 0
	with torch.no_grad():
		for data in test_loader:
			data = data.to(device)
			out = model(data.x, data.edge_index, data.batch).view(-1)
			loss = criterion(out, data.y.view(-1).float())
			total_loss += loss.item()
	test_rmse = math.sqrt(total_loss / len(test_loader))
	print(f'Test RMSE: {test_rmse:.4f}')

def main():
	args = get_args()
	graph_regression(args)

if __name__ == "__main__":
	main()
