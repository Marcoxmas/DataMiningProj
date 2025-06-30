import argparse
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

from src.KANG import KANG
from src.utils import set_seed

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed = 42
set_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
	parser = argparse.ArgumentParser(description="GKAN - Graph Classification Example")
	parser.add_argument("--dataset_name", type=str, default="MUTAG", choices=["MUTAG", "PROTEINS"], help="Dataset name")
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

def graph_classification(args):
	dataset = TUDataset(root=f'./dataset/{args.dataset_name}', name=args.dataset_name)

	shuffled_dataset = dataset.shuffle()
	train_size 		= int(0.8 * len(dataset))
	val_size 			= int(0.1 * len(dataset))
	train_dataset = shuffled_dataset[:train_size]
	val_dataset 	= shuffled_dataset[train_size:train_size + val_size]
	test_dataset 	= shuffled_dataset[train_size + val_size:]
	train_loader 	= DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
	val_loader 		= DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
	test_loader 	= DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

	model = KANG(
		dataset.num_node_features,
		args.hidden_channels,
		dataset.num_classes,
		args.layers,
		args.grid_min,
		args.grid_max,
		args.num_grids,
		args.dropout,
		device=device,
	).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
	criterion = nn.CrossEntropyLoss()

	best_val_acc = 0
	early_stop_counter = 0
	best_epoch = -1
	best_model_path = f"./experiments/graph_classification/gkan.pth"


	for epoch in range(args.epochs):
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
			best_epoch = epoch
			best_val_acc = val_acc
			torch.save(model.state_dict(), best_model_path)
			early_stop_counter = 0
		else:
			early_stop_counter += 1
		if early_stop_counter >= args.patience:
			break

		if epoch % args.log_freq == 0 or epoch == args.epochs - 1:
			print(f"Epoch {epoch:03d}: Val Acc: {val_acc:.4f} ")

	# Load best model and evaluate on test set
	print(f"\nBest model was saved at epoch {best_epoch} with val acc: {best_val_acc:.4f}")
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

	print(f'Test Loss: Test Acc: {test_acc:.4f}')

	# # ---------------------------
	# # Plot and save the training metrics:
	# # ---------------------------
	# epochs_range = list(range(len(train_losses)))
	# fig, ax1 = plt.subplots(figsize=(8, 6))
	# ax2 = ax1.twinx()
	# ax1.plot(epochs_range, train_losses, "r-", label="Train Loss")
	# ax2.plot(epochs_range, train_accs, "b-", label="Train Accuracy")
	# ax1.set_xlabel("Epoch")
	# ax1.set_ylabel("Loss", color="r")
	# ax2.set_ylabel("Accuracy", color="b")
	# ax1.tick_params(axis="y", labelcolor="r")
	# ax2.tick_params(axis="y", labelcolor="b")
	# fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
	# plt.savefig("./experiments/graph_classification/images/train_plot.png", dpi=300)
	# plt.close(fig)

	# # ---------------------------
	# # Plot and save the validation metrics:
	# # ---------------------------
	# fig, ax1 = plt.subplots(figsize=(8, 6))
	# ax2 = ax1.twinx()
	# ax1.plot(epochs_range, val_losses, "r-", label="Val Loss")
	# ax2.plot(epochs_range, val_accs, "b-", label="Val Accuracy")
	# ax1.set_xlabel("Epoch")
	# ax1.set_ylabel("Loss", color="r")
	# ax2.set_ylabel("Accuracy", color="b")
	# ax1.tick_params(axis="y", labelcolor="r")
	# ax2.tick_params(axis="y", labelcolor="b")
	# fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
	# plt.savefig("./experiments/graph_classification/images/val_plot.png", dpi=300)
	# plt.close(fig)

def main():
	args = get_args()
	graph_classification(args)

if __name__ == "__main__":
	main()