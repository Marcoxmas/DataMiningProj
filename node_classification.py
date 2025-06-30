import torch
import argparse
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from src.KANG import KANG
from src.utils import train, validate, test, set_seed

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
set_seed(seed=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
	parser = argparse.ArgumentParser(description="GKAN - Node Classification Example")
	parser.add_argument("--dataset", type=str, default="Cora", choices=["Cora", "CiteSeer", "PubMed"], help="Dataset name")
	parser.add_argument("--epochs", type=int, default=1000, help="Training epochs")
	parser.add_argument("--patience", type=int, default=400, help="Early stopping patience")
	parser.add_argument("--lr", type=float, default=0.001, help="Learning rate") # <- KAN
	parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay") # <- KAN
	parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate") # <- KAN
	parser.add_argument("--hidden_channels", type=int, default=32, help="Hidden layer dimension") # <- KAN
	# parser.add_argument("--lr", type=float, default=0.008, help="Learning rate") # <- MLP
	# parser.add_argument("--wd", type=float, default=0.0001, help="Weight decay") # <- MLP
	# parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate") # <- MLP
	# parser.add_argument("--hidden_channels", type=int, default=64, help="Hidden layer dimension") # <- MLP
	parser.add_argument("--layers", type=int, default=2, help="Number of GNN layers")
	parser.add_argument("--num_grids", type=int, default=4, help="Number of control points")
	parser.add_argument("--grid_max", type=int, default=12, help="Max value range for control points")
	parser.add_argument("--grid_min", type=int, default=-5, help="Min value range for control points")
	parser.add_argument("--aggregation", type=str, default="mean", choices=["mean", "add", "max"], help="Node aggregation function")
	parser.add_argument("--log_freq", type=int, default=20, help="Logging frequency (epochs)")
	return parser.parse_args()

def node_classification(args):
	assert args.hidden_channels > 0, "hidden_channels must be positive"
	assert args.grid_min < args.grid_max, "grid_min must be less than grid_max"
	assert 0 <= args.dropout <= 1, "dropout must be between 0 and 1"
	assert args.lr > 0, "learning rate must be positive"
	assert args.wd >= 0, "weight decay must be non-negative"
	assert args.num_grids > 0, "num_grids must be positive"

	dataset = Planetoid(
		root=f"./dataset/{args.dataset}", 
		name=args.dataset, 
	)
		# transform=NormalizeFeatures()

	data = dataset[0].to(device)

	model = KANG(
		dataset.num_features,
		args.hidden_channels,
		dataset.num_classes,
		args.layers,
		args.grid_min,
		args.grid_max,
		args.num_grids,
		args.dropout,
		device=device,
		residuals=False,
		kan=True,
		linspace=False,
		trainable_grid=True
	).to(device)

	lr=args.lr
	weight_decay=args.wd

	optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

	best_val_acc = 0
	patience_trigger = 0
	best_epoch = 0
	best_model_path = "./experiments/node_classification/gkan.pth"

	train_accs = []
	train_losses = []
	val_accs = []
	val_losses = []

	for epoch in range(args.epochs):
		train_acc, train_loss = train(model, data, optimizer)
		val_acc, val_loss = validate(model, data)

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

		if patience_trigger > args.patience:
			print(f"[i] Early stopping triggered @ epoch {epoch}")
			break

		if epoch % args.log_freq == 0 or epoch == args.epochs - 1:
			print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

	print(f"Best model saved at epoch {best_epoch} with Val Acc: {best_val_acc:.4f}")
	
	print(f"Loading best model from {best_model_path} for testing...")
	model.load_state_dict(torch.load(best_model_path))
	test_acc = test(model, data)
	print(f"Test Acc: {test_acc:.4f}")

	# ---------------------------
	# Plot and save the training metrics:
	# ---------------------------
	epochs_range = list(range(len(train_losses)))
	fig, ax1 = plt.subplots(figsize=(8, 6))
	ax2 = ax1.twinx()
	ax1.plot(epochs_range, train_losses, "r-", label="Train Loss")
	ax2.plot(epochs_range, train_accs, "b-", label="Train Accuracy")
	ax1.set_xlabel("Epoch")
	ax1.set_ylabel("Loss", color="r")
	ax2.set_ylabel("Accuracy", color="b")
	ax1.tick_params(axis="y", labelcolor="r")
	ax2.tick_params(axis="y", labelcolor="b")
	fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
	plt.savefig("./experiments/node_classification/images/train_plot.png", dpi=300)
	plt.close(fig)

	# ---------------------------
	# Plot and save the validation metrics:
	# ---------------------------
	fig, ax1 = plt.subplots(figsize=(8, 6))
	ax2 = ax1.twinx()
	ax1.plot(epochs_range, val_losses, "r-", label="Val Loss")
	ax2.plot(epochs_range, val_accs, "b-", label="Val Accuracy")
	ax1.set_xlabel("Epoch")
	ax1.set_ylabel("Loss", color="r")
	ax2.set_ylabel("Accuracy", color="b")
	ax1.tick_params(axis="y", labelcolor="r")
	ax2.tick_params(axis="y", labelcolor="b")
	fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
	plt.savefig("./experiments/node_classification/images/val_plot.png", dpi=300)
	plt.close(fig)

def main():
	args = get_args()
	node_classification(args)

if __name__ == "__main__":
	main()