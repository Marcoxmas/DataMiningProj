import argparse
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from src.KANG import KANG
from src.utils import train_link_predictor, eval_link_predictor, set_seed

torch.backends.cudnn.deterministic	= True
torch.backends.cudnn.benchmark			= False
set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
		parser = argparse.ArgumentParser(description="GKAN - Link Prediction Example")
		parser.add_argument("--dataset_name", type=str, default="Cora", choices=["Cora", "CiteSeer", "PubMed"], help="Dataset name")
		parser.add_argument("--epochs", type=int, default=1000, help="Training epochs")
		parser.add_argument("--patience", type=int, default=300, help="Early stopping patience")
		parser.add_argument("--lr", type=float, default=0.008, help="Learning rate")
		parser.add_argument("--wd", type=float, default=0.0005, help="Weight decay")
		parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
		parser.add_argument("--hidden_channels", type=int, default=8, help="Hidden layer dimension")
		parser.add_argument("--layers", type=int, default=2, help="Number of GNN layers")
		parser.add_argument("--grid_min", type=int, default=-12, help="")
		parser.add_argument("--grid_max", type=int, default=11, help="")
		parser.add_argument("--num_grids", type=int, default=7, help="Number of control points")
		parser.add_argument("--log_freq", type=int, default=20, help="Logging frequency (epochs)")
		return parser.parse_args()

def link_prediction(args):
		dataset = Planetoid(root=f"./dataset/{args.dataset_name}", name=args.dataset_name, transform=NormalizeFeatures())
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
				device = device,
		).to(device)
		optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
		criterion = nn.BCEWithLogitsLoss()

		split = T.RandomLinkSplit(
			num_val=0.1,
			num_test=0.1,
			is_undirected=True,
			add_negative_train_samples=False,
			neg_sampling_ratio=1.0
		)
		train_data, val_data, test_data = split(data)

		best_model_path = f"./experiments/link_prediction/gkan.pth"

		best_val_auc = train_link_predictor(
				model, train_data, val_data, optimizer, criterion, 
				best_model_path, args.epochs, args.patience, verbose=True, 
				log_freq=args.log_freq
		)


		print(f"Best model saved with val_auc: {best_val_auc:.4f}")
		
		print(f"Loading best model from {best_model_path} for testing...")
		model.load_state_dict(torch.load(best_model_path))
		test_auc = eval_link_predictor(model, test_data)
		print(f"Test AUC: {test_auc:.4f}")


def main():
		args = get_args()
		link_prediction(args)

if __name__ == "__main__":
		main()