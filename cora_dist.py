import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops
from torch_geometric.datasets import Planetoid

from torch_geometric.nn.models import MLP

from src.KANGConv import KANGConv
from src.KANLinear import KANLinear
from KAND import FastKANLayer
from src.utils import set_seed, train

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

set_seed(42)

# Global variable to decide whether to apply LayerNorm or not
APPLY_LAYERNORM = True
DIM_INDEX = 18

class KANG(nn.Module):
	def __init__(self,
								in_channels, 
								hidden_channels, 
								out_channels, 
								num_layers, 
								grid_min=-1,
								grid_max=1,
								num_grids=2,
								dropout=0.0, 
								device='cpu', 
								aggr='mean',
								residuals=False,
								kan=True,
								linspace=False,
								trainable_grid=True,
								bsplines=False):
			super(KANG, self).__init__()
			self.dropout = dropout
			self.residuals = residuals
			self.convs = nn.ModuleList()

			# First Layer
			self.convs.append(
					KANGConv(
							in_channels, 
							hidden_channels,
							grid_min,
							grid_max,
							num_grids,
							device,
							aggr,
							kan,
							linspace=linspace,
							trainable_grid=trainable_grid,
							bsplines=bsplines
					)
			)

			# Subsequent Conv layers
			for _ in range(num_layers-1):
					self.convs.append(
							KANGConv(
									hidden_channels, 
									hidden_channels,
									grid_min,
									grid_max,
									num_grids,
									device,
									aggr,
									kan,
									linspace=linspace,
									trainable_grid=trainable_grid,
									bsplines=bsplines
							)
					)

			# Readout Layer
			if kan:
					if bsplines:
							self.out_layer = KANLinear(
									hidden_channels,
									out_channels,
									grid_size=num_grids,
									grid_range=[grid_min, grid_max]
							)
					else:
							self.out_layer = FastKANLayer(
									hidden_channels,
									out_channels,
									grid_min,
									grid_max,
									num_grids,
									device=device,
									linspace=linspace,
									trainable_grid=trainable_grid
							)
			else:
					self.out_layer = MLP([hidden_channels, out_channels])

			# Create a LayerNorm for each conv layer (without learnable parameters)
			self.layer_norms = nn.ModuleList([
					nn.LayerNorm(hidden_channels, elementwise_affine=False, bias=False) 
					for _ in range(num_layers)
			])

	def forward(self, x, edge_index, batch=None):
			x.requires_grad_(True)
			edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
			x = F.dropout(x, p=self.dropout, training=self.training)

			res = None
			for i, conv in enumerate(self.convs):
					if self.residuals and i > 0:
							res = x  
					x = conv(x, edge_index)
					# Conditionally apply LayerNorm based on the global variable
					if APPLY_LAYERNORM:
							x = self.layer_norms[i](x)
					x = F.dropout(x, p=self.dropout, training=self.training)
					if self.residuals and i > 0:
							x += res

			if batch is not None:
					x = global_mean_pool(x, batch)
			x = self.out_layer(x)
			return F.log_softmax(x, dim=1)
		

cfg = {'hidden_channels': 32, 'dropout': 0.1, 'lr': 0.001, 'wd': 4e-4,  
			 'grid_min': -15, 'grid_max': 20, 'num_grids': 4, 'num_layers': 2}

def main():
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		if not torch.backends.mps.is_available():
				if not torch.backends.mps.is_built():
						print("MPS not available because the current PyTorch install was not "
									"built with MPS enabled.")
				else:
						print("MPS not available because the current MacOS version is not 12.3+ "
									"and/or you do not have an MPS-enabled device on this machine.")

		else:
				device = torch.device("mps")

		print(f'[i] Using device {device}')
		dataset_name = 'Cora'
		dataset = Planetoid(root=f'./dataset/{dataset_name}', name=dataset_name)
		data = dataset[0].to(device)

		in_channels = dataset.num_features
		out_channels = dataset.num_classes

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

		# Train for 300 epochs
		for _ in tqdm(range(1), leave=False):
				train(model, data, optimizer)

		# Set the model to evaluation mode
		model.eval()
		with torch.no_grad():
				# Use the original features and edge_index
				x = data.x.clone()
				edge_index = data.edge_index
				
				# Get output after the first convolution layer
				out_conv = model.convs[0](x, edge_index)
				# If APPLY_LAYERNORM is True, apply the corresponding layer norm manually
				if APPLY_LAYERNORM:
						out_conv = model.layer_norms[0](out_conv)

		# Instead of flattening all dimensions, select only one dimension (column)
		out_conv_single_dim = out_conv[:, DIM_INDEX].cpu().detach().numpy()

		# Print summary statistics for the selected dimension
		print(f"Distribution of feature dimension {DIM_INDEX} after first layer (LayerNorm={APPLY_LAYERNORM}):")
		print(f"Mean: {out_conv_single_dim.mean():.4f}, Std: {out_conv_single_dim.std():.4f}, "
					f"Min: {out_conv_single_dim.min():.4f}, Max: {out_conv_single_dim.max():.4f}\n")

		# Create the filename based on whether LayerNorm is applied
		filename = ("./experiments/dist/first_layer_with_layernorm_dim0.pdf" 
								if APPLY_LAYERNORM 
								else "./experiments/dist/first_layer_without_layernorm_dim0.pdf")

		# Plot the histogram with purple bins (alpha=0.2)
		plt.figure(figsize=(8, 6))
		counts, bins, patches = plt.hist(out_conv_single_dim, bins=50, color='purple', alpha=0.2, edgecolor='black')
		plt.xlabel("Feature Value", fontsize=12)
		plt.ylabel("Frequency", fontsize=12)
		
		# Determine the range for the red dots based on the data of the selected dimension
		data_min = out_conv_single_dim.min()
		data_max = out_conv_single_dim.max()
		mean_val = out_conv_single_dim.mean()
		std_val = out_conv_single_dim.std()
		N_dots = cfg['num_grids']  # Number of red dots to add

		if not APPLY_LAYERNORM:
				# Evenly spaced red dots across the data range
				red_dot_x = np.linspace(data_min, data_max, N_dots)
		else:
				# Red dots sampled from a Gaussian distribution with the same mean and std
				# red_dot_x = np.random.normal(loc=mean_val, scale=std_val, size=N_dots)
				red_dot_x = mean_val + std_val * torch.randn(N_dots)  
				red_dot_x = torch.clamp(red_dot_x, min=cfg['grid_min'], max=cfg['grid_max'])  
				red_dot_x, _ = torch.sort(red_dot_x)  
				# red_dot_x = np.clip(red_dot_x, data_min, data_max)
		
		# Place the red dots slightly above the bottom of the current y-axis range
		ax = plt.gca()
		y_min, y_max = ax.get_ylim()
		red_dot_y = np.full_like(red_dot_x, y_min + 0.05 * (y_max - y_min))
		
		plt.scatter(red_dot_x, red_dot_y, color='#cb1517', zorder=3)
		
		plt.savefig(filename, format='pdf')
		plt.close()
		print(f"Saved '{filename}'.")

if __name__ == '__main__':
		main()