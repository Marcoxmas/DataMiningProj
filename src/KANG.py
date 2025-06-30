import torch.nn as nn

import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops

from torch_geometric.nn.models import MLP

from src.KANGConv import KANGConv
from src.KANLinear import KANLinear
from src.KAND import KAND

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
			linspace = False,
			trainable_grid = True,
			bsplines=False
		):
		super(KANG, self).__init__()
		self.dropout		= dropout
		self.residuals	= residuals
		self.convs			= nn.ModuleList()

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
				linspace = linspace,
				trainable_grid = trainable_grid,
				bsplines = bsplines
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
					linspace = linspace,
					trainable_grid = trainable_grid,
					bsplines = bsplines
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
				self.out_layer = KAND(
					hidden_channels,
					out_channels,
					grid_min,
					grid_max,
					num_grids,
					device=device,
					linspace = linspace,
					trainable_grid = trainable_grid
				)
		else:
			self.out_layer = MLP([hidden_channels, out_channels])

		self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_channels, elementwise_affine=False, bias=False) for _ in range(num_layers)])

	def forward(self, x, edge_index, batch=None):
		x.requires_grad_(True)
		edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
		x = F.dropout(x, p=self.dropout, training=self.training)

		res = None
		for i, conv in enumerate(self.convs):
			if self.residuals and i > 0: res = x  
			x = conv(x, edge_index)
			x = self.layer_norms[i](x)
			x = F.dropout(x, p=self.dropout, training=self.training)
			if self.residuals and i > 0: x += res

		# 2. Readout layer
		if batch != None:
			x = global_mean_pool(x, batch) 
		x = self.out_layer(x)

		return F.log_softmax(x, dim=1)
	
	def encode(self, x, edge_index, batch=None):
		x.requires_grad_(True)

		edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
		x = F.dropout(x, p=self.dropout, training=self.training)

		res = None
		for i, conv in enumerate(self.convs):
			if self.residuals and i > 0: res = x  
			x = F.dropout(x, p=self.dropout, training=self.training)
			x = conv(x, edge_index)
			x = self.layer_norms[i](x)
			if self.residuals and i > 0: x += res

		# 2. Readout layer
		if batch != None:
			x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

		return x
	
	def decode(self, z, edge_label_index):
		return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)  # product of a pair of nodes on each edge

	def decode_all(self, z):
		prob_adj = z @ z.t()
		return (prob_adj > 0).nonzero(as_tuple=False).t()