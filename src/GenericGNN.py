import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.models import MLP
from torch_geometric.nn.conv import GINConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops

# *-------------------------*
# | Genering GNN Definition |
# *-------------------------*
class GNN(nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels, num_layers, conv_layer, dropout, residuals=False):
		super(GNN, self).__init__()
		self.dropout = dropout
		self.residuals = residuals
		self.convs = nn.ModuleList()

		if conv_layer == GINConv:
			mlp = MLP([in_channels, hidden_channels])
			self.convs.append(GINConv(nn=mlp))
			for _ in range(num_layers - 1):
				mlp = MLP([hidden_channels, hidden_channels])
				self.convs.append(GINConv(nn=mlp))
			self.out_layer = MLP([hidden_channels, out_channels])
		else:
			self.convs.append(conv_layer(in_channels, hidden_channels))
			for _ in range(num_layers - 1):
				self.convs.append(conv_layer(hidden_channels, hidden_channels))
			self.out_layer = nn.Linear(hidden_channels, out_channels)

		self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])

	def forward(self, x, edge_index, batch=None):
		edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
		x = F.dropout(x, p=self.dropout, training=self.training)

		res = None
		for i, conv in enumerate(self.convs):
			if self.residuals and i > 0: res = x  
			x = conv(x, edge_index)
			x = F.relu(x)
			x = F.dropout(x, p=self.dropout, training=self.training)
			x = self.layer_norms[i](x)
			if self.residuals and i > 0: x += res

		if batch != None:
			x = global_mean_pool(x, batch) 
		x = self.out_layer(x)
		return F.log_softmax(x, dim=1)
	
	def encode(self, x, edge_index, batch=None):
		edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
		x = F.dropout(x, p=self.dropout, training=self.training)

		res = None
		for i, conv in enumerate(self.convs):
			if self.residuals and i > 0: res = x  
			x = conv(x, edge_index)
			x = F.relu(x)
			x = self.layer_norms[i](x)
			x = F.dropout(x, p=self.dropout, training=self.training)
			if self.residuals and i > 0: x += res

		if batch != None:
			x = global_mean_pool(x, batch) 

		return x
	
	def decode(self, z, edge_label_index):
		return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

	def decode_all(self, z):
		prob_adj = z @ z.t()
		return (prob_adj > 0).nonzero(as_tuple=False).t()