import math
from typing import Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import spmm, add_self_loops
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor


class KANLinear(torch.nn.Module):
	def __init__(
		self,
		in_features,
		out_features,
		grid_size=5,
		spline_order=3,
		scale_noise=0.1,
		scale_base=1.0,
		scale_spline=1.0,
		enable_standalone_scale_spline=True,
		base_activation=torch.nn.SiLU,
		grid_eps=0.02,
		grid_range=[-1, 1],
	):
		super(KANLinear, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.grid_size = grid_size
		self.spline_order = spline_order

		h = (grid_range[1] - grid_range[0]) / grid_size
		grid = (
			(
				torch.arange(-spline_order, grid_size + spline_order + 1) * h
				+ grid_range[0]
			)
			.expand(in_features, -1)
			.contiguous()
		)
		self.register_buffer("grid", grid)

		self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
		self.spline_weight = torch.nn.Parameter(
			torch.Tensor(out_features, in_features, grid_size + spline_order)
		)
		if enable_standalone_scale_spline:
			self.spline_scaler = torch.nn.Parameter(
				torch.Tensor(out_features, in_features)
			)

		self.scale_noise = scale_noise
		self.scale_base = scale_base
		self.scale_spline = scale_spline
		self.enable_standalone_scale_spline = enable_standalone_scale_spline
		self.base_activation = base_activation()
		self.grid_eps = grid_eps

		self.reset_parameters()

	def reset_parameters(self):
		torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
		with torch.no_grad():
			noise = (
				(
					torch.rand(self.grid_size + 1, self.in_features, self.out_features)
					- 1 / 2
				)
				* self.scale_noise
				/ self.grid_size
			)
			self.spline_weight.data.copy_(
				(self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
				* self.curve2coeff(
					self.grid.T[self.spline_order : -self.spline_order],
					noise,
				)
			)
			if self.enable_standalone_scale_spline:
				# torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
				# torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)
				torch.nn.init.xavier_uniform_(self.spline_scaler) #New

	def b_splines(self, x: torch.Tensor):
		"""
		Compute the B-spline bases for the given input tensor.

		Args:
				x (torch.Tensor): Input tensor of shape (batch_size, in_features).

		Returns:
				torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
		"""
		assert x.dim() == 2 and x.size(1) == self.in_features

		grid: torch.Tensor = (
			self.grid
		)  # (in_features, grid_size + 2 * spline_order + 1)
		x = x.unsqueeze(-1)
		bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
		for k in range(1, self.spline_order + 1):
			bases = (
				(x - grid[:, : -(k + 1)])
				/ (grid[:, k:-1] - grid[:, : -(k + 1)])
				* bases[:, :, :-1]
			) + (
				(grid[:, k + 1 :] - x)
				/ (grid[:, k + 1 :] - grid[:, 1:(-k)])
				* bases[:, :, 1:]
			)

		assert bases.size() == (
			x.size(0),
			self.in_features,
			self.grid_size + self.spline_order,
		)
		return bases.contiguous()

	def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
		"""
		Compute the coefficients of the curve that interpolates the given points.

		Args:
				x (torch.Tensor): Input tensor of shape (batch_size, in_features).
				y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

		Returns:
				torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
		"""
		assert x.dim() == 2 and x.size(1) == self.in_features
		assert y.size() == (x.size(0), self.in_features, self.out_features)

		A = self.b_splines(x).transpose(
			0, 1
		)  # (in_features, batch_size, grid_size + spline_order)
		B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
		solution = torch.linalg.lstsq(
			A, B
		).solution  # (in_features, grid_size + spline_order, out_features)
		result = solution.permute(
			2, 0, 1
		)  # (out_features, in_features, grid_size + spline_order)

		assert result.size() == (
			self.out_features,
			self.in_features,
			self.grid_size + self.spline_order,
		)
		return result.contiguous()

	@property
	def scaled_spline_weight(self):
		return self.spline_weight * (
			self.spline_scaler.unsqueeze(-1)
			if self.enable_standalone_scale_spline
			else 1.0
		)

	def forward(self, x: torch.Tensor):
		assert x.size(-1) == self.in_features
		original_shape = x.shape
		x = x.view(-1, self.in_features)

		base_output = F.linear(self.base_activation(x), self.base_weight)
		spline_output = F.linear(
			self.b_splines(x).view(x.size(0), -1),
			self.scaled_spline_weight.view(self.out_features, -1),
		)
		output = base_output + spline_output
		
		output = output.view(*original_shape[:-1], self.out_features)
		return output

	@torch.no_grad()
	def update_grid(self, x: torch.Tensor, margin=0.01):
		assert x.dim() == 2 and x.size(1) == self.in_features
		batch = x.size(0)

		splines = self.b_splines(x)  # (batch, in, coeff)
		splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
		orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
		orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
		unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
		unreduced_spline_output = unreduced_spline_output.permute(
				1, 0, 2
		)  # (batch, in, out)

		# sort each channel individually to collect data distribution
		x_sorted = torch.sort(x, dim=0)[0]
		grid_adaptive = x_sorted[
			torch.linspace(
				0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
			)
		]

		uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
		grid_uniform = (
			torch.arange(
				self.grid_size + 1, dtype=torch.float32, device=x.device
			).unsqueeze(1)
			* uniform_step
			+ x_sorted[0]
			- margin
		)

		grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
		grid = torch.concatenate(
			[
				grid[:1]
				- uniform_step
				* torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
				grid,
				grid[-1:]
				+ uniform_step
				* torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
			],
			dim=0,
		)

		self.grid.copy_(grid.T)
		self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

	def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
		"""
		Compute the regularization loss.

		This is a dumb simulation of the original L1 regularization as stated in the
		paper, since the original one requires computing absolutes and entropy from the
		expanded (batch, in_features, out_features) intermediate tensor, which is hidden
		behind the F.linear function if we want an memory efficient implementation.

		The L1 regularization is now computed as mean absolute value of the spline
		weights. The authors implementation also includes this term in addition to the
		sample-based regularization.
		"""
		l1_fake = self.spline_weight.abs().mean(-1)
		regularization_loss_activation = l1_fake.sum()
		p = l1_fake / regularization_loss_activation
		regularization_loss_entropy = -torch.sum(p * p.log())
		return (
			regularize_activation * regularization_loss_activation
			+ regularize_entropy * regularization_loss_entropy
		)
	
class KANGConv(MessagePassing):
	def __init__(self, in_channels, out_channels, n_basis, degree, device, aggr='add', base_activation=torch.nn.SiLU, **kwargs):
		kwargs.setdefault('aggr', aggr) 
		super().__init__(**kwargs)

		self.n_basis = n_basis
		self.degree = degree
		self.device = device

		self.nn = KANLinear(in_channels, out_channels, n_basis, degree, base_activation=base_activation)
		
		self.reset_parameters()

	def reset_parameters(self):
		super().reset_parameters()

	def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, size: Size = None,) -> Tensor:
		if isinstance(x, Tensor):
			x = (x, x)
	
		out = self.propagate(edge_index, x=x, size=size)

		return self.nn(out)

	def message(self, x_j: Tensor) -> Tensor:
		return x_j

	def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
		if isinstance(adj_t, SparseTensor):
			adj_t = adj_t.set_value(None, layout=None)
		return spmm(adj_t, x[0], reduce=self.aggr)

	def __repr__(self) -> str:
		return f'{self.__class__.__name__}(nn={self.nn})'
	
class KANG(nn.Module):
	def __init__(self,
			in_channels, 
			hidden_channels, 
			out_channels, 
			num_layers, 
			n_basis, 
			degree, 
			dropout, 
			device, 
			aggr='add', 
			base_activation=torch.nn.SiLU,
			residuals=False
		):
		super(KANG, self).__init__()
		self.dropout		= dropout
		self.residuals	= residuals
		self.convs			= nn.ModuleList()

		# First Layer
		self.convs.append(KANGConv(in_channels, hidden_channels, n_basis, degree, device, aggr, base_activation=base_activation))

		# Subsequent Conv layers
		for _ in range(num_layers-1):
			self.convs.append(KANGConv(hidden_channels, hidden_channels, n_basis, degree, device, aggr, base_activation=base_activation))

		# Readout Layer
		self.out_layer = KANLinear(hidden_channels, out_channels, n_basis, degree, base_activation=base_activation)

		# Normalization Layers
		self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])

	def forward(self, x, edge_index, batch=None):
		x.requires_grad_(True)
		edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

		res = None
		for i, conv in enumerate(self.convs):
			res = x.clone()
			x = conv(x, edge_index)
			x = F.dropout(x, p=self.dropout, training=self.training)
			x = self.layer_norms[i](x)
			if self.residuals and i > 0: x += res

		# 2. Readout layer
		if batch != None:
			x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
		x = self.out_layer(x)

		return F.log_softmax(x, dim=1)
	
	def encode(self, x, edge_index):
		for i, conv in enumerate(self.convs):
			x = conv(x, edge_index)
			x = F.relu(x)
			x = self.layer_norms[i](x) # Use pre-initialized LayerNorm

		# x = self.linear(x)
		x = self.out_layer(x)
		# x = self.out_layer(x, edge_index)
		
		# return F.log_softmax(x, dim=1), all_edge_weights
		return x
	
	def decode(self, z, edge_label_index):
		return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)  # product of a pair of nodes on each edge

	def decode_all(self, z):
		prob_adj = z @ z.t()
		return (prob_adj > 0).nonzero(as_tuple=False).t()