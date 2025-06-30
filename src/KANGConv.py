from typing import Union

from torch import Tensor
from torch_geometric.utils import spmm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.nn.models import MLP

from src.KANLinear import KANLinear
from src.KAND import KAND

class KANGConv(MessagePassing):
	def __init__(
			self, 
			in_channels, 
			out_channels, 
			grid_min=-1,
			grid_max=1,
			num_grids=2,
			device='cpu', 
			aggr='mean',
			kan=True,
			linspace = False,
			trainable_grid = True,
			bsplines=False,
			**kwargs
		):

		kwargs.setdefault('aggr', aggr) 
		super().__init__(**kwargs)

		self.device = device

		if kan:
			if bsplines:
				self.nn = KANLinear(
					in_channels,
					out_channels,
					grid_size=num_grids,
					grid_range=[grid_min, grid_max]
				)
			else:
				self.nn = KAND(
					in_channels,
					out_channels,
					grid_min,
					grid_max,
					num_grids,
					device=device,
					linspace = linspace,
					trainable_grid = trainable_grid
				)
		else:
			self.nn = MLP([in_channels, out_channels])
		
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