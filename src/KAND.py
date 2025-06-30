import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *

class SplineLinear(nn.Linear):
		def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
				self.init_scale = init_scale
				super().__init__(in_features, out_features, bias=False, **kw)

		def reset_parameters(self) -> None:
				torch.nn.init.kaiming_uniform_(self.weight, a=self.init_scale)
				# Initialize weights from a normal distribution with mean 0 and std 1
				# torch.nn.init.normal_(self.weight, mean=0.0, std=1.0)
				# self.weight.data *= self.init_scale

class RadialBasisFunction(nn.Module):
		def __init__(
				self,
				grid_min: float = -1.,
				grid_max: float = 1.,
				num_grids: int = 8,
				denominator: float = None,  # larger denominators lead to smoother basis
				linspace: bool = False,
				trainable_grid: bool = True
		):
				super().__init__()
				self.grid_min = grid_min
				self.grid_max = grid_max
				self.num_grids = num_grids

				grid = None
				if linspace:
					grid = torch.linspace(grid_min, grid_max, num_grids)
				else:
					# Sample control points from a Gaussian distribution with mean 0 and std 1
					grid = torch.randn(num_grids)  # Sample from N(0, 1)
					
					# Scale and shift the sampled points to fit within [grid_min, grid_max]
					grid = torch.clamp(grid, grid_min, grid_max)
					
					# Sort the grid to ensure it's in ascending order
					grid, _ = torch.sort(grid)

				self.grid = torch.nn.Parameter(grid, requires_grad=trainable_grid)

				self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

		def forward(self, x):
				return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

class KAND(nn.Module):
		def __init__(
				self,
				input_dim: int,
				output_dim: int,
				grid_min: float = -2.,
				grid_max: float = 2.,
				num_grids: int = 8,
				use_base_update: bool = True,
				use_layernorm: bool = True,
				base_activation = F.silu,
				spline_weight_init_scale: float = 0.05,
				device='cpu',
				linspace=False,
				trainable_grid=True
		) -> None:
				super().__init__()
				self.input_dim	= input_dim
				self.output_dim	= output_dim
				self.layernorm	= None
				self.device			= device
				self.use_layernorm = use_layernorm
				if use_layernorm:
						assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
						self.layernorm = nn.LayerNorm(input_dim)
				self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids, linspace, trainable_grid)
				self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
				self.spline_linear.reset_parameters()
				self.use_base_update = use_base_update
				if use_base_update:
						self.base_activation = base_activation
						self.base_linear = nn.Linear(input_dim, output_dim)

		def forward(self, x):
				if self.layernorm is not None and self.use_layernorm:
						spline_basis = self.rbf(self.layernorm(x))
				else:
						spline_basis = self.rbf(x)

				ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
				if self.use_base_update:
						base = self.base_linear(self.base_activation(x))
						ret = ret + base
				return ret
		
		def plot_curve(
				self,
				input_index: int,
				output_index: int,
				num_pts: int = 1000,
				num_extrapolate_bins: int = 2
		):
				'''this function returns the learned curves in a FastKANLayer.
				input_index: the selected index of the input, in [0, input_dim) .
				output_index: the selected index of the output, in [0, output_dim) .
				num_pts: num of points sampled for the curve.
				num_extrapolate_bins (N_e): num of bins extrapolating from the given grids. The curve 
						will be calculate in the range of [grid_min - h * N_e, grid_max + h * N_e].
				'''
				ng = self.rbf.num_grids
				h = self.rbf.denominator
				assert input_index < self.input_dim
				assert output_index < self.output_dim

				w = self.spline_linear.weight[
						output_index, input_index * ng : (input_index + 1) * ng
				]   # num_grids,

				x = torch.linspace(
						self.rbf.grid_min - num_extrapolate_bins * h,
						self.rbf.grid_max + num_extrapolate_bins * h,
						num_pts
				).to(self.device)  # num_pts, num_grids

				grid_x = self.rbf.grid

				with torch.no_grad():
					rbf_vals = self.rbf(x.to(w.dtype))  # shape: (num_pts, num_grids)
					y = (w * rbf_vals).sum(dim=-1)      # shape: (num_pts,)

					rbf_at_controls = self.rbf(grid_x.to(w.dtype))  # shape: (num_grids, num_grids)
					grid_y = (w * rbf_at_controls).sum(dim=-1)  # shape: (num_grids,)

				return x, y, grid_x, grid_y