import torch
import argparse
import matplotlib as mpl
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from torch_geometric.datasets import Planetoid

from src.KANG import KANG
from src.utils import train, validate, test, set_seed

torch.backends.cudnn.deterministic	= True
torch.backends.cudnn.benchmark		= False
set_seed(seed=42)

def plot_spline_evolution(spline_evolution, epochs,
						  layer_name="readout",
						  input_index=0, output_index=0,
						  plot_interval=5,
						  save_path=None):
	fig, ax = plt.subplots(figsize=(10, 6))
	
	cmap = plt.cm.viridis
	norm = mpl.colors.Normalize(vmin=0, vmax=epochs)
	sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
	sm.set_array([])

	for epoch, x_curve, y_curve, grid_x, grid_y in spline_evolution:
		if epoch % plot_interval == 0 or epoch == spline_evolution[-1][0]:
			color = cmap(norm(epoch))
			ax.plot(x_curve.numpy(), y_curve.numpy(), color=color, alpha=0.8, zorder=1)
			ax.scatter(grid_x.numpy(), grid_y.numpy(), color='#c00',
					   marker='o', s=15, zorder=2)

	cbar = fig.colorbar(sm, ax=ax)
	cbar.set_label('Epoch')

	ax.set_xlabel("Input")
	ax.set_ylabel("Spline")
	# ax.set_title(f"Spline Evolution for layer '{layer_name}'\n"
	# 			 f"(input={input_index}, output={output_index})")
	# ax.set_xlim(-2, 2)
	fig.savefig(save_path)
	plt.close(fig)
	print(f"Plot saved in {save_path}")

def get_monitored_layer(model, layer):
	if layer == 'readout':
		return model.out_layer
	else:
		try:
			layer_idx = int(layer)
			if 0 <= layer_idx < len(model.convs):
				return model.convs[layer_idx].nn
			else:
				raise ValueError(f"Layer index out of range. Ci sono solo {len(model.convs)} conv layers.")
		except ValueError:
			raise ValueError("Il parametro layer deve essere 'readout' oppure un indice intero valido.")

def main(args):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# Caricamento del dataset Cora
	dataset = Planetoid(
		root=f"./dataset/{args.dataset}", 
		name=args.dataset, 
	)
		# transform=NormalizeFeatures()
	data = dataset[0].to(device)
	
	# Inizializzazione del modello KANG
	model = KANG(
		in_channels=dataset.num_features,
		hidden_channels=32,
		out_channels=dataset.num_classes,
		num_layers=2,
		grid_min=-15,
		grid_max=20,
		num_grids=4,
		dropout=0.1,
		device=device,
		linspace=args.E,
		trainable_grid=args.T
	).to(device)
	
	optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=4e-4)
	
	best_val_acc = 0
	patience_trigger = 0
	best_epoch = 0
	best_model_path = args.model_path

	epochs = 1000
	patience = 400
	
	spline_evolution = []  # Lista per salvare gli snapshot della spline (epoch, x, y, ctrl_points, ctrl_responses)
	
	# Durante il training, salviamo gli snapshot per l'evoluzione
	for epoch in tqdm(range(epochs), desc="Training", leave=False):
		_, _ = train(model, data, optimizer)
		val_acc, _ = validate(model, data)
		
		if val_acc > best_val_acc:
			best_val_acc = val_acc
			best_epoch = epoch
			patience_trigger = 0
			torch.save(model.state_dict(), best_model_path)
		else:
			patience_trigger += 1

		# Save monitored layer snapshot of splines
		monitored_layer = get_monitored_layer(model, args.layer)

		x_curve, y_curve, grid_x, grid_y = monitored_layer.plot_curve(
			input_index=args.input_index, 
			output_index=args.output_index
		)

		spline_evolution.append((
			epoch,
			x_curve.detach().cpu(),
			y_curve.detach().cpu(),
			grid_x.detach().cpu(),
			grid_y.detach().cpu(),
		))
		
		if patience_trigger > patience:
			print(f"[i] Early stopping triggered @ epoch {epoch}")
			break
	
	print(f"Loading best model from {best_model_path} for testing...")
	model.load_state_dict(torch.load(best_model_path))
	test_acc = test(model, data)
	print(f"Test Acc: {test_acc:.4f}")
	
	save_path = f"./experiments/splines/spline_evolution_{'E' if args.E else 'G'}{'T' if args.T else '!T'}.pdf"
	plot_spline_evolution(
		spline_evolution[:best_epoch+1],
		epochs=best_epoch,
		layer_name=args.layer,
		input_index=args.input_index,
		output_index=args.output_index,
		plot_interval=args.plot_interval,
		save_path=save_path
	)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
			description="Displays the evolution of a layer's spline in the KANG model during training."
	)
	parser.add_argument("--dataset", type=str, default="Cora", help="Name of the dataset (default: Cora)")
	parser.add_argument("--model_path", type=str, default="./experiments/splines/gkan.pth", help="Path to save the model")

	# Arguments for spline monitoring
	parser.add_argument("--layer", type=str, default="readout",
											help="Layer to monitor: 'readout' or an index (e.g., '0' for the first conv layer)")
	parser.add_argument("--input_index", type=int, default=0, help="Index of the input feature to visualize")
	parser.add_argument("--output_index", type=int, default=0, help="Index of the output neuron to visualize")
	parser.add_argument("--plot_interval", type=int, default=5, help="Interval for plotting a snapshot during training")
	parser.add_argument("--T", action="store_true", help="Use trainable knots")
	parser.add_argument("--E", action="store_true", help="Use evenly distributed knots")
	
	args = parser.parse_args()
	main(args)