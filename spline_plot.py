import argparse
import torch
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from src.KANG import KANG

# Assumiamo che le classi KANG, KANGConv, FastKANLayer, SplineLinear e RadialBasisFunction
# siano già definite e importate, ad esempio:
# from model import KANG

# Funzione di supporto per selezionare il layer da monitorare
def get_monitored_layer(model, layer):
	"""
	Se layer è la stringa "readout", restituisce il layer di output.
	Se layer è un intero (espresso come stringa) restituisce il conv layer corrispondente.
	"""
	if layer.lower() == "readout":
		return model.out_layer
	else:
		try:
			layer_idx = int(layer)
			if 0 <= layer_idx < len(model.convs):
				return model.convs[layer_idx].nn
			else:
				raise ValueError(f"Layer index out of range. Ci sono solo {len(model.convs)} conv layers.")
		except ValueError:
			raise ValueError("L'argomento layer deve essere 'readout' oppure un indice intero.")

def main(args):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# Carica il dataset Cora
	dataset = Planetoid(
		root=f"./dataset/Cora", 
		name="Cora", 
		transform=NormalizeFeatures()
	)
	
	# Istanzia il modello KANG
	model = KANG(
		in_channels = dataset.num_features,
		hidden_channels = 32,
		out_channels = dataset.num_classes,
		num_layers = 2,
		grid_min = -15,
		grid_max = 20,
		num_grids = 4,
		dropout = 0.2,
		device = device,
		residuals=True
	).to(device)
	
	# Carica il modello pre-trainato
	model.load_state_dict(torch.load(args.model_path, map_location=device))
	model.eval()
	
	# Seleziona il layer da monitorare (readout oppure conv layer specificato)
	monitored_layer = get_monitored_layer(model, args.layer)
	
	# Estrae la spline (e i control points) per gli indici specificati
	# La funzione plot_curve è stata modificata per restituire:
	# (x_curve, y_curve, ctrl_points, ctrl_responses)
	x_curve, y_curve, ctrl_points, ctrl_responses = monitored_layer.plot_curve(
		input_index=args.input_index,
		output_index=args.output_index
	)
	
	# Plot: la spline e la risposta effettiva in ciascun control point
	plt.figure(figsize=(8, 6))
	plt.plot(x_curve.detach().cpu().numpy(), y_curve.detach().cpu().numpy(), 
			 label="Spline Curve", lw=2, zorder=1)
	plt.scatter(ctrl_points.detach().cpu().numpy(), ctrl_responses.detach().cpu().numpy(), 
				color="#c00", marker='.', s=50, label="Control Points", zorder=2)
	plt.xlabel("Valori di Input")
	plt.ylabel("Valori della Spline")
	plt.title(f"Spline Explanation\nLayer: {args.layer}, Input Index: {args.input_index}, Output Index: {args.output_index}")
	plt.legend()
	plt.savefig(args.save_path)
	plt.close()
	print(f"Plot della spline salvato in {args.save_path}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Carica un modello pre-trainato su Cora e salva il plot della spline (con control points) del layer specificato."
	)
	parser.add_argument("--model_path", type=str, default="./comparisons/node_classification/models/kang_Cora.pth", help="Percorso del modello pre-trainato")
	parser.add_argument("--layer", type=str, default="readout", help="Layer da spiegare: 'readout' oppure indice (es. '0')")
	parser.add_argument("--input_index", type=int, default=0, help="Indice dell'input da spiegare")
	parser.add_argument("--output_index", type=int, default=0, help="Indice dell'output da spiegare")
	parser.add_argument("--save_path", type=str, default="./experiments/splines/spline_explanation.png", help="Percorso in cui salvare il plot della spline")
	
	args = parser.parse_args()
	main(args)