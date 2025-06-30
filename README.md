# Kolmogorov–Arnold Graph Neural Networks (KANG)

This repository contains the implementation of the **Kolmogorov–Arnold Network for Graphs (KANG)**, introduced in the paper:

**Kolmogorov–Arnold Graph Neural Networks**

> Kolmogorov–Arnold Networks (KANs) recently emerged as a powerful alternative to traditional multilayer perceptrons, providing enhanced generalization and intrinsic interpretability through learnable spline-based activation functions. Motivated by the need for powerful and transparent graph neural networks (GNNs), we propose the Kolmogorov–Arnold Network for Graphs (KANG), a novel GNN architecture that integrates KANs into the message-passing framework. Experiments on benchmark datasets for node classification, link prediction, and graph classification tasks show that KANG consistently outperforms established GNN architectures.

## Repository Overview

This repository provides scripts to train, evaluate, and analyze the **KANG model**. The structure of the repository is as follows:

### Main Scripts

- **`graph_classification.py`** - Implementation of graph classification experiments.
- **`node_classification.py`** - Implementation of node classification experiments.
- **`link_prediction.py`** - Implementation of link prediction experiments.
- **`kan_implementation_ablation.py`** - Script for testing different KAN-based configurations.
- **`oversmoothing.py`** - Analysis of oversmoothing effects in GNNs.
- **`residuals_or_not_residuals.py`** - Investigating the impact of residual connections in KANG.
- **`comparisons.py`** - Comparing KANG with other GNN architectures.
- **`ablation_grid.py`** - Grid search for hyperparameter tuning.
- **`sensitivity_analysis.py`** - Sensitivity analysis of hyperparameters.
- **`scalability.py`** - Evaluating the computational scalability of KANG.
- **`spline_evolution.py`** - Monitoring the evolution of spline-based activations during training.
- **`spline_plot.py`** - Visualization of spline functions used in KANG.
- **`smiles_to_graph.py`** - Convert SMILES strings to PyTorch Geometric graphs for molecular ML tasks.

### Training and Execution Scripts

- **`run_ablation.sh`** - Shell script to run the ablation study.
- **`run_finetune_all.sh`** - Shell script to fine-tune all GNN models.
- **`run_kan_implementation_ablation.sh`** - Shell script to test different KAN-based configurations.
- **`run_oversmoothing.sh`** - Shell script to analyze oversmoothing effects.
- **`run_residuals.sh`** - Shell script to test the impact of residual connections.

### Hyperparameter Tuning and Experiment Tracking

- **`wandb_gkan_finetune.py`** - Fine-tuning KANG using Weights & Biases (WandB).
- **`wandb_gnn_finetune.py`** - Fine-tuning baseline GNN models using WandB.

## Running Experiments

### Node Classification

```bash
python node_classification.py --dataset Cora --epochs 200 --lr 0.01
```

### Link Prediction

```bash
python link_prediction.py --dataset PubMed --epochs 500 --lr 0.005
```

### Graph Classification

```bash
python graph_classification.py --dataset MUTAG --epochs 300 --lr 0.001
```

### Ablation Study

```bash
bash run_ablation.sh
```

## Citation

If you use this code, please cite our paper:

Anonymus repo for submission

## License

This repository is licensed under the MIT License.

## Contact

For questions or collaborations, please reach out via GitHub issues or email.

Anonymus repo for submission