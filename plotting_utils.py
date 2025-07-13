"""
Plotting utilities for training metrics visualization.
"""
import os
import matplotlib.pyplot as plt

def plot_training_metrics(train_losses, val_metrics, task_type, dataset_name, target_column=None):
    """
    Plot training loss and validation metrics for the best hyperparameter trial.
    
    Args:
        train_losses: List of training losses per epoch
        val_metrics: List of validation metrics per epoch (accuracy for classification, MAE for regression)
        task_type: 'classification' or 'regression'
        dataset_name: Name of the dataset
        target_column: Target column name for regression tasks
    
    Returns:
        str: Path to the saved plot
    """
    # Create plots directory structure
    if task_type == "classification":
        plots_dir = f"experiments/graph_classification/plots"
        plot_filename = f"{dataset_name}_best_trial_metrics.png"
        val_metric_name = "Validation Accuracy"
    else:
        plots_dir = f"experiments/graph_regression/plots"
        plot_filename = f"{dataset_name}_{target_column}_best_trial_metrics.png"
        val_metric_name = "Validation MAE"
    
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create the plot with dual y-axes
    epochs_range = list(range(len(train_losses)))
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot training loss on primary y-axis
    color1 = 'red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color1)
    line1 = ax1.plot(epochs_range, train_losses, color=color1, linewidth=2, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Create secondary y-axis for validation metric
    ax2 = ax1.twinx()
    color2 = 'blue'
    ax2.set_ylabel(val_metric_name, color=color2)
    line2 = ax2.plot(epochs_range, val_metrics, color=color2, linewidth=2, label=val_metric_name)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add title and legend
    ax1.set_title(f'Training Progress - {dataset_name}')
    
    # Combine legends from both axes
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    # Adjust layout and save
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training metrics plot saved to: {plot_path}")
    return plot_path
