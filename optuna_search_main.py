import argparse
import numpy as np
import optuna
import os
import json
from graph_classification import graph_classification
from graph_regression import graph_regression
from plotting_utils import plot_training_metrics

def get_args():
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization with Optuna")
    parser.add_argument("--task", type=str, choices=["classification", "regression"], required=True, help="Task type")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--target_column", type=str, default="mu", help="Target column for regression tasks")
    parser.add_argument("--use_subset", action="store_true", help="Use a smaller subset of the dataset during tuning")
    parser.add_argument("--subset_ratio", type=float, default=0.3, help="Ratio of dataset to use for subset (default: 0.3)")
    return parser.parse_args()

def optuna_search(task_type, dataset_name, target_column, use_subset=True, subset_ratio=0.3):
    def objective(trial):
        try:
            import argparse
            args = argparse.Namespace()
            args.dataset_name = dataset_name
            args.target_column = target_column
            args.lr = trial.suggest_float("lr", 0.001, 0.01, log=True)
            args.wd = trial.suggest_float("wd", 1e-5, 1e-3, log=True)
            args.hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64, 128, 256])
            args.layers = trial.suggest_int("layers", 1, 7)
            args.dropout = trial.suggest_float("dropout", 0.1, 0.5)
            args.num_grids = trial.suggest_categorical("num_grids", [10, 12, 14, 16])
            args.batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
            args.gamma = trial.suggest_float("gamma", 0.5, 2.5)
            args.grid_min = -1.1
            args.grid_max = 1.1
            args.epochs = 50
            args.patience = 30
            args.log_freq = args.epochs // 10
            args.use_weighted_loss = True
            args.use_roc_auc = True
            # Enable subset for faster hyperparameter tuning
            args.use_subset = use_subset
            args.subset_ratio = subset_ratio

            if task_type == "classification":
                print("Running classification with:", args)
                best_val_acc = graph_classification(args)
                # Check for invalid results
                if best_val_acc is None or np.isnan(best_val_acc) or np.isinf(best_val_acc):
                    raise ValueError("Invalid validation accuracy returned")
                return best_val_acc  # maximize accuracy
            elif task_type == "regression":
                print("Running regression with:", args)
                best_val_score = graph_regression(args)
                # Check for invalid results
                if best_val_score is None or np.isnan(best_val_score) or np.isinf(best_val_score):
                    raise ValueError("Invalid validation score returned")
                return best_val_score  # minimize MAE
        except Exception as e:
            print(f"Trial failed with error: {e}")
            # Return a penalty value instead of raising
            if task_type == "classification":
                return 0.0  # Worst possible accuracy
            else:
                return float('inf')  # Worst possible MAE
    study = optuna.create_study(direction="minimize" if task_type == "regression" else "maximize")
    study.optimize(objective, n_trials=20)

    # Save best hyperparameters
    os.makedirs("experiments/hparam_search", exist_ok=True)
    with open("experiments/hparam_search/best_trial.json", "w") as f:
        best_params = study.best_trial.params.copy()
        best_params["use_subset"] = use_subset
        best_params["subset_ratio"] = subset_ratio
        json.dump(best_params, f, indent=4)

    print("\nBest hyperparameters:")
    print(study.best_trial.params)
    if use_subset:
        print(f"Note: Hyperparameters found using {subset_ratio*100:.0f}% of the dataset for faster tuning.")
    
    # Run the best trial again to get training history for plotting
    print("\nRunning best hyperparameters again on FULL dataset to generate training plots...")
    import argparse
    best_args = argparse.Namespace()
    best_args.dataset_name = dataset_name
    best_args.target_column = target_column
    
    # Apply best hyperparameters
    for param, value in study.best_trial.params.items():
        setattr(best_args, param, value)
    
    # Set fixed parameters - DISABLE subset for final training
    best_args.grid_min = -1.1
    best_args.grid_max = 1.1
    best_args.epochs = 200
    best_args.patience = 50
    best_args.log_freq = best_args.epochs // 10
    best_args.use_weighted_loss = True
    best_args.use_roc_auc = True
    best_args.return_history = True  # Flag to return training history
    best_args.use_subset = False  # Use full dataset for final training
    best_args.subset_ratio = 1.0
    
    if task_type == "classification":
        try:
            best_val_acc, train_losses, val_metrics = graph_classification(best_args, return_history=True)
            plot_training_metrics(train_losses, val_metrics, task_type, dataset_name)
        except Exception as e:
            print(f"Warning: Could not generate training plots for classification: {e}")
    elif task_type == "regression":
        try:
            best_val_score, train_losses, val_metrics = graph_regression(best_args, return_history=True)
            plot_training_metrics(train_losses, val_metrics, task_type, dataset_name, target_column)
        except Exception as e:
            print(f"Warning: Could not generate training plots for regression: {e}")

if __name__ == "__main__":
    args = get_args()
    optuna_search(args.task, args.dataset_name, args.target_column, args.use_subset, args.subset_ratio)
