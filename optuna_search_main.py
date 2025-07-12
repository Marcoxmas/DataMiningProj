import argparse
import optuna
import os
import json
from graph_classification import graph_classification
from graph_regression import graph_regression

def get_args():
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization with Optuna")
    parser.add_argument("--task", type=str, choices=["classification", "regression"], required=True, help="Task type")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--target_column", type=str, default="mu", help="Target column for regression tasks")
    return parser.parse_args()

def optuna_search(task_type, dataset_name, target_column):
    def objective(trial):
        import argparse
        args = argparse.Namespace()
        args.dataset_name = dataset_name
        args.target_column = target_column
        args.lr = trial.suggest_float("lr", 0.0025, 0.0035)
        args.wd = trial.suggest_float("wd", 8e-5, 3e-4)
        args.hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64, 128])
        args.layers = trial.suggest_int("layers", 1, 6)
        args.dropout = trial.suggest_float("dropout", 0.0001, 0.01)
        args.num_grids = trial.suggest_categorical("num_grids", [10, 12])
        args.batch_size = trial.suggest_categorical("batch_size", [128])
        args.grid_min = -10
        args.grid_max = 3
        args.epochs = 100
        args.patience = 20
        args.log_freq = 50
        args.use_weighted_loss = False
        args.use_roc_auc = True

        if task_type == "classification":
            print("Running classification with:", args)
            best_val_acc = graph_classification(args)
            return best_val_acc  # maximize accuracy
        elif task_type == "regression":
            print("Running regression with:", args)
            best_val_score = graph_regression(args)
            return best_val_score  # minimize MAE

    study = optuna.create_study(direction="minimize" if task_type == "regression" else "maximize")
    study.optimize(objective, n_trials=20)

    os.makedirs("experiments/hparam_search", exist_ok=True)
    with open("experiments/hparam_search/best_trial.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=4)

    print("\nBest hyperparameters:")
    print(study.best_trial.params)

if __name__ == "__main__":
    args = get_args()
    optuna_search(args.task, args.dataset_name, args.target_column)
