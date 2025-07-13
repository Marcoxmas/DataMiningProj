#!/usr/bin/env python3
"""
Feature Distribution Analysis for Molecular Datasets
Analyzes feature distributions across QM8, QM9, HIV, and ToxCast datasets.
Generates statistical summaries and distribution plots for each dataset and globally.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from qm9_dataset import QM9GraphDataset
from qm8_dataset import QM8GraphDataset
from hiv_dataset import HIVGraphDataset
from toxcast_dataset import ToxCastGraphDataset

def analyze_single_dataset(dataset, dataset_name, output_dir, sample_size=1000):
    """Analyze the distribution of node features for a single dataset."""
    
    print(f"\n{'='*60}")
    print(f"ANALYZING DATASET: {dataset_name}")
    print(f"{'='*60}")
    
    # Sample molecules for analysis
    actual_sample_size = min(sample_size, len(dataset))
    all_features = []
    
    print(f"Sampling {actual_sample_size} molecules from {len(dataset)} total...")
    
    successful_samples = 0
    for i in range(actual_sample_size):
        try:
            data = dataset[i]
            if data is not None and hasattr(data, 'x') and data.x is not None:
                all_features.append(data.x)
                successful_samples += 1
        except Exception as e:
            print(f"Warning: Failed to load sample {i}: {e}")
            continue
    
    if not all_features:
        print(f"No valid data found for {dataset_name}!")
        return None
    
    # Concatenate all features
    all_features = torch.cat(all_features, dim=0)  # Shape: [total_nodes, num_features]
    
    print(f"Successfully analyzed {successful_samples} molecules")
    print(f"Total nodes: {all_features.shape[0]}, Features per node: {all_features.shape[1]}")
    
    # Calculate statistics
    feature_min = all_features.min(dim=0)[0]
    feature_max = all_features.max(dim=0)[0]
    feature_mean = all_features.mean(dim=0)
    feature_std = all_features.std(dim=0)
    
    # Overall statistics (across all features and nodes)
    global_min = all_features.min().item()
    global_max = all_features.max().item()
    global_mean = all_features.mean().item()
    global_std = all_features.std().item()
    
    print(f"\nGLOBAL FEATURE STATISTICS FOR {dataset_name}")
    print(f"Global Min: {global_min:.4f}")
    print(f"Global Max: {global_max:.4f}")
    print(f"Global Mean: {global_mean:.4f}")
    print(f"Global Std: {global_std:.4f}")
    
    # Create detailed plots for this dataset
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Histogram of all feature values
    axes[0].hist(all_features.flatten().numpy(), bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    axes[0].axvline(global_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {global_mean:.3f}')
    axes[0].axvline(global_mean - global_std, color='orange', linestyle='--', linewidth=2, label=f'Mean-Std: {global_mean - global_std:.3f}')
    axes[0].axvline(global_mean + global_std, color='orange', linestyle='--', linewidth=2, label=f'Mean+Std: {global_mean + global_std:.3f}')
    axes[0].set_xlabel('Feature Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{dataset_name}: Distribution of All Feature Values')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Per-feature mean and std
    feature_indices = range(len(feature_min))
    axes[1].errorbar(feature_indices, feature_mean.numpy(), 
                    yerr=feature_std.numpy(), fmt='o', alpha=0.6, capsize=2, markersize=3)
    axes[1].set_xlabel('Feature Index')
    axes[1].set_ylabel('Feature Value')
    axes[1].set_title(f'{dataset_name}: Per-Feature Mean ± Std')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f'{dataset_name}_feature_distribution.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Feature distribution plot saved as '{plot_filename}'")
    plt.close()
    
    return {
        'dataset_name': dataset_name,
        'num_samples': successful_samples,
        'num_nodes': all_features.shape[0],
        'num_features': all_features.shape[1],
        'global_min': global_min,
        'global_max': global_max,
        'global_mean': global_mean,
        'global_std': global_std,
        'feature_min': feature_min,
        'feature_max': feature_max,
        'feature_mean': feature_mean,
        'feature_std': feature_std
    }

def load_datasets():
    """Load QM8, QM9, HIV, and ToxCast datasets."""
    datasets = {}
    
    # QM9 dataset
    try:
        qm9_dataset = QM9GraphDataset(root='./dataset/QM9_mu', target_column='mu')
        datasets['QM9'] = qm9_dataset
        print(f"Loaded QM9 dataset: {len(qm9_dataset)} molecules")
    except Exception as e:
        print(f"Failed to load QM9: {e}")
    
    # QM8 dataset
    qm8_targets = ['E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2']
    for target in qm8_targets:
        try:
            qm8_dataset = QM8GraphDataset(root=f'./dataset/QM8_{target}', target_column=target)
            datasets['QM8'] = qm8_dataset
            print(f"Loaded QM8 {target} dataset: {len(qm8_dataset)} molecules")
            break
        except Exception as e:
            print(f"Failed to load QM8 {target}: {e}")
    
    # HIV dataset
    try:
        hiv_dataset = HIVGraphDataset(root='./dataset/HIV')
        datasets['HIV'] = hiv_dataset
        print(f"Loaded HIV dataset: {len(hiv_dataset)} molecules")
    except Exception as e:
        print(f"Failed to load HIV: {e}")
    
    # ToxCast dataset
    toxcast_targets = ['NVS_NR_hER', 'TOX21_ARE_BLA_Agonist_ratio', 'TOX21_p53_BLA_p1_ratio']
    for target in toxcast_targets:
        try:
            toxcast_dataset = ToxCastGraphDataset(root=f'./dataset/TOXCAST/{target}', target_column=target)
            datasets['ToxCast'] = toxcast_dataset
            print(f"Loaded ToxCast {target} dataset: {len(toxcast_dataset)} molecules")
            break
        except Exception as e:
            print(f"Failed to load ToxCast {target}: {e}")
    
    return datasets

def analyze_all_datasets():
    """Analyze feature distributions for all available datasets."""
    
    # Create output directory
    output_dir = "features_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    print("LOADING DATASETS...")
    datasets = load_datasets()
    
    if not datasets:
        print("No datasets could be loaded!")
        return None
    
    print(f"\nANALYZING {len(datasets)} DATASETS...")
    
    all_results = []
    all_features_combined = []
    
    for dataset_name, dataset in datasets.items():
        result = analyze_single_dataset(dataset, dataset_name, output_dir, sample_size=5000)
        if result:
            all_results.append(result)
            # Collect features for global analysis
            try:
                data_sample = dataset[0]
                if data_sample is not None and hasattr(data_sample, 'x'):
                    all_features_combined.append(data_sample.x)
            except:
                pass
    
    # Create global analysis
    if all_results and all_features_combined:
        create_global_analysis(all_results, all_features_combined, output_dir)
    
    # Save individual dataset summaries
    save_dataset_summaries(all_results, output_dir)
    
    return all_results

def create_global_analysis(all_results, all_features_combined, output_dir):
    """Create a global analysis plot comparing all datasets."""
    
    print("\nCreating global analysis...")
    
    # Create global comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Global mean comparison
    dataset_names = [r['dataset_name'] for r in all_results]
    global_means = [r['global_mean'] for r in all_results]
    global_stds = [r['global_std'] for r in all_results]
    
    x_pos = np.arange(len(dataset_names))
    bars = axes[0, 0].bar(x_pos, global_means, yerr=global_stds, capsize=5, alpha=0.7, color=['blue', 'green', 'red', 'orange'])
    axes[0, 0].set_xlabel('Dataset')
    axes[0, 0].set_ylabel('Global Mean ± Std')
    axes[0, 0].set_title('Global Feature Means Across Datasets')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(dataset_names, rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(global_means, global_stds)):
        axes[0, 0].text(i, mean + std + 0.01, f'{mean:.3f}', ha='center', va='bottom')
    
    # Plot 2: Feature dimensionality comparison
    num_features = [r['num_features'] for r in all_results]
    bars = axes[0, 1].bar(x_pos, num_features, alpha=0.7, color=['blue', 'green', 'red', 'orange'])
    axes[0, 1].set_xlabel('Dataset')
    axes[0, 1].set_ylabel('Number of Features')
    axes[0, 1].set_title('Feature Dimensionality Across Datasets')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(dataset_names, rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, num_feat in enumerate(num_features):
        axes[0, 1].text(i, num_feat + 1, str(num_feat), ha='center', va='bottom')
    
    # Plot 3: Sample size comparison
    num_samples = [r['num_samples'] for r in all_results]
    bars = axes[1, 0].bar(x_pos, num_samples, alpha=0.7, color=['blue', 'green', 'red', 'orange'])
    axes[1, 0].set_xlabel('Dataset')
    axes[1, 0].set_ylabel('Number of Samples Analyzed')
    axes[1, 0].set_title('Sample Sizes Analyzed')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(dataset_names, rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, samples in enumerate(num_samples):
        axes[1, 0].text(i, samples + 10, str(samples), ha='center', va='bottom')
    
    # Plot 4: Range comparison
    global_mins = [r['global_min'] for r in all_results]
    global_maxs = [r['global_max'] for r in all_results]
    range_centers = [(min_val + max_val) / 2 for min_val, max_val in zip(global_mins, global_maxs)]
    range_heights = [max_val - min_val for min_val, max_val in zip(global_mins, global_maxs)]
    
    bars = axes[1, 1].bar(x_pos, range_heights, bottom=global_mins, alpha=0.7, color=['blue', 'green', 'red', 'orange'])
    axes[1, 1].set_xlabel('Dataset')
    axes[1, 1].set_ylabel('Feature Value Range')
    axes[1, 1].set_title('Feature Value Ranges Across Datasets')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(dataset_names, rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add range labels
    for i, (min_val, max_val) in enumerate(zip(global_mins, global_maxs)):
        axes[1, 1].text(i, max_val + 0.02, f'[{min_val:.2f}, {max_val:.2f}]', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    global_plot_filename = os.path.join(output_dir, 'global_comparison.png')
    plt.savefig(global_plot_filename, dpi=300, bbox_inches='tight')
    print(f"Global comparison plot saved as '{global_plot_filename}'")
    plt.close()

def save_dataset_summaries(all_results, output_dir):
    """Save individual summaries for each dataset and a global summary."""
    
    # Save individual dataset summaries
    for result in all_results:
        dataset_filename = os.path.join(output_dir, f"{result['dataset_name']}_summary.txt")
        with open(dataset_filename, 'w') as f:
            f.write(f"FEATURE ANALYSIS SUMMARY: {result['dataset_name']}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Dataset: {result['dataset_name']}\n")
            f.write(f"Samples analyzed: {result['num_samples']}\n")
            f.write(f"Total nodes: {result['num_nodes']}\n")
            f.write(f"Features per node: {result['num_features']}\n\n")
            f.write("GLOBAL STATISTICS:\n")
            f.write(f"  Minimum value: {result['global_min']:.6f}\n")
            f.write(f"  Maximum value: {result['global_max']:.6f}\n")
            f.write(f"  Mean value: {result['global_mean']:.6f}\n")
            f.write(f"  Standard deviation: {result['global_std']:.6f}\n")
            f.write(f"  Value range: [{result['global_min']:.6f}, {result['global_max']:.6f}]\n")
    
    # Save global summary
    global_filename = os.path.join(output_dir, "global_summary.txt")
    with open(global_filename, 'w') as f:
        f.write("GLOBAL FEATURE ANALYSIS SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write("Analysis of molecular datasets: QM8, QM9, HIV, and ToxCast\n")
        f.write(f"Total datasets analyzed: {len(all_results)}\n\n")
        
        f.write("DATASET COMPARISON TABLE:\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Dataset':<15} {'Samples':<8} {'Nodes':<8} {'Features':<9} {'Min':<8} {'Max':<8} {'Mean':<8} {'Std':<8}\n")
        f.write("-"*100 + "\n")
        
        for result in all_results:
            f.write(f"{result['dataset_name']:<15} "
                   f"{result['num_samples']:<8} "
                   f"{result['num_nodes']:<8} "
                   f"{result['num_features']:<9} "
                   f"{result['global_min']:<8.4f} "
                   f"{result['global_max']:<8.4f} "
                   f"{result['global_mean']:<8.4f} "
                   f"{result['global_std']:<8.4f}\n")
        
        f.write("-"*100 + "\n\n")
        
        # Overall statistics
        all_mins = [r['global_min'] for r in all_results]
        all_maxs = [r['global_max'] for r in all_results]
        all_means = [r['global_mean'] for r in all_results]
        all_stds = [r['global_std'] for r in all_results]
        
        f.write("OVERALL STATISTICS ACROSS ALL DATASETS:\n")
        f.write(f"  Minimum of all minimums: {min(all_mins):.6f}\n")
        f.write(f"  Maximum of all maximums: {max(all_maxs):.6f}\n")
        f.write(f"  Average of all means: {np.mean(all_means):.6f}\n")
        f.write(f"  Average of all standard deviations: {np.mean(all_stds):.6f}\n")
        f.write(f"  Overall range: [{min(all_mins):.6f}, {max(all_maxs):.6f}]\n\n")
    
    print(f"Individual dataset summaries saved in '{output_dir}' folder")
    print(f"Global summary saved as '{global_filename}'")

if __name__ == "__main__":
    # Analyze all datasets
    all_results = analyze_all_datasets()
    
    if all_results:
        print(f"\n" + "="*80)
        print(f"ANALYSIS COMPLETE!")
        print(f"Datasets analyzed: {len(all_results)}")
        print(f"Results saved in 'features_analysis' folder:")
        print(f"  - Individual dataset plots and summaries")
        print(f"  - Global comparison plot")
        print(f"  - Global summary report")
        print(f"="*80)
    else:
        print("No datasets could be analyzed!")
