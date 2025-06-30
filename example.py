import os
from toxcast_dataset import ToxCastGraphDataset, ToxCastMultiTaskDataset

# Define the target assays for multitask learning
assays = [
    "TOX21_p53_BLA_p3_ch1",
    "TOX21_p53_BLA_p4_ratio",
    "TOX21_AhR_LUC_Agonist",
    "TOX21_Aromatase_Inhibition",
    "TOX21_AutoFluor_HEK293_Cell_blue"
]

print("=== Creating Multi-Task Dataset ===")
dataset_root = "data/toxcast_graph_data"
multitask_root = os.path.join(dataset_root, "multitask")

# Create multitask dataset
print(f"\nCreating multitask dataset for assays: {assays}")
multitask_dataset = ToxCastMultiTaskDataset(root=multitask_root, target_columns=assays)

print(f"Multitask dataset created successfully!")
print(f" - Total molecules: {len(multitask_dataset)}")
print(f" - Number of tasks: {multitask_dataset.get_num_tasks()}")
print(f" - Target columns: {multitask_dataset.get_target_columns()}")

# Inspect a sample molecule
if len(multitask_dataset) > 0:
    sample = multitask_dataset[0]
    print(f" - Sample molecule features:")
    print(f"   * Nodes: {sample.num_nodes}")
    print(f"   * Edges: {sample.num_edges}")
    print(f"   * Node features shape: {sample.x.shape}")
    print(f"   * Edge features shape: {sample.edge_attr.shape}")
    print(f"   * Labels shape: {sample.y.shape}")
    print(f"   * Labels: {sample.y.tolist()}")

print("\n=== Comparison: Individual Single-Task Datasets ===")
# Create individual datasets for comparison
for i, assay in enumerate(assays):
    print(f"\n[{i+1}/{len(assays)}] Processing dataset for assay: {assay}")
    try:
        dataset = ToxCastGraphDataset(root=os.path.join(dataset_root, assay), target_column=assay)
        print(f" - Saved {len(dataset)} molecules for: {assay}")
    except Exception as e:
        print(f" - Error creating dataset for {assay}: {e}")

print(f"\n=== Summary ===")
print(f"Multitask dataset: {len(multitask_dataset)} molecules with {multitask_dataset.get_num_tasks()} tasks")
print(f"Multitask approach allows training a single model on all {len(assays)} assays simultaneously")
print(f"This enables better feature sharing and potentially improved performance on related tasks")