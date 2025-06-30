import os
import torch
from toxcast_dataset import ToxCastGraphDataset
from smiles_to_graph import print_graph_info

def view_toxcast_examples(dataset_folder, assay_name, num_examples=5):
    dataset_path = os.path.join(dataset_folder, assay_name)
    if not os.path.exists(dataset_path):
        print(f"Dataset folder for assay '{assay_name}' does not exist at {dataset_path}.")
        return

    dataset = ToxCastGraphDataset(root=dataset_path, target_column=assay_name)
    print(f"Viewing {num_examples} examples from the dataset for assay: {assay_name}")

    for i, data in enumerate(dataset[:num_examples]):
        print(f"Example {i + 1}:")
        print(print_graph_info(data))
        print("-" * 40)

if __name__ == "__main__":
    dataset_folder = "dataset/TOXCAST"  # Corrected the path
    assay_name = "TOX21_p53_BLA_p3_ch1"  # Change this to view a different assay
    view_toxcast_examples(dataset_folder, assay_name)