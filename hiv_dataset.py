import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
import os
import shutil
from smiles_to_graph import smiles_to_data

class HIVGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.csv_file = "data/HIV.csv"
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def len(self):
        return len(self.slices['x']) - 1

    def get(self, idx):
        # Use the parent class method which properly handles slicing
        return super().get(idx)

    def process(self):
        """
        Processes the raw data into PyTorch Geometric Data objects and saves them.
        """
        df = pd.read_csv(self.raw_paths[0])
        
        data_list = []
        valid_count = 0
        invalid_count = 0
        
        for idx, row in df.iterrows():
            smiles = row['smiles']
            label = row['HIV_active']  # Use HIV_active column as the label
            
            # Skip if label is missing
            if pd.isnull(label):
                invalid_count += 1
                continue
                
            try:
                data = smiles_to_data(smiles, labels=int(label))
                if data is not None and hasattr(data, 'edge_index'):
                    data_list.append(data)
                    valid_count += 1
                else:
                    print(f"Excluded invalid graph for SMILES: {smiles}")
                    invalid_count += 1
            except Exception as e:
                print(f"Error processing SMILES '{smiles}': {e}")
                invalid_count += 1
        
        print(f"Processed {valid_count} valid molecules, {invalid_count} invalid/missing")
        
        if not data_list:
            raise ValueError("No valid data found! Check your data.")
        
        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def download(self):
        """
        Downloads the dataset if it does not exist.
        """
        # Copy the main CSV file to the raw directory if it doesn't exist
        raw_csv_path = self.raw_paths[0]
        os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)
        
        main_csv_path = os.path.join(os.path.dirname(os.path.dirname(self.root)), self.csv_file)
        
        if not os.path.exists(raw_csv_path) and os.path.exists(main_csv_path):
            print(f"Copying {main_csv_path} to {raw_csv_path}")
            shutil.copy2(main_csv_path, raw_csv_path)
        elif not os.path.exists(main_csv_path):
            raise FileNotFoundError(f"Main CSV file not found at {main_csv_path}")
        elif os.path.exists(raw_csv_path):
            print(f"CSV file already exists at {raw_csv_path}")

    @property
    def raw_file_names(self):
        """
        Returns the list of raw file names that must be present in the raw directory.
        """
        return ['HIV.csv']

    @property
    def processed_file_names(self):
        """
        Returns the list of processed file names that must be present in the processed directory.
        """
        return ['data.pt']

    @property
    def num_classes(self):
        """
        Returns the number of classes in the dataset.
        """
        return 2  # Binary classification for HIV activity

# Example usage
if __name__ == "__main__":
    dataset = HIVGraphDataset(root="dataset/HIV")
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Number of node features: {dataset.num_node_features}")
    for i in range(min(5, len(dataset))):
        data = dataset[i]
        print(f"Sample {i}: {data}")
