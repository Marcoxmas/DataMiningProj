import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
import os
import shutil
from smiles_to_graph import smiles_to_data

class QM8GraphDataset(InMemoryDataset):

    def __init__(self, root, target_column='E1-CC2', transform=None, pre_transform=None):
        self.csv_file = "data/qm8.csv"
        self.target_column = target_column

        # Define available target columns for QM8
        self.available_targets = [
            'E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2',
            'E1-PBE0', 'E2-PBE0', 'f1-PBE0', 'f2-PBE0',
            'E1-PBE0.1', 'E2-PBE0.1', 'f1-PBE0.1', 'f2-PBE0.1',
            'E1-CAM', 'E2-CAM', 'f1-CAM', 'f2-CAM'
        ]

        if target_column not in self.available_targets:
            raise ValueError(f"target_column '{target_column}' not available. "
                           f"Available targets: {self.available_targets}")
        
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def len(self):
        return len(self.slices['x']) - 1

    def get(self, idx):
        return super().get(idx)

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        
        data_list = []
        valid_count = 0
        invalid_count = 0
        
        print(f"Processing QM8 dataset with target column: {self.target_column}")
        print(f"Total molecules to process: {len(df)}")
        
        for idx, row in df.iterrows():
            if idx % 10000 == 0:
                print(f"Processed {idx}/{len(df)} molecules...")
                
            smiles = row['smiles']
            target_value = row[self.target_column]
            
            if pd.isnull(target_value):
                invalid_count += 1
                continue
                
            try:
                data = smiles_to_data(smiles, labels=float(target_value))
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
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def download(self):
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
        return ['qm8.csv']

    @property
    def processed_file_names(self):
        return [f'data_{self.target_column}.pt']

    @property
    def num_classes(self):
        return 1

    def get_target_statistics(self):
        if not hasattr(self, '_target_stats'):
            df = pd.read_csv(self.raw_paths[0])
            target_values = df[self.target_column].dropna()
            
            self._target_stats = {
                'mean': float(target_values.mean()),
                'std': float(target_values.std()),
                'min': float(target_values.min()),
                'max': float(target_values.max()),
                'count': len(target_values)
            }
        
        return self._target_stats

    def print_dataset_info(self):
        stats = self.get_target_statistics()
        print(f"\nQM8 Dataset Information:")
        print(f"Target column: {self.target_column}")
        print(f"Number of samples: {len(self)}")
        print(f"Target statistics:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std:  {stats['std']:.4f}")
        print(f"  Min:  {stats['min']:.4f}")
        print(f"  Max:  {stats['max']:.4f}")
        print(f"  Count: {stats['count']}")
        
        sample = self.get(0)
        print(f"Node features: {sample.x.shape}")
        print(f"Edge features: {sample.edge_attr.shape}")

if __name__ == "__main__":
    dataset = QM8GraphDataset(root='./dataset/QM8')
    dataset.print_dataset_info()
    
    print("\n" + "="*50)
    dataset_homo = QM8GraphDataset(root='./dataset/QM8_HOMO', target_column='homo')
    dataset_homo.print_dataset_info()
    
    print(f"\nExample molecule (first sample):")
    sample = dataset.get(0)
    print(f"Number of atoms: {sample.x.shape[0]}")
    print(f"Number of edges: {sample.edge_index.shape[1]}")
    print(f"Target value ({dataset.target_column}): {sample.y.item():.4f}")
