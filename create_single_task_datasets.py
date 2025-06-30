import os
import pandas as pd
from toxcast_dataset import ToxCastGraphDataset

# Define the target assays for single-task datasets
assays = [
    "TOX21_p53_BLA_p3_ch1",
    "TOX21_p53_BLA_p4_ratio",
    "TOX21_AhR_LUC_Agonist",
    "TOX21_Aromatase_Inhibition",
    "TOX21_AutoFluor_HEK293_Cell_blue"
]

data_root = "dataset/TOXCAST"
raw_csv_path = "data/toxcast_data.csv"

# Ensure the raw CSV file exists
if not os.path.exists(raw_csv_path):
    raise FileNotFoundError(f"Raw CSV file not found at {raw_csv_path}")

# Create the TOXCAST directory if it doesn't exist
os.makedirs(data_root, exist_ok=True)

# Create single-task datasets
for assay in assays:
    print(f"Processing dataset for assay: {assay}")
    assay_root = os.path.join(data_root, assay)
    try:
        dataset = ToxCastGraphDataset(root=assay_root, target_column=assay)
        print(f" - Saved {len(dataset)} molecules for assay: {assay}")
    except Exception as e:
        print(f" - Error creating dataset for assay {assay}: {e}")

print("\nAll single-task datasets have been created successfully!")
