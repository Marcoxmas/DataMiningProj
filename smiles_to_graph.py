import torch
from torch_geometric.data import Data
from rdkit import Chem


def one_hot(value, choices):
    encoding = [0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1
    return encoding


def atom_features(atom):
    """Returns a rich atom feature vector."""
    features = []
    # 1. Atom type (one-hot over 100 common elements, atomic number)
    atom_type_list = list(range(1, 101))  # Atomic numbers from 1 to 100
    features += one_hot(atom.GetAtomicNum(), atom_type_list)

    # 2. Number of bonds (degree, one-hot 0-5)
    features += one_hot(atom.GetDegree(), list(range(6)))

    # 3. Formal charge (scaled integer, normalize to [-1, 1] assuming range -5 to 5)
    formal_charge = float(atom.GetFormalCharge())
    features.append(formal_charge / 5.0)  # Normalized

    # 4. Chirality (one-hot: R, S, None, other)
    chirality = atom.GetChiralTag()
    chirality_list = [
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,   # R
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,  # S
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,      # None
        Chem.rdchem.ChiralType.CHI_OTHER             # Other
    ]
    features += one_hot(chirality, chirality_list)

    # 5. Number of Hydrogens (one-hot 0-4)
    features += one_hot(atom.GetTotalNumHs(includeNeighbors=True), list(range(5)))

    # 6. Hybridization (one-hot: sp, sp2, sp3, etc.)
    hybridization_list = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]
    features += one_hot(atom.GetHybridization(), hybridization_list)

    # 7. Aromaticity (boolean)
    features.append(int(atom.GetIsAromatic()))

    # 8. Atomic mass (continuous, scaled by 0.01, then normalized to [0, 1] for common elements)
    # Most common atomic masses are < 250, so scale by 0.01 and then divide by 2.5
    atomic_mass = atom.GetMass() * 0.01
    features.append(atomic_mass / 2.5)

    # Convert to tensor and clamp to [0, 1] for all features (except one-hot and chirality, which are already 0/1)
    feat_tensor = torch.tensor(features, dtype=torch.float)
    #feat_tensor = torch.clamp(feat_tensor, 0.0, 1.0)
    return feat_tensor

def bond_features(bond):
    """Returns a rich bond feature vector."""
    bond_type_list = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    stereo_list = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOANY,
    ]
    features = []
    features += one_hot(bond.GetBondType(), bond_type_list)
    features.append(int(bond.GetIsConjugated()))
    features.append(int(bond.IsInRing()))
    features += one_hot(bond.GetStereo(), stereo_list)
    return torch.tensor(features, dtype=torch.float)


def bond_to_edge_index(mol, smiles=None):
    """Returns edge_index and edge_attr for a molecule."""
    edge_index = [[], []]
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        # Add both directions (undirected graph)
        edge_index[0] += [i, j]
        edge_index[1] += [j, i]
        edge_attr += [bf, bf]
    if not edge_attr:
        print(f"Skipped molecule with no bonds: {smiles}")
        return None, None
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.stack(edge_attr)
    return edge_index, edge_attr


def smiles_to_data(smiles: str, labels=None, include_hydrogens: bool = False) -> Data:
    """
    Converts a SMILES string to a PyTorch Geometric Data object.
    
    Args:
        smiles (str): SMILES string.
        labels (float, list, torch.Tensor, or None): Single label, list of labels, or tensor for multitask learning.
        include_hydrogens (bool): Whether to add explicit hydrogens to the molecule.
        
    Returns:
        torch_geometric.data.Data: Graph object or None if parsing fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if include_hydrogens:
        mol = Chem.AddHs(mol)

    # Node features
    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()])

    # Edge index and edge attributes
    edge_index, edge_attr = bond_to_edge_index(mol, smiles=smiles)
    if edge_index is None or edge_attr is None or edge_index.numel() == 0 or x.numel() == 0:
        print(f"Excluded invalid or empty graph for SMILES: {smiles}")
        return None

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if labels is not None:
        if isinstance(labels, (list, tuple)):
            data.y = torch.tensor(labels, dtype=torch.float)
        elif isinstance(labels, torch.Tensor):
            data.y = labels.float()
        else:
            # Single label (backward compatibility)
            data.y = torch.tensor([labels], dtype=torch.float)

    return data

def print_graph_info(data: Data):
    print("Graph Information:")
    print(f" - Number of nodes: {data.num_nodes}")
    print(f" - Node features:\n{data.x}")
    print(f" - Edge index:\n{data.edge_index}")
    print(f" - Number of edges: {data.num_edges}")
    print(f" - Edge attributes:\n{data.edge_attr}")
    if hasattr(data, 'y'):
        print(f" - Label: {data.y}")
    else:
        print(" - No label assigned.")