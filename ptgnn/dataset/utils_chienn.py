"""
The content of this file is exclusively written by the authors of ChiENN(https://github.com/gmum/ChiENN/tree/master).
The origins of the functions will be provided as links.
"""
import ssl
import urllib
from pathlib import Path

import numpy as np
import rdkit
import torch
import torch_geometric

from ptgnn.features.chiro.embedding_functions import embedConformerWithAllPaths


def download_url_to_path(url, path):
    """
     https://github.com/gmum/ChiENN/blob/master/experiments/graphgps/dataset/utils.py
    """
    path = Path(path)
    if path.exists():
        return Path

    path.parent.mkdir(parents=True, exist_ok=True)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        f.write(data.read())

    return path


def get_positions(mol: rdkit.Chem.Mol):
    """
    From: https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/experiments/graphgps/dataset/utils.py#L65
    """
    conf = mol.GetConformer()
    return np.array(
        [
            [
                conf.GetAtomPosition(k).x,
                conf.GetAtomPosition(k).y,
                conf.GetAtomPosition(k).z,
            ]
            for k in range(mol.GetNumAtoms())
        ]
    )


def get_chiro_data_from_mol(mol: rdkit.Chem.Mol):
    """
    From: https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/experiments/graphgps/dataset/utils.py#L65
    Copied from `ChIRo.model.datasets_samplers.MaskedGraphDataset.process_mol`. It encoded molecule with some basic
    chemical features. It also provides chiral tag, which can be then masked in `graphgps.dataset.rs_dataset.RS`.
    """
    atom_symbols, edge_index, edge_features, node_features, bond_distances, bond_distance_index, bond_angles, bond_angle_index, dihedral_angles, dihedral_angle_index = embedConformerWithAllPaths(
        mol, repeats=False)

    bond_angles = bond_angles % (2 * np.pi)
    dihedral_angles = dihedral_angles % (2 * np.pi)
    pos = get_positions(mol)

    data = torch_geometric.data.Data(x=torch.as_tensor(node_features),
                                     edge_index=torch.as_tensor(edge_index, dtype=torch.long),
                                     edge_attr=torch.as_tensor(edge_features),
                                     pos=torch.as_tensor(pos, dtype=torch.float))
    data.bond_distances = torch.as_tensor(bond_distances)
    data.bond_distance_index = torch.as_tensor(bond_distance_index, dtype=torch.long).T
    data.bond_angles = torch.as_tensor(bond_angles)
    data.bond_angle_index = torch.as_tensor(bond_angle_index, dtype=torch.long).T
    data.dihedral_angles = torch.as_tensor(dihedral_angles)
    data.dihedral_angle_index = torch.as_tensor(dihedral_angle_index, dtype=torch.long).T

    return data
