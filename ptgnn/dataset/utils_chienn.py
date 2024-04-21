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
    Adapted from https://github.com/gmum/ChiENN/blob/master/experiments/graphgps/dataset/utils.py

    :param url: URL from which to download the dataset file(s).
    :type url: str
    :param path: Path to which to save the file(s)
    :type path: str
    :return: path to which the files were saved
    :rtype: Path
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
    Fetches the positions of the atoms in the passed molecules and returns it as a numpy array.
    Adapted from:
    https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/experiments/graphgps/dataset/utils.py#L65

    :param mol: Molecule from which to get the positions for all atoms
    :type mol: rdkit.Chem.Mol
    :return: positions
    :rtype: np.ndarray[float]
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
    Copied from `ChIRo.model.datasets_samplers.MaskedGraphDataset.process_mol`. It encodes a molecule with some basic
    chemical features. It also provides chiral tag, which can be then masked in `graphgps.dataset.rs_dataset.RS`.

    :param mol: Molecule for which to generate the features
    :type mol: rdkit.Chem.Mol
    :return: data object generated from the passed molecule
    :rtype: torch_geometric.data.Data
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


def convert_target_for_task(
        target: torch.Tensor,
        task_type: str,
        scale_label: float = 1.0
) -> torch.Tensor:
    """
    Modifies the label to match the task prediction type.

    Adapted from:
    https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/experiments/graphgps/dataset/utils.py#L142

    :param target: label to modify
    :type target: torch.Tensor
    :param task_type: Task type to which to fit the label to
    :type task_type: str
    :param scale_label: Parameter by which the label is to scale in case of regression
    :type scale_label: float
    :return: modified label
    :rtype: torch.Tensor
    """
    if task_type in ['regression', 'regression_rank']:
        return target.float() * scale_label
    elif task_type == 'classification_multilabel':
        return target.float().view(1, -1)
    return target.long()
