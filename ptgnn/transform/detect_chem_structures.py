import copy

from rdkit import Chem
from rdkit.Chem import AllChem


CIS_TRANS_STEREO_LIST = [
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE
]


def get_chiral_center_node_mask(
        mol,
        chiral_center_select_potential,
        node_mapping,
        only_out: bool
):
    # get chiral centers
    chiral_center_list = AllChem.FindMolChiralCenters(mol, includeUnassigned=chiral_center_select_potential)
    # modify so that only the node idx is left in the list (not interested in R/S)
    chiral_center_list = [c[0] for c in chiral_center_list]

    # browse through node mapping and select edges
    # separation between only_out mode
    if only_out:
        return [
            idx
            for idx, (key, value) in enumerate(node_mapping.items())
            if key[0] in chiral_center_list
        ]

    else:
        return [
            idx
            for idx, (key, value) in enumerate(node_mapping.items())
            if key[0] in chiral_center_list or key[1] in chiral_center_list
        ]


def fetch_cis_trans_edges(node_mapping, molecule, cis_trans_edges_select_potential):
    nodes_list = []
    # iterate over bonds
    for bond in molecule.GetBonds():
        # check if bond is a double bond and stereo is specified
        # add potential stereo elements to check list
        check_list = copy.copy(CIS_TRANS_STEREO_LIST)
        if cis_trans_edges_select_potential:
            check_list += [Chem.rdchem.BondStereo.STEREOANY]
        # random (cis or trans)
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and bond.GetStereo() in CIS_TRANS_STEREO_LIST:
            # fetch ends of the bond
            node_a = bond.GetBeginAtom().GetIdx()
            node_b = bond.GetEndAtom().GetIdx()

            # check if first direction edge is in node mapping
            if (node_a, node_b) in node_mapping:
                nodes_list.append((node_a, node_b))
            if (node_b, node_a) in node_mapping:
                nodes_list.append((node_b, node_a))

    return nodes_list
