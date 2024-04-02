import copy

import rdkit
import typing
from rdkit import Chem
from rdkit.Chem import AllChem


CIS_TRANS_STEREO_LIST = [
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE
]


def get_chiral_center_node_mask(
        mol: rdkit.Chem.rdchem.Mol,
        chiral_center_select_potential: bool,
        node_mapping: typing.Dict[typing.Tuple[int, int], int],
        only_out: bool
) -> typing.List[int]:
    """
    Get centers for which a chiral center is marked (or possible).

    :param mol: Molecule
    :type mol: rdkit.Chem.rdchem.Mol
    :param chiral_center_select_potential: Whether or not to include potential chiral centers
    :type chiral_center_select_potential: bool
    :param node_mapping: Node mapping of edges to nodes in case of an edge graph
    :type node_mapping: typing.Dict[typing.Tuple[int, int], int]
    :param only_out: Parameter that controls whether to only consider the source node neighbors or also the target node
        neighbors
    :type only_out: bool
    :return: List of edges -> nodes which are (potentially) a chiral center
    :rtype: typing.List[int]
    """
    # get chiral centers
    chiral_center_list = AllChem.FindMolChiralCenters(mol, includeUnassigned=chiral_center_select_potential)
    # modify so that only the node idx is left in the list (not interested in R/S)
    chiral_center_list = [c[0] for c in chiral_center_list]

    # browse through node mapping and select edges
    # separation between only_out mode
    if only_out:
        return [
            value
            for key, value in node_mapping.items()
            if key[0] in chiral_center_list
        ]

    else:
        return [
            idx
            for idx, (key, value) in enumerate(node_mapping.items())
            if key[0] in chiral_center_list or key[1] in chiral_center_list
        ]


def fetch_cis_trans_edges(
        node_mapping: typing.Dict[typing.Tuple[int, int], int],
        molecule: rdkit.Chem.rdchem.Mol,
        cis_trans_edges_select_potential: bool
) -> typing.List[typing.Tuple[int, int]]:
    """
    Fetches edges which are marked (or have potential to be) as a cis/trans or E/Z type.

    :param node_mapping: Mapping of edges to nodes in case of edge graph
    :type node_mapping: typing.Dict[typing.Tuple[int, int], int]
    :param molecule: Molecule
    :type molecule: rdkit.Chem.rdchem.Mol
    :param cis_trans_edges_select_potential: Whether or not to return potential but unmarked edges
    :type cis_trans_edges_select_potential: bool
    :return: Edges for which cis/trans or E/Z is defined (or possible)
    :rtype:
    """
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
            # fetch ends of the bond and check correct number of bonds
            node_a = bond.GetBeginAtom()
            if node_a.GetDegree() != 3:
                continue
            node_a = node_a.GetIdx()

            node_b = bond.GetEndAtom()
            if node_b.GetDegree() != 3:
                continue
            node_b = node_b.GetIdx()

            # check if first direction edge is in node mapping
            if (node_a, node_b) in node_mapping:
                nodes_list.append((node_a, node_b))
            if (node_b, node_a) in node_mapping:
                nodes_list.append((node_b, node_a))

    return nodes_list


def detect_possible_axial_chiral_edges(
        molecule: rdkit.Chem.rdchem.Mol,
        node_mapping: typing.Dict[typing.Tuple[int, int], int] = None,
) -> typing.List[typing.Tuple[int, int]]:
    """
    Returns candidates of edges related to axial chirality. Method is not entirely accurate, may detect more edges than
    are truly a candidate. Implemented axial chirality between rings.

    :param molecule: Molecule
    :type molecule: rdkit.Chem.rdchem.Mol
    :param node_mapping: Maps edges to nodes in case of an edge graph transformation. Default: ``None``
    :type node_mapping: typing.Dict[typing.Tuple[int, int], int]
    :return: Edges or nodes where axial chirality is possible.
    :rtype: typing.List[typing.Tuple[int, int]]
    """
    # create storage for potential axial chirality edges
    potential_edges = []

    # fetch the adjacency matrix of molecule
    adjacency_matrix = Chem.GetAdjacencyMatrix(molecule)

    # iterate over ring combinations
    for i, ring_a in enumerate(molecule.GetRingInfo().AtomRings()):
        for j, ring_b in enumerate(molecule.GetRingInfo().AtomRings()):

            # if it's the same ring then skip
            if i == j:
                continue

            # if ring share atoms then skip
            if len(set(ring_a + ring_b)) < (len(ring_a) + len(ring_b)):
                continue

            # get edges connecting the two rings
            connection_edges = [
                (i, j)
                for i in ring_a
                for j in ring_b
                if adjacency_matrix[i, j]
            ]

            # if there is only one edge connecting rings: yes. else no
            if len(connection_edges) == 1:
                potential_edges += connection_edges

    if node_mapping is None:
        return potential_edges
    else:
        return [
            edge
            for edge in potential_edges
            if edge in node_mapping
        ]
