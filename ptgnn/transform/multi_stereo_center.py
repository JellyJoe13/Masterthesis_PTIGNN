from itertools import chain

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def limited_bfs_stereo_path_search(
        current_molecule,
        current_stereocenters,
        start_stereo
):
    """
    Uses a BFS approach to connect only few stereocenters with maximal effect - not really working for molecules with
    more than 3 stereo centers. Approach was cancelled(difference in diastereomers is too small) so never further
    developed.

    :param current_molecule: molecule
    :type current_molecule: rdkit.Chem.rdchem.Mol
    :param current_stereocenters: List of stereocenter (indices)
    :type current_stereocenters: typing.List[int]
    :param start_stereo: stereo center to search (builds tree until closest center is reached
    :return: found closest stereo center(s), predecessor matrix - compare with default path algorithms like Dijkstra
    """
    # init procedure
    # fetch adjacency matrix
    adjacency_matrix = Chem.GetAdjacencyMatrix(current_molecule)
    # init visited nodes
    visited_nodes = np.array([False] * adjacency_matrix.shape[0])
    visited_nodes[start_stereo] = True
    # init predecessors
    predecessors = [[] for _ in range(adjacency_matrix.shape[0])]
    # init current_layer
    current_layer = [start_stereo]

    # iterate as long as possible
    while True:
        # init next layer
        next_layer = []

        # iterate over element in current layer
        for elem in current_layer:
            # fetch outgoing edges
            next_layer_cand = np.where(adjacency_matrix[elem])[0]

            # check and subselect
            next_layer_cand = next_layer_cand[~visited_nodes[next_layer_cand]]

            # add predecessor
            for i in next_layer_cand:
                predecessors[i].append(elem)

            # check and add
            next_layer.extend(next_layer_cand)

        # set new layer nodes to visited (so that they cannot be visited anymore
        visited_nodes[next_layer] = True

        # calc if any of the found new nodes are stereocenters
        stereo_test = np.isin(next_layer, current_stereocenters)
        # if so, stop looping
        if stereo_test.any():
            return np.array(next_layer)[stereo_test], predecessors

        # set new current_layer
        current_layer = next_layer


def calc_paths(
        current_path,
        predecessors
):
    """Calc paths from predecessors and current path"""
    # get predecessors of last node in path
    pred = predecessors[current_path[-1]]

    # if empty list then return current path
    if len(pred):
        return list(chain.from_iterable([
            calc_paths(current_path + [next_node], predecessors)
            for next_node in pred
        ]))
    else:
        # no further elements
        return [current_path]


def calc_edges_multiple_stereo_centers(
        molecule,
        include_unassigned
):
    """Calc which edges are to be connected for a molecule with multiple stereo centers (not working fully for molecules
    with more than 3 stereo centers)"""
    # get stereocenters
    stereo_centers = AllChem.FindMolChiralCenters(molecule, includeUnassigned=include_unassigned)
    stereo_centers = [k for k, _ in stereo_centers]

    # include assert, if less than 2 centers then do nothing
    if len(stereo_centers) < 0:
        return []

    # init storage where the edges to create/re-tree are stored
    stereo_paths = []

    # iterate over all stereo_centers
    for start_stereo in stereo_centers:

        # get pre-path(s) to nearest stereo center
        found_stereos, predecessors = limited_bfs_stereo_path_search(
            molecule,
            stereo_centers,
            start_stereo
        )

        # get the resulting paths
        current_paths = list(chain.from_iterable([
            calc_paths([found_stereos[0]], predecessors)
        ]))

        # process paths, check if already present (double edges/nodes not used)
        for path in current_paths:

            # create both directions
            direction_a = ((path[0], path[1]), (path[-2], path[-1]))
            direction_b = ((path[-1], path[-2]), (path[1], path[0]))

            # check if already in paths and if not add it
            if direction_a not in stereo_paths and direction_b not in stereo_paths:
                stereo_paths.append(direction_a)

    return stereo_paths
