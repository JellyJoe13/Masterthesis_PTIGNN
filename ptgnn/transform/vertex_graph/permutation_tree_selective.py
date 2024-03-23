import json

import torch
import torch_geometric
from rdkit.Chem import AllChem

from ptgnn.transform.edge_graph.permutation_tree_selective import custom_to_edge_graph
from ptgnn.transform.detect_chem_structures import fetch_cis_trans_edges, detect_possible_axial_chiral_edges
from ptgnn.transform.multi_stereo_center import calc_edges_multiple_stereo_centers
from ptgnn.transform.ptree_matrix import permutation_tree_to_order_matrix


def get_cis_trans_ordering(data, node_a, node_b):
    # get neighbors
    neighbors = data.edge_index[:, data.edge_index[1] == node_a]
    neighbors = neighbors[:, neighbors[0] != node_b]

    neighbors = torch.cat([neighbors, data.edge_index[:, data.edge_index[1] == node_b]], dim=-1)
    neighbors = neighbors[:, neighbors[0] != node_a]

    # get positions of neighbors
    pos = data.pos[neighbors[0]]

    # calculate order by getting the nearest of the neighbors, then the next until finished
    # calculate distances with 2-norm
    distances = torch.zeros(4,4)
    for i in range(4):
        for j in range(4):
            distances[i, j] = ((pos[i] - pos[j]) ** 2).sum().sqrt()

    order = [0]
    while len(distances):
        rel_order = distances[order[-1]].argsort()
        for i in range(4):
            if not rel_order[i] in order:
                order.append(int(rel_order[i]))
                break
        else:
            order = torch.tensor(order)
            break

    return neighbors[0, order].tolist()


def permutation_tree_vertex_transformation(
        data,
        mol,
        k,
        tetrahedral_chiral: bool = True,
        chiral_center_selective: bool = False,
        chiral_center_select_potential: bool = True,
        cis_trans_edges: bool = False,
        cis_trans_edges_select_potential: bool = False,
        create_order_matrix: bool = True,
        axial_chirality: bool = False,
        multi_stereo_center_dia: bool = False
) -> torch_geometric.data.Data:
    # get the edge graph transformation
    # required for some permutation tree creations (due to circle index needed from edges)
    edge_graph, node_mapping = custom_to_edge_graph(
        data=data
    )

    # calculate the list of chiral centers
    chiral_center_list = AllChem.FindMolChiralCenters(
        mol,
        includeUnassigned=chiral_center_select_potential
    )
    chiral_center_list = [
        c[0]
        for c in chiral_center_list
    ]

    # init permutation tree list
    permutation_trees = []

    # iterate over nodes in data object
    for idx in range(data.x.shape[0]):

        # fetch the outgoing edges
        out_edges = data.edge_index[:, data.edge_index[0] == idx]

        # adds check whether there are enough neighbors to justify complex chiral ptree
        if out_edges.shape[1] < 4 or not tetrahedral_chiral or \
                (chiral_center_selective and not idx in chiral_center_list):
            # append following tree
            permutation_trees.append(
                json.dumps({
                    "S": [
                        int(idx),
                        {"P": out_edges[1].tolist()}
                    ]
                })
            )
            continue

        # iterate over edges, get circle index and create permutation tree
        circle_indices = [
            edge_graph.circle_index[node_mapping[(x, y)]]
            for x, y in out_edges.T.tolist()
        ]

        # convert the cycle indices into node indices (currently edge indices)
        reverse_dict = dict(zip(node_mapping.values(), node_mapping.keys()))
        circle_indices = [
            [
                reverse_dict[c][0]
                for c in circ_ind
            ]
            for circ_ind in circle_indices
        ]

        # merge tree and add to list
        permutation_trees.append(json.dumps({
            "S": [
                int(idx),
                {
                    "P": [
                        {
                            "Z": circle_index
                        }
                        for circle_index in circle_indices
                    ]
                }
            ]
        }))

    if cis_trans_edges:
        cis_trans_nodes_list = fetch_cis_trans_edges(
            node_mapping,
            mol,
            cis_trans_edges_select_potential
        )

        # iterate over edges and generate new ptrees for these entries
        for node_a, node_b in cis_trans_nodes_list:
            permutation_trees[node_a] = json.dumps({
                'S': [
                    int(node_a),
                    {'C': get_cis_trans_ordering(data, node_a, node_b)}
                ]
            })

    if axial_chirality:
        # calculate candidates for axial chirality
        potential_axial_nodes_list = detect_possible_axial_chiral_edges(
            molecule=mol
        )

        # iterate over edges and generate new ptrees for these entries
        for node_a, node_b in potential_axial_nodes_list:
            permutation_trees[node_a] = json.dumps({
                'S': [
                    int(node_a),
                    {'C': get_cis_trans_ordering(data, node_a, node_b)}
                ]
            })

    # in case only diastereomers are to be distinguished
    if multi_stereo_center_dia:
        # get reverse dict
        reverse_dict = dict(zip(node_mapping.values(), node_mapping.keys()))

        # get stereo paths
        stereo_paths = calc_edges_multiple_stereo_centers(mol, chiral_center_select_potential)

        # iterate over stereo paths
        for ((source_a, source_b), (target_a, target_b)) in stereo_paths:
            # get circle indices
            source_circle_index = edge_graph.circle_index[node_mapping[(source_a, source_b)]]
            target_circle_index = edge_graph.circle_index[node_mapping[(target_b, target_a)]]

            # create new node with zero embedding
            data.x = torch.cat([
                data.x,
                torch.zeros(1, data.x.shape[-1])
            ], dim=0)

            # create new permutation tree
            permutation_trees.append(
                json.dumps({
                    "P": [
                        {
                            "P": [
                                {
                                    "Z": [
                                        reverse_dict[circ_ind][0]
                                        for circ_ind in source_circle_index
                                    ]
                                },
                                {
                                    "Z": [
                                        reverse_dict[circ_ind][0]
                                        for circ_ind in target_circle_index
                                    ]
                                }
                            ]
                        },
                        {
                            "P": [
                                {
                                    "Z": [
                                        reverse_dict[circ_ind][0]
                                        for circ_ind in source_circle_index[::-1]
                                    ]
                                },
                                {
                                    "Z": [
                                        reverse_dict[circ_ind][0]
                                        for circ_ind in target_circle_index[::-1]
                                    ]
                                }
                            ]
                        }
                    ]
                })
            )

    # register permutation trees in data object
    data.ptree = permutation_trees

    if create_order_matrix:
        # generate order matrix
        data = permutation_tree_to_order_matrix(data, k)

    return data
