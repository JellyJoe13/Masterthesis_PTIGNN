import copy
import json
from collections import defaultdict

import rdkit
import torch_geometric
import torch

from ptgnn.transform.detect_chem_structures import get_chiral_center_node_mask, fetch_cis_trans_edges, \
    detect_possible_axial_chiral_edges
from ptgnn.transform.edge_graph.basic_permutation_tree import _circle_index_to_primordial_tree
from ptgnn.transform.edge_graph.chienn.get_circle_index import get_circle_index
from ptgnn.transform.edge_graph.cyclic_tree import cyclic_tree_edge, cyclic_tree_edge_minimal, cyclic_tree_edge_light
from ptgnn.transform.edge_graph.permutation_tree_special import get_cistrans_tree
from ptgnn.transform.multi_stereo_center import calc_edges_multiple_stereo_centers
from ptgnn.transform.ptree_matrix import permutation_tree_to_order_matrix
from ptgnn.transform.tree_separation import separate_tree_into_subtrees


def custom_to_edge_graph(data):
    """
    Modified version of to_edge_graph from ``ptgnn.transform.chienn.to_edge_graph``

    Original source:
    https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/chienn/data/edge_graph/to_edge_graph.py#L12

    :param data: vertex graph
    :type data: torch_geometric.data.Data
    :return: edge graph
    :rtype: torch_geometric.data.Data
    """
    # make sure that edges are undirected (in chemical context necessary)
    if torch_geometric.utils.is_undirected(data.edge_index):
        edge_index, edge_attr = data.edge_index, data.edge_attr
    else:
        edge_index, edge_attr = torch_geometric.utils.to_undirected(
            edge_index=data.edge_index,
            edge_attr=data.edge_attr
        )

    # create the new nodes
    node_storage = []
    node_mapping = {}
    for (a,b), edge_attr in zip(edge_index.T.tolist(), edge_attr):
        # create the embedding for the new node
        embedding_a2b = torch.cat([
            data.x[a],
            edge_attr,
            data.x[b]
        ])  # x_{i, j} = x'_i | e'_{i, j} | x'_j.

        # create the new position
        pos = torch.cat([data.pos[a], data.pos[b]])

        # todo: add only one if duplicate edges/nodes are required
        #   find solution for problem of embedding. either don't care and take one direction or sum up?
        # add to the storages
        node_mapping[(a, b)] = len(node_storage)
        node_storage.append({
            'a': a,
            'b': b,
            'a_attr': data.x[a],
            'node_attr': embedding_a2b,
            'old_edge_attr': edge_attr,
            'pos': pos
        })

    # create dictionary for ingoing nodes (helper for later)
    in_nodes = defaultdict(list)

    # iterate over new nodes
    for i, node_dict in enumerate(node_storage):
        # unpack edge source and target
        a, b = node_dict['a'], node_dict['b']

        # add source to each target
        in_nodes[b].append({'node_idx': i, 'start_node_idx': a})

    # create new edges
    new_edges = []

    # iterate over new nodes
    for i, node_dict in enumerate(node_storage):

        # unpack source and target
        a, b = node_dict['a'], node_dict['b']

        # get the edge embeddings (former node embedding)
        ab_old_edge_attr = node_dict['old_edge_attr']

        # get the attributes of the source node
        a_attr = node_dict['a_attr']

        # get the indices ingoing to a
        a_in_nodes_indices = [d['node_idx'] for d in in_nodes[a]]

        # iterate over them
        for in_node_c in a_in_nodes_indices:
            # fetch the current ingoing node
            in_node = node_storage[in_node_c]
            # ... and extract the node embedding
            ca_old_edge_attr = in_node['old_edge_attr']

            # e_{(i, j), (j, k)} = e'_(i, j) | x'_j | e'_{k, j}:
            edge_attr = torch.cat([ca_old_edge_attr, a_attr, ab_old_edge_attr])
            new_edges.append({'edge': [in_node_c, i], 'edge_attr': edge_attr})

    parallel_node_index = []
    for node_dict in node_storage:
        a, b = node_dict['a'], node_dict['b']
        parallel_idx = node_mapping[(b, a)]
        parallel_node_index.append(parallel_idx)

    new_x = [d['node_attr'] for d in node_storage]
    new_pos = [d['pos'] for d in node_storage]
    new_edge_index = [d['edge'] for d in new_edges]
    new_edge_attr = [d['edge_attr'] for d in new_edges]
    new_x = torch.stack(new_x)
    new_pos = torch.stack(new_pos)
    new_edge_index = torch.tensor(new_edge_index).T
    new_edge_attr = torch.stack(new_edge_attr)
    parallel_node_index = torch.tensor(parallel_node_index)

    data = torch_geometric.data.Data(x=new_x, edge_index=new_edge_index, edge_attr=new_edge_attr, pos=new_pos)
    data.parallel_node_index = parallel_node_index
    data.circle_index = get_circle_index(data, clockwise=False)
    return data, node_mapping


def remove_duplicate_edges_function(data, node_mapping):
    """
    Function to remove duplicate nodes

    :param data:
    :param node_mapping:
    :return: transformed graph, new node mapping
    :rtype: typing.Tuple[torch_geometric.data.Data, dict]
    """

    # determine which nodes to keep and which to discard
    keep_nodes = []
    discard_nodes = []

    for source, target in node_mapping.keys():
        if (target, source) in keep_nodes:
            discard_nodes.append((source, target))
        else:
            keep_nodes.append((source, target))

    # get indices of keep and discard
    keep_idx = torch.tensor([
        node_mapping[node]
        for node in keep_nodes
    ])
    discard_idx = torch.tensor([
        node_mapping[node]
        for node in discard_nodes
    ])

    # modify edge set so that bad nodes are swapped out with good nodes
    def _replace(i: int):
        if i in keep_idx:
            return i
        else:
            return data.parallel_node_index[i]

    # subselect edge related data
    edge_index = torch.tensor([
        [
            _replace(source),
            _replace(target)
        ]
        for source, target in data.edge_index.T.tolist()
    ]).T

    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    edge_attr = data.edge_attr[mask]

    # map indices of edge index to new nodes
    key, value = keep_idx.sort()
    d = dict(zip(key.tolist(), value.tolist()))
    edge_index = torch.tensor([[d[a], d[b]] for a, b in edge_index.T.tolist()]).T

    nodes = []
    # create absolute choice of which node embedding to choose
    for idx in keep_idx:
        # get elements to compare
        x = data.x[idx]
        y = data.x[data.parallel_node_index[idx]]

        # compare
        one = x < y
        two = x > y

        # catch exception where two nodes are identical
        if (not one.any()) and (not two.any()):
            nodes.append(x)
        # absolute comparison
        elif torch.where(one)[0][0] < torch.where(two)[0][0]:
            nodes.append(x)
        else:
            nodes.append(y)
    nodes = torch.stack(nodes)

    # merge ptrees
    def _map_tree(tree):
        if isinstance(tree, dict):
            _map_tree(next(iter(tree.values())))
        elif isinstance(tree, list):
            for i in range(len(tree)):
                if isinstance(tree[i], int):
                    tree[i] = d[int(_replace(tree[i]))]
                else:
                    _map_tree(tree[i])
        return tree

    def _merge_map_ptrees(tree_a, tree_b, idx):
        # get s node rooted list
        tree_a = json.loads(tree_a)['S']
        tree_b = json.loads(tree_b)['S']

        # catch cases where tree is just nonsensical
        if len(tree_a) < 3 and len(tree_b) < 3:
            return json.dumps(_map_tree({"S": [int(idx)]}))
        elif len(tree_a) < 3:
            return json.dumps(_map_tree({
                "S": [
                    int(idx),
                    tree_b[-1]
                ]
            }))
        elif len(tree_b) < 3:
            return json.dumps(_map_tree({
                "S": [
                    int(idx),
                    tree_a[-1]
                ]
            }))
        # create tree
        tree_a = tree_a[-1]
        tree_b = tree_b[-1]
        # map with _replace, then map with d dict
        return json.dumps(_map_tree({
            "S": [
                int(idx),
                {
                    "P": [
                        tree_a,
                        tree_b
                    ]
                }
            ]
        }))

    ptrees = [
        _merge_map_ptrees(data.ptree[idx], data.ptree[data.parallel_node_index[idx]], idx)
        for idx in keep_idx
    ]

    # set elements into data
    new_data = torch_geometric.data.Data(
        x=nodes,
        edge_index=edge_index,
        edge_attr=edge_attr,
        ptree=ptrees
    )

    # filter and map node_mapping
    node_mapping = dict(zip(keep_nodes, range(len(keep_nodes))))

    return new_data, node_mapping


def permutation_tree_transformation(
        data: torch_geometric.data.Data,
        mol: rdkit.Chem.rdchem.Mol,
        k: int,
        tetrahedral_chiral: bool = True,
        chiral_center_selective: bool = False,
        chiral_center_select_potential: bool = True,
        remove_duplicate_edges: bool = False,
        cis_trans_edges: bool = False,
        cis_trans_edges_select_potential: bool = False,
        axial_chirality: bool = False,
        create_order_matrix: bool = True,
        multi_stereo_center_dia: bool = False,
        multi_stereo_center_dia_mode: int = 1,
        separate_tree: bool = False,
        add_cyclic_trees: bool = False,
        cyclic_tree_mode: str = "complex",  # alternatives: light, minimal
        use_new_inv: bool = False
) -> torch_geometric.data.Data:
    """
    Function with many bool control variables. Determines how the permutation tree is constructed and to which
    stereoisomer to be sensitive to.

    :param data: vertex graph
    :param mol: molecule which corresponds to vertex graph
    :param k: number of neighbors to set into context. Not always necessary.
    :param tetrahedral_chiral: Sensitive to tetrahedral chiral centers if enabled
    :param chiral_center_selective: Sensitive to only marked tetrahedral chiral centers (and not by default to all
        nodes/atoms)
    :param chiral_center_select_potential: Sensitive also to potential stereocenters (if selective is true)
    :param remove_duplicate_edges: remove duplicate edges and reduce complexity - also often called reduction
    :param cis_trans_edges: enable cis/trans trees (only selective)
    :param cis_trans_edges_select_potential: enable cis/trans trees also for unmarked but potential cis/trans bonds
    :param axial_chirality: Enable axial chirality - requires that rdkit has correct position values stored (is not true
        by default for version 2023.*.*)
    :param create_order_matrix: whether or not to create the order matrix.
    :param multi_stereo_center_dia: Whether or not do multiple stereo center enantiomer invariance. Not advisable as
        model can distinguish these stereoisomers (see master thesis) but difference is 10^(-4) which is too small.
    :param multi_stereo_center_dia_mode: Mode that controls where to insert the MC trees. 0 creates new nodes that are
        not connected to remaining graph, 1 overwrites existing edges
    :param separate_tree: Whether to separate tree - make simplest trees with only one internal node for each ptree
        for each node.
    :param add_cyclic_trees: Add cyclic trees for graph - extra nodes, and more connections
    :param cyclic_tree_mode: Mode of the cylcic trees. Either complex, light or minimal - see master thesis
    :type cyclic_tree_mode: str
    :param use_new_inv: Whether or not to separate C and Q types into P(Z,Z) & P(S,S) (if false) else P(Z2,Z2) & P(S2,S2)
    :return: transformed graph
    :rtype: torch_geometric.data.Data
    """
    # transform to edge graph using custom function
    edge_graph, node_mapping = custom_to_edge_graph(
        data=data,
        # remove_duplicate_edges=remove_duplicate_edges  # removed as removal is done at the end
    )
    original_node_mapping = copy.deepcopy(node_mapping)

    # ==================================================================================================================
    # create default behaviour: P tree or C tree - in case of tetrahedral_chiral and !chiral_center_selective Z
    # ==================================================================================================================
    # get default type
    default_type = "P" if not tetrahedral_chiral or chiral_center_selective else "Z"

    # produce default trees for each node
    permutation_trees = [
        _circle_index_to_primordial_tree(circle_index, parallel_node, idx, inner_type=default_type)
        for idx, (circle_index, parallel_node) in enumerate(
            zip(
                edge_graph.circle_index,
                edge_graph.parallel_node_index
            )
        )
    ]
    # register permutation trees in data object
    edge_graph.ptree = permutation_trees

    # ==================================================================================================================
    # case for selective Z setting: there where chiral centers are present/possible
    # ==================================================================================================================
    # control over only_out parameter
    only_out = True
    # deactivated as not relevant
    # if remove_duplicate_edges:
    #    only_out = False

    # do selective chiral node differntiation
    if tetrahedral_chiral and chiral_center_selective:
        # get mask
        mask = get_chiral_center_node_mask(
            mol=mol,
            chiral_center_select_potential=chiral_center_select_potential,
            node_mapping=node_mapping,
            only_out=only_out
        )

        # generate new permutation trees for this mask
        for i in mask:
            edge_graph.ptree[i] = _circle_index_to_primordial_tree(
                edge_graph.circle_index[i],
                edge_graph.parallel_node_index[i],
                i,
                inner_type="Z"
            )

    # removal of double edges
    if remove_duplicate_edges:
        edge_graph, node_mapping = remove_duplicate_edges_function(edge_graph, node_mapping)

    # add cis/trans specific trees
    if cis_trans_edges:
        # calculate the bonds for which cis/trans permutation graphs should be created
        cis_trans_nodes_list = fetch_cis_trans_edges(node_mapping, mol, cis_trans_edges_select_potential)

        # iterate over edges and generate new ptrees for these entries
        for node_a, node_b in cis_trans_nodes_list:
            edge_graph.ptree[node_mapping[(node_a, node_b)]] = get_cistrans_tree(
                vertex_graph=data,
                node_a=node_a,
                node_b=node_b,
                node_mapping=node_mapping
            )

    if axial_chirality:
        # calculate candidates for axial chirality
        potential_axial_nodes_list = detect_possible_axial_chiral_edges(
            molecule=mol,
            node_mapping=node_mapping
        )

        # iterate over edges and generate new ptrees for these entries
        for node_a, node_b in potential_axial_nodes_list:
            edge_graph.ptree[node_mapping[(node_a, node_b)]] = get_cistrans_tree(
                vertex_graph=data,
                node_a=node_a,
                node_b=node_b,
                node_mapping=node_mapping
            )

    # in case only diastereomers are to be distinguished
    if multi_stereo_center_dia:
        # get stereo paths
        stereo_paths = calc_edges_multiple_stereo_centers(mol, chiral_center_select_potential)
        #todo: pairwise connections between all stereo centers

        # iterate over stereo_paths
        for (source_a, source_b), (target_a, target_b) in stereo_paths:
            # get circle indices
            source_circle_index = edge_graph.circle_index[original_node_mapping[(source_a, source_b)]]
            target_circle_index = edge_graph.circle_index[original_node_mapping[(target_b, target_a)]]

            # create tree to insert
            temp_tree = {
                "P": [
                    {
                        "P": [
                            {"Z": source_circle_index},
                            {"Z": target_circle_index}
                        ]
                    },
                    {
                        "P": [
                            {"Z": source_circle_index[::-1]},
                            {"Z": target_circle_index[::-1]}
                        ]
                    }
                ]
            }

            if multi_stereo_center_dia_mode == 0:
                create_new_node = True
                if (source_a, target_b) in node_mapping:
                    edge_graph.ptree[node_mapping[(source_a, target_b)]] = json.dumps(temp_tree)
                    create_new_node = False
                if (target_b, source_a) in node_mapping:
                    edge_graph.ptree[node_mapping[(target_b, source_a)]] = json.dumps(temp_tree)
                    create_new_node = False

                if create_new_node:
                    node_mapping[(source_a, target_b)] = len(node_mapping)
                    edge_graph.x = torch.cat([edge_graph.x, torch.zeros(1, edge_graph.x.shape[-1])], dim=0)
                    edge_graph.ptree.append(json.dumps(temp_tree))
            elif multi_stereo_center_dia_mode == 1:
                if (source_a, source_b) in node_mapping:
                    tree = json.loads(edge_graph.ptree[node_mapping[(source_a, source_b)]])
                    tree['S'][-1] = temp_tree
                    edge_graph.ptree[node_mapping[(source_a, source_b)]] = json.dumps(tree)

                if (target_b, target_a) in node_mapping:
                    tree = json.loads(edge_graph.ptree[node_mapping[(target_b, target_a)]])
                    tree['S'][-1] = temp_tree
                    edge_graph.ptree[node_mapping[(target_b, target_a)]] = json.dumps(tree)


    if add_cyclic_trees:
        if cyclic_tree_mode == "complex":
            edge_graph = cyclic_tree_edge(edge_graph, mol, node_mapping)
        elif cyclic_tree_mode == "light":
            edge_graph = cyclic_tree_edge_light(edge_graph, mol, node_mapping)
        elif cyclic_tree_mode == "minimal":
            edge_graph = cyclic_tree_edge_minimal(edge_graph, mol, node_mapping)
        else:
            raise NotImplementedError(f"cyclic tree mode {cyclic_tree_mode} is not implemented.")

    if separate_tree:
        edge_graph = separate_tree_into_subtrees(edge_graph)

    if create_order_matrix:
        # generate order matrix
        edge_graph = permutation_tree_to_order_matrix(edge_graph, k, use_new_inv)

    return edge_graph

