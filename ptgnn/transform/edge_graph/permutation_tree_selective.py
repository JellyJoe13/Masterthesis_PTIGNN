import json
from collections import defaultdict

import torch_geometric
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from ptgnn.transform.edge_graph.basic_permutation_tree import _circle_index_to_primordial_tree
from ptgnn.transform.edge_graph.chienn.get_circle_index import get_circle_index
from ptgnn.transform.ptree_matrix import permutation_tree_to_order_matrix


def custom_to_edge_graph(data):
    """
    Modified version of to_edge_graph from ``ptgnn.transform.chienn.to_edge_graph``

    Original source:
    https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/chienn/data/edge_graph/to_edge_graph.py#L12
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


def remove_duplicate_edges_function(data, node_mapping):

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
        data,
        mol,
        k,
        tetrahedral_chiral: bool = True,
        chiral_center_selective: bool = False,
        chiral_center_select_potential: bool = True,
        remove_duplicate_edges: bool = False
) -> torch_geometric.data.Data:
    # transform to edge graph using custom function
    edge_graph, node_mapping = custom_to_edge_graph(
        data=data,
        # remove_duplicate_edges=remove_duplicate_edges  # removed as removal is done at the end
    )
    # todo: vertex transformation - do edge graph, collect circular indices and create constructs with it

    # ==================================================================================================================
    # create default behaviour: P tree or C tree - in case of tetrahedral_chiral and !chiral_center_selective Z
    # ==================================================================================================================
    # get default type
    default_type = "P" if tetrahedral_chiral and chiral_center_selective else "Z"

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
    data.ptree = permutation_trees

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
            data.ptree[i] = _circle_index_to_primordial_tree(
                data.circle_index[i],
                data.parallel_node[i],
                i,
                inner_type="Z"
            )

    # removal of double edges
    if remove_duplicate_edges:
        data, node_mapping = remove_duplicate_edges_function(data, node_mapping)

    # generate order matrix
    data = permutation_tree_to_order_matrix(data, k)

    return data

