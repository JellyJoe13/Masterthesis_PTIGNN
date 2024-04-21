import json

import torch_geometric
from typing import List

from ptgnn.transform.edge_graph.chienn.to_edge_graph import to_edge_graph
from ptgnn.transform.ptree_matrix import permutation_tree_to_order_matrix


def _circle_index_to_primordial_tree(
        circle_index: List[int],
        parallel_node: int,
        self_node: int,
        inner_type: str = "Z"
):
    """
    Turns a ChiENN cycle index to its equivalent permutation tree (S(self, parallel, Type(cycle_index))).

    :param circle_index: circle index, also often called cycle index because of the cyclic order it imposes
    :type circle_index: List[int]
    :param parallel_node: parallel node (each edge has a parallel node due to its edge graph transformation)
    :type parallel_node: int
    :param self_node: node itself
    :type self_node: int
    :param inner_type: Specifies which inner type to choose - Type of aforementioned tree. Default Z.
    :type inner_type: str
    :return: permutation tree as a string
    :rtype: str
    """
    # if nothing in the circular index return empty string
    if len(circle_index) == 0:
        return json.dumps({"S": [
            int(self_node),
            int(parallel_node)
        ]})
    else:
        # not including parallel node index
        # return f"Z{[i for i in circle_index]}"

        # including parallel node index
        # return f"P[{parallel_node}, Z{[i for i in circle_index]}]"
        return json.dumps({
            "S": [
                int(self_node),
                int(parallel_node),
                {
                    inner_type: [int(i) for i in circle_index]
                }
            ]
        })


def basic_permutation_tree_chienn_replication(data: torch_geometric.data.Data):
    """
    Wrapper function for _circle_index_to_primordial_tree(...) function and the to edge graph transformation.

    :param data: input data (vertex graph)
    :type data: torch_geometric.data.Data
    :return: edge graph with permutation tree
    :rtype: torch_geometric.data.Data
    """
    edge_graph = to_edge_graph(data)

    ptree = [
        _circle_index_to_primordial_tree(circle_index, parallel_node, idx)
        for idx, (circle_index, parallel_node) in enumerate(zip(edge_graph.circle_index, edge_graph.parallel_node_index))
    ]

    new_element = torch_geometric.data.Data(
        x=edge_graph.x,
        edge_index=edge_graph.edge_index,
        edge_attr=edge_graph.edge_attr,
        pos=edge_graph.pos,
        y=edge_graph.y,
        ptree=ptree
    )

    return new_element


def order_matrix_permutation_tree_chienn_replication(data: torch_geometric.data.Data, k):
    """
    Wrapper around basic_permutation_tree_chienn_replication(...) which produces the order matrices which are required
    for the advanced permutation tree model.

    :param data: vertex graph
    :type data: torch_geometric.data.Data
    :param k: k value - how many elements to set into context at a time
    :type k: int
    :return: edge graph with permutation tree and order matrix(-ces)
    :rtype: torch_geometric.data.Data
    """
    data = basic_permutation_tree_chienn_replication(data)

    # generate order matrix
    data = permutation_tree_to_order_matrix(data, k)

    return data
