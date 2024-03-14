import json

import torch_geometric
from typing import List

from ptgnn.transform.edge_graph.chienn.to_edge_graph import to_edge_graph
from ptgnn.transform.ptree_matrix import permutation_tree_to_order_matrix


def basic_permutation_tree_chienn_replication(data: torch_geometric.data.Data):
    edge_graph = to_edge_graph(data)

    def _circle_index_to_primordial_tree(circle_index: List[int], parallel_node: int):
        # if nothing in the circular index return empty string
        if len(circle_index) == 0:
            return json.dumps({"P": [int(parallel_node)]})
        else:
            # not including parallel node index
            # return f"Z{[i for i in circle_index]}"

            # including parallel node index
            # return f"P[{parallel_node}, Z{[i for i in circle_index]}]"
            return json.dumps({
                "P": [
                    int(parallel_node),
                    {
                        "Z": [int(i) for i in circle_index]
                    }
                ]
            })

    ptree = [
        _circle_index_to_primordial_tree(circle_index, parallel_node)
        for circle_index, parallel_node in zip(edge_graph.circle_index, edge_graph.parallel_node_index)
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
    data = basic_permutation_tree_chienn_replication(data)

    # generate order matrix
    data = permutation_tree_to_order_matrix(data, k)

    return data
