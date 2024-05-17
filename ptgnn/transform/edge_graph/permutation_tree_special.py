import json

import torch


def get_cistrans_tree(vertex_graph, node_a, node_b, node_mapping):
    """
    Produce cistrans permutation tree for edge between two nodes.

    :param vertex_graph: vertex graph
    :type vertex_graph: torch_geometric.data.Data
    :param node_a: one of the nodes of the double bond in question
    :type node_a: int
    :param node_b: the other of the two nodes of the double bond in question
    :type node_b: int
    :param node_mapping: node mapping between node pairs to edges (in the course of edge graph transformation)
    :type node_mapping: dict
    :return: permutation tree
    :rtype: str
    """
    # get neighbors
    neighbors = vertex_graph.edge_index[:, vertex_graph.edge_index[1] == node_a]
    neighbors = neighbors[:, neighbors[0] != node_b]

    neighbors = torch.cat([neighbors, vertex_graph.edge_index[:, vertex_graph.edge_index[1] == node_b]], dim=-1)
    neighbors = neighbors[:, neighbors[0] != node_a]

    # get positions of neighbors
    pos = vertex_graph.pos[neighbors[0]]

    # calculate distances with 2-norm
    distances = torch.zeros(4,4)
    for i in range(4):
        for j in range(4):
            distances[i, j] = ((pos[i] - pos[j]) ** 2).sum().sqrt()

    def calc_order(distances):
        order = [0]
        while len(distances):
            rel_order = distances[order[-1]].argsort()
            for i in range(4):
                if not rel_order[i] in order:
                    order.append(int(rel_order[i]))
                    break
            else:
                return torch.tensor(order)

    # calculate order by getting the nearest of the neighbors, then the next until finished
    order = calc_order(distances)
    neighbor_order = neighbors[:, order]
    # possible todo: add check if it first goes to its counterpart, currently [x, x, y, y,], could be checked for

    # fetch the correct nodes for the neighbors
    edge_list = [
        node_mapping[(x, y)] if (x, y) in node_mapping else node_mapping[(y, x)]
        for x, y in neighbor_order.T.tolist()
    ]

    # current and reference node
    node_list = []
    if (node_a, node_b) in node_mapping:
        node_list += [node_mapping[(node_a, node_b)]]
    if (node_b, node_a) in node_mapping:
        node_list += [node_mapping[(node_b, node_a)]]
    # create tree and return it
    return json.dumps({
        "S": node_list + [
            {"C": edge_list}
        ]
    })
