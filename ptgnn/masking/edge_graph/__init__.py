import torch_geometric as pyg
from ..vertex_graph import num_default_node_feature_last_chiral, num_default_edge_attr_last_chiral


def edge_graph_masking(data: pyg.data.Data) -> pyg.data.Data:
    """
    Based on the rules from ``ptgnn.masking.vertex_graph.vertex_graph_masking(...)`` in combination with the edge graph
    transformation in ``ptgnn.transfrom.edge_graph.to_edge_graph.to_edge_graph(...)`` this behavior is created.
    Adapted/rewritten from:
    https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/experiments/graphgps/dataset/utils.py#L220
    """
    # get dimension shapes
    edge_dim = 14
    node_dim = 52

    # get how many chiral positions need to be masked
    n_node_chiral = num_default_node_feature_last_chiral
    n_edge_chiral = num_default_edge_attr_last_chiral

    # new node is an edge (in the edge graph)
    # data for this consists of node | edge | node
    # first part (node)
    data.x[:, node_dim - n_node_chiral: node_dim] = 0.
    # second part (edge)
    data.x[:, node_dim + edge_dim - n_edge_chiral: node_dim + edge_dim] = 0.
    # third part (node)
    data.x[:, node_dim + edge_dim + node_dim - n_node_chiral: node_dim + edge_dim + node_dim] = 0.

    # new edge is a node connection of two edges
    # data for this consists of edge | node | edge
    # first part (edge)
    data.edge_attr[:, edge_dim - n_edge_chiral: edge_dim] = 0.
    # second part (node)
    data.edge_attr[:, edge_dim + node_dim - n_node_chiral: edge_dim + node_dim] = 0.
    # third part (edge)
    data.edge_attr[:, edge_dim + node_dim + edge_dim - n_edge_chiral: edge_dim + node_dim + edge_dim] = 0.

    return data
