import torch_geometric as pyg

num_default_node_feature_last_chiral = 9
num_default_edge_attr_last_chiral = 7


def vertex_graph_masking(data: pyg.data.Data) -> pyg.data.Data:
    """
    According to
    https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/experiments/graphgps/dataset/utils.py#L209
    is to remove last 9 elements from node features and last 7 from edge features. (Removing by zeroing)
    """
    data.x[:, -num_default_node_feature_last_chiral:] = 0.
    data.edge_attr[:, -num_default_edge_attr_last_chiral] = 0.

    return data
