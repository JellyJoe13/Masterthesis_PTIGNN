from ptgnn.transform.edge_graph.basic_permutation_tree import (
    basic_permutation_tree_chienn_replication,
    order_matrix_permutation_tree_chienn_replication
)
from ptgnn.transform.edge_graph.chienn.to_edge_graph import to_edge_graph
from ptgnn.transform.edge_graph.permutation_tree_selective import permutation_tree_transformation


def edge_graph_transform(
        data,
        transformation_mode='chienn',
        transformation_parameters={},
        mol=None
):
    if transformation_mode == 'default' or transformation_mode == 'chienn':
        return to_edge_graph(data)
    elif transformation_mode == "chienn_tree_basic":
        return basic_permutation_tree_chienn_replication(data)
    elif transformation_mode == "chienn_tree_order_matrix":
        return order_matrix_permutation_tree_chienn_replication(data, transformation_parameters['k'])
    elif transformation_mode == "permutation_tree":
        assert mol is not None
        return permutation_tree_transformation(
            data=data,
            mol=mol,
            **transformation_parameters
        )
    else:
        raise NotImplementedError(f"Mode {transformation_mode} is not implemented in the function edge_graph_transform")
