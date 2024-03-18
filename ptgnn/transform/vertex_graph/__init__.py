from .permutation_tree_selective import permutation_tree_vertex_transformation


def vertex_graph_transform(
        data,
        transformation_mode=None,
        transformation_parameters={},
        mol=None
):
    # currently does not alter data as input graph is a vertex graph
    if transformation_mode == "default":
        return data
    elif transformation_mode == "permutation_tree":
        return permutation_tree_vertex_transformation(data, mol, **transformation_parameters)
    else:
        raise NotImplementedError(f"Mode {transformation_mode} is not implemented in the function "
                                  f"vertex_graph_transform")
