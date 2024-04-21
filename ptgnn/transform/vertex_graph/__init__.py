from .permutation_tree_selective import permutation_tree_vertex_transformation


def vertex_graph_transform(
        data,
        transformation_mode=None,
        transformation_parameters={},
        mol=None
):
    """
    Transformation function switch for vertex graph transformations.

    :param data: vertex graph
    :type data: torch_geometric.data.Data
    :param transformation_mode: transformation mode, determines which registered function to use for the transformation
    :type transformation_mode: str
    :param transformation_parameters: parameters to pass to the registered transformation functon
    :type transformation_parameters: dict
    :param mol: molecule
    :type mol: rdkit.Chem.rdchem.Mol
    :return: torch geometric data object using input data and molecule
    :rtype torch_geometric.data.Data
    """
    # currently does not alter data as input graph is a vertex graph
    if transformation_mode == "default":
        return data
    elif transformation_mode == "permutation_tree":
        return permutation_tree_vertex_transformation(data, mol, **transformation_parameters)
    else:
        raise NotImplementedError(f"Mode {transformation_mode} is not implemented in the function "
                                  f"vertex_graph_transform")
