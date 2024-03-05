from ptgnn.transform.edge_graph.chienn.to_edge_graph import to_edge_graph


def edge_graph_transform(
        data,
        transformation_mode='chienn',
        transformation_parameters={}
):
    if transformation_mode == 'default' or transformation_mode == 'chienn':
        return to_edge_graph(data)
    else:
        raise NotImplementedError(f"Mode {transformation_mode} is not implemented in the function edge_graph_transform")
