from ptgnn.transform.edge_graph import edge_graph_transform
from ptgnn.transform.vertex_graph import vertex_graph_transform

PRE_TRANSFORM_MAPPING = {
    "vertex": vertex_graph_transform,
    "edge": edge_graph_transform
}
