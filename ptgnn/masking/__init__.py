from ptgnn.masking.edge_graph import edge_graph_masking
from ptgnn.masking.vertex_graph import vertex_graph_masking

MASKING_MAPPING = {
    "vertex": vertex_graph_masking,
    "edge": edge_graph_masking
}
