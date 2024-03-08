import torch.nn
import torch_geometric.data


class GraphEmbedding(torch.nn.Module):
    def __init__(
            self,
            node_in_dim: int,
            node_out_dim: int,
            edge_in_dim: int,
            edge_out_dim: int,
            node_type: str,
            edge_type: str,
            **kwargs
    ):
        # model init
        super(GraphEmbedding, self).__init__()

        # set internal params
        self.edge_out_dim = edge_out_dim
        self.edge_in_dim = edge_in_dim
        self.node_out_dim = node_out_dim
        self.node_in_dim = node_in_dim

        if node_type == "linear":
            self.node_embedder = torch.nn.Linear(node_in_dim, node_out_dim)
        else:
            raise NotImplementedError(f"type {node_type} is not yet implemented for GraphEmbedding.")

        if edge_type == 'linear':
            self.edge_embedder = torch.nn.Linear(edge_in_dim, edge_out_dim)
        else:
            raise NotImplementedError(f"type {edge_type} is not yet implemented for GraphEmbedding.")

    def forward(self, batch: torch_geometric.data.Batch):
        # apply node embedder
        batch.x = self.node_embedder(batch.x)

        # apply edge embedder
        batch.edge_attr = self.edge_embedder(batch.edge_attr)

        return batch
        