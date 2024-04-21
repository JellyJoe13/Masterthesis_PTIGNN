import torch.nn
import torch_geometric.data


class GraphEmbedding(torch.nn.Module):
    """
    Embedding module for a graph. Applies node and edge level embeddings and thus embed both of them to different sizes,
    normally hidden dimension.
    """
    def __init__(
            self,
            node_in_dim: int,
            node_out_dim: int,
            edge_in_dim: int,
            edge_out_dim: int,
            node_type: str,
            edge_type: str,
            rwse_params: dict = {},
            **kwargs
    ):
        """
        Init function of GraphEmbedding class

        :param node_in_dim: node input dimension
        :type node_in_dim: int
        :param node_out_dim: node output dimension
        :type node_out_dim: int
        :param edge_in_dim: edge input dimension
        :type edge_in_dim: int
        :param edge_out_dim: edge output dimension
        :type edge_out_dim: int
        :param node_type: type of embedding for the nodes
        :type node_type: str
        :param edge_type: type of embedding for the edges
        :type edge_type: str
        :param rwse_params: parameters of the RWSE embedding (if applicable)
        :type rwse_params: dict
        :param kwargs: Auxiliary parameters
        """
        # model init
        super(GraphEmbedding, self).__init__()

        # set internal params
        self.edge_out_dim = edge_out_dim
        self.edge_in_dim = edge_in_dim
        self.node_out_dim = node_out_dim
        self.node_in_dim = node_in_dim

        if node_type == "linear":
            self.node_embedder = torch.nn.Linear(node_in_dim, node_out_dim)
        elif node_type == "linear+RWSE":
            # fetch dim_pos_enc
            dim_pos_enc = rwse_params['dim_pos_enc']

            self.pre_embedding = torch.nn.Linear(node_in_dim, node_out_dim - dim_pos_enc)

            self.node_embedder = RWSE(node_out_dim, **rwse_params)
        else:
            raise NotImplementedError(f"type {node_type} is not yet implemented for GraphEmbedding.")

        if edge_type == 'linear':
            self.edge_embedder = torch.nn.Linear(edge_in_dim, edge_out_dim)
        else:
            raise NotImplementedError(f"type {edge_type} is not yet implemented for GraphEmbedding.")

    def forward(self, batch: torch_geometric.data.Batch):
        # apply node embedder
        if isinstance(self.node_embedder, RWSE):
            batch.x = self.pre_embedding(batch.x)
            batch = self.node_embedder(batch)
        else:
            batch.x = self.node_embedder(batch.x)

        # apply edge embedder
        batch.edge_attr = self.edge_embedder(batch.edge_attr)

        return batch


class RWSE(torch.nn.Module):
    """
    Adapted from https://github.com/gmum/ChiENN/blob/master/experiments/graphgps/encoder/kernel_pos_encoder.py
    """
    def __init__(
            self,
            dim_emb,
            dim_pos_enc: int = 16,
            model_type: str = "linear",
            n_layers: int = 3,
            norm_type: str = 'batchnorm',  # or 'none'
    ):
        super(RWSE, self).__init__()

        if dim_emb - dim_pos_enc < 1:
            raise ValueError(f"PE dim size {dim_pos_enc} is too large for "
                             f"desired embedding size of {dim_emb}.")

        num_rw_steps = 20

        if norm_type == 'batchnorm':
            self.raw_norm = torch.nn.BatchNorm1d(num_rw_steps)
        else:
            self.raw_norm = None

        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(torch.nn.Linear(num_rw_steps, dim_pos_enc))
                layers.append(torch.nn.ReLU())
            else:
                layers.append(torch.nn.Linear(num_rw_steps, 2 * dim_pos_enc))
                layers.append(torch.nn.ReLU())
                for _ in range(n_layers - 2):
                    layers.append(torch.nn.Linear(2 * dim_pos_enc, 2 * dim_pos_enc))
                    layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Linear(2 * dim_pos_enc, dim_pos_enc))
                layers.append(torch.nn.ReLU())
            self.pe_encoder = torch.nn.Sequential(*layers)
        elif model_type == 'linear':
            self.pe_encoder = torch.nn.Linear(num_rw_steps, dim_pos_enc)
        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                             f"'{model_type}' encoder model.")

    def forward(self, batch):
        # requires precomputing
        pestat_var = f"pestat_RWSE"
        if not hasattr(batch, pestat_var):
            raise ValueError(f"Precomputed '{pestat_var}' variable is "
                             f"required for {self.__class__.__name__}; set "
                             f"config 'posenc_{self.kernel_type}.enable' to "
                             f"True, and also set 'posenc.kernel.times' values")

        pos_enc = getattr(batch, pestat_var)  # (Num nodes) x (Num kernel times)
        # pos_enc = batch.rw_landing  # (Num nodes) x (Num kernel times)
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)  # (Num nodes) x dim_pe

        # add normal embedding to pos embedding
        h = batch.x

        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, pos_enc), 1)

        return batch
        