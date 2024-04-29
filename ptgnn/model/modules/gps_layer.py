import torch.nn
import torch_geometric.data.data
from torch.nn import Linear, Dropout, Sequential, ReLU, BatchNorm1d
from torch_geometric.nn import GraphNorm

from ptgnn.model.chienn.chienn_layer import ChiENNLayer
from ptgnn.model.modules.ptree.advanced_tree_layer import AdvancedPermutationTreeLayer
from ptgnn.model.modules.ptree.advanced_tree_layer_e import AdvancedPermutationEdgeTreeLayer
from ptgnn.model.modules.ptree.advanced_tree_layer_rnn import AdvancedPermutationTreeLayerRNN


class CustomGPSLayer(torch.nn.Module):
    """
    Adapted from https://github.com/gmum/ChiENN/blob/master/experiments/graphgps/layer/gps_layer.py#L17

    Model with residual connection around stored layer and a feed forward network (also with residual connection)
    """
    def __init__(
            self,
            hidden_dim: int,
            local_model: str,
            local_model_params: dict,
            global_model: str = None,
            dropout: float = 0.,
            norm_type: str = "batch",
            **kwargs
    ):
        """
        Init function of CustomGPSLayer

        :param hidden_dim: hidden dimension
        :type hidden_dim: int
        :param local_model: local model of the GPS layer, i.e. to wrap with a residual connection
        :type local_model: str
        :param local_model_params: parameters to pass to the init of the local model
        :type local_model_params: dict
        :param global_model: global model of the GPS model - not implemented, ignore it. (Original model had option to
            use a global model, i.e. Transformer and it was not clear whether this will be added or not).
        :type global_model: str
        :param dropout: dropout rate
        :type dropout: float
        :param norm_type: defines which norm to use. Either batch or layer
        :type norm_type: str
        :param kwargs: Auxiliary parameters
        """
        super(CustomGPSLayer, self).__init__()
        # local parameters
        self.hidden_dim = hidden_dim

        # local model diff
        if local_model == 'chienn':
            self.local_model = ChiENNLayer(
                dropout=dropout,
                hidden_dim=self.hidden_dim,
                **local_model_params
            )
        elif local_model == 'permutation_tree':
            self.local_model = AdvancedPermutationTreeLayer(
                hidden_dim=self.hidden_dim,
                **local_model_params
            )
        elif local_model == 'permutation_tree_rnn':
            self.local_model = AdvancedPermutationTreeLayerRNN(
                hidden_dim=self.hidden_dim,
                **local_model_params
            )
        elif local_model == 'permutation_tree_e':
            self.local_model = AdvancedPermutationEdgeTreeLayer(
                hidden_dim=self.hidden_dim,
                **local_model_params
            )
        else:
            raise NotImplementedError(f"local model {local_model} is not yet implemented")
        self.dropout_local = torch.nn.Dropout(dropout)

        # global model diff
        # transformer and this kind of stuff
        if global_model is None:
            self.global_model = None
        else:
            raise NotImplementedError("global model not implemented")

        # init norms
        if norm_type == "layer":
            self.norm1_local = GraphNorm(self.hidden_dim)
            self.norm1_attn = GraphNorm(self.hidden_dim)
            self.norm2 = GraphNorm(self.hidden_dim)
        elif norm_type == "batch":
            self.norm1_local = BatchNorm1d(self.hidden_dim)
            self.norm1_attn = BatchNorm1d(self.hidden_dim)
            self.norm2 = BatchNorm1d(self.hidden_dim)
        else:
            raise NotImplementedError(f"norm type {norm_type} is not implemented")

        # feed forward
        self.feed_forward = Sequential(*[
            Linear(self.hidden_dim, self.hidden_dim * 2),
            ReLU(),
            Dropout(dropout),
            Linear(self.hidden_dim * 2, self.hidden_dim),
            Dropout(dropout)
        ])

    def forward(
            self,
            batch
    ):
        initial_embedding = batch.x

        # put through local model
        local_node_embedding = self.local_model(batch)
        if isinstance(local_node_embedding, torch_geometric.data.data.BaseData):
            local_node_embedding = local_node_embedding.x

        local_node_embedding = self.dropout_local(local_node_embedding)

        # residual connection
        local_node_embedding = local_node_embedding + initial_embedding

        # norm
        if isinstance(self.norm1_local, GraphNorm):
            local_node_embedding = self.norm1_local(local_node_embedding, batch.batch)
        else:
            local_node_embedding = self.norm1_local(local_node_embedding)

        # feed forward with residual connection
        local_node_embedding = local_node_embedding + self.feed_forward(local_node_embedding)

        # norm
        if isinstance(self.norm1_local, GraphNorm):
            local_node_embedding = self.norm2(local_node_embedding, batch.batch)
        else:
            local_node_embedding = self.norm2(local_node_embedding)

        batch.x = local_node_embedding
        return batch
