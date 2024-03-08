import torch.nn
import typing

from ptgnn.model.modules.graph_embedding import GraphEmbedding


class CustomModel(torch.nn.Module):
    def __init__(
            self,
            data_sizes: typing.Tuple[int, int],
            model_config: dict
    ):
        # model init
        super(CustomModel, self).__init__()

        # set internal parameters
        self.hidden_dim = model_config['hidden_dim']

        # init module list
        modules = []

        # extract module config
        module_config = model_config['modules']

        # get maximal number of layers
        n_layers = max([
            key
            for key, _ in module_config.items()
        ]) + 1

        # define dimensions
        node_dim, edge_dim = data_sizes

        for layer_idx in range(n_layers):
            # extract layer config
            layer_config = module_config[layer_idx]

            # get type
            layer_type = layer_config['type']
            if layer_type == 'graph_embedding':
                modules.append(
                    GraphEmbedding(
                        node_in_dim=node_dim,
                        edge_in_dim=edge_dim,
                        node_out_dim=self.hidden_dim,
                        edge_out_dim=self.hidden_dim,
                        **layer_config
                    )
                )
                node_dim = self.hidden_dim
                edge_dim = self.hidden_dim
            else:
                raise NotImplementedError(f"layer type {layer_type} not yet implemented.")

        # make list to module list and save in param
        self.layers = torch.nn.ModuleList(modules)

    def forward(self, batch):
        for layer in self.layers:
            batch = layer(batch)

        return batch
