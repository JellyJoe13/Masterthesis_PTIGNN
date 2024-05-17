import torch.nn
import typing

import torch_geometric.data

from ptgnn.model.chienn.chienn_layer import ChiENNLayer
from ptgnn.model.modules.custom_wrapper import CustomWrapper, InputOutputWrapper
from ptgnn.model.modules.gps_layer import CustomGPSLayer
from ptgnn.model.modules.graph_embedding import GraphEmbedding
from ptgnn.model.modules.head_pooling import SANHead
from ptgnn.model.modules.ptree.advanced_tree_layer import AdvancedPermutationTreeLayer
from ptgnn.model.modules.ptree.complex_ptree_layer import ComplexPtreeLayer


class CustomModel(torch.nn.Module):
    """
    Model designed to be as customizable as possible through the input dictionary that specifies internal structure
    """
    def __init__(
            self,
            data_sizes: typing.Tuple[int, int],
            model_config: typing.Dict[str, typing.Any],
    ):
        """
        Init function of ``CustomModel``

        :param data_sizes: node and edge dimension as a tuple
        :type data_sizes: tying.Tuple[int, int]
        :param model_config: Config as a dictionary which contents determine how the internal structure looks like
        :type model_config: typing.Dict[str, typing.Any]
        """
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
        out_dim = model_config['out_dim'] if 'out_dim' in model_config else 1

        for layer_idx in range(n_layers):
            # extract layer config
            layer_config = module_config[layer_idx]

            # get type
            layer_type = layer_config['type']
            # get number of times layer should be inserted
            n_times_layer = layer_config['times'] if 'times' in layer_config else 1

            # extract parameters
            param_config = layer_config['parameter'] if 'parameter' in layer_config else {}

            # repeatedly create layer
            for _ in range(n_times_layer):
                # create different layers
                if layer_type == 'graph_embedding':
                    modules.append(
                        GraphEmbedding(
                            node_in_dim=node_dim,
                            edge_in_dim=edge_dim,
                            node_out_dim=self.hidden_dim,
                            edge_out_dim=self.hidden_dim,
                            **param_config
                        )
                    )
                    node_dim = self.hidden_dim
                    edge_dim = self.hidden_dim
                elif layer_type == "chienn":
                    modules.append(
                        CustomWrapper(
                            ChiENNLayer(
                                hidden_dim=self.hidden_dim,
                                **param_config
                            )
                        )
                    )
                elif layer_type == "gps_layer":
                    modules.append(
                        CustomGPSLayer(
                            hidden_dim=self.hidden_dim,
                            **param_config
                        )
                    )
                elif layer_type == "permutation_tree":
                    modules.append(
                        CustomWrapper(
                            AdvancedPermutationTreeLayer(
                                hidden_dim=self.hidden_dim,
                                **param_config
                            )
                        )
                    )
                elif layer_type == "linear_dummy":
                    modules.append(
                        InputOutputWrapper(
                            torch.nn.Sequential(*[
                                torch.nn.Linear(self.hidden_dim, self.hidden_dim),
                                torch.nn.ReLU()
                            ])
                        )
                    )
                else:
                    raise NotImplementedError(f"layer type {layer_type} not yet implemented.")

        # make list to module list and save in param
        self.layers = torch.nn.ModuleList(modules)

        # add the head part - i.e. the part where graph pooling and reducing to output dimension happens
        head_config = model_config['head']
        if head_config['type'] == "san_head":
            self.head = SANHead(
                in_dim=self.hidden_dim,
                out_dim=out_dim,
                **head_config
            )
        else:
            raise NotImplementedError("Head type other than san_head not implemented.")

    def forward(
            self,
            batch: torch_geometric.data.Batch
    ) -> torch.Tensor:
        """
        Forward function of ``CustomModel``

        :param batch: Batch object to be passed through layers of this model and the head (pooling)
        :type batch: torch_geometric.data.Batch
        :return: Labels of the graphs in the input batch
        :rtype: torch.Tensor
        """
        for layer in self.layers:
            batch = layer(batch)

        return self.head(batch)
