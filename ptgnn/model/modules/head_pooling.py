import torch.nn
import torch_geometric as pyg


class SANHead(torch.nn.Module):
    """
    Head (Element that reduced graph object to graph level prediction), in this case simplified SANHead.

    Simplified reproduction of
    https://github.com/gmum/ChiENN/blob/master/experiments/graphgps/head/san_graph.py
    """
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            pool_function: str = 'add',
            n_layer: int = 3,
            **kwargs
    ):
        """
        Init function of SANHead

        :param in_dim: input dimension
        :type in_dim: int
        :param out_dim: output dimension
        :type out_dim: int
        :param pool_function: Pool aggregation function - needs to be permutation invariant
        :type pool_function: str
        :param n_layer: number of layers
        :type n_layer: int
        :param kwargs: Auxiliary parameters
        """
        super(SANHead, self).__init__()

        # select pooling function
        if pool_function == 'add' or pool_function == "sum":
            self.pool_function = pyg.nn.global_add_pool
        elif pool_function == "mean":
            self.pool_function = pyg.nn.global_mean_pool
        elif pool_function == "max":
            self.pool_function == pyg.nn.global_max_pool
        else:
            raise NotImplementedError(f"pool function {type} not implemented.")

        # init linear layers
        module_list = []
        for idx in range(n_layer - 1):
            module_list += [
                torch.nn.Linear(in_dim // (2 ** idx), in_dim // (2 ** (idx+1)), bias=True),
                torch.nn.ReLU()
            ]

        module_list.append(
            torch.nn.Linear(in_dim // (2 ** (n_layer-1)), out_dim, bias=True)
        )
        self.linear_list = torch.nn.Sequential(*module_list)

    def forward(
            self,
            batch: pyg.data.Batch
    ):
        # get graph embedding through pooling
        graph_embedding = self.pool_function(batch.x, batch.batch)

        # put graph embedding through linear layers to reduce to output dimension
        graph_embedding = self.linear_list(graph_embedding)

        # return prediction with label
        return graph_embedding, batch.y
