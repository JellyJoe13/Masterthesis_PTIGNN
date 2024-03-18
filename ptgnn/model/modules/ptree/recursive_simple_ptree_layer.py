import torch.nn
import torch_geometric.data


class RecursiveSimplePtreeLayer(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            k: int
    ):
        super(RecursiveSimplePtreeLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.k = k

        # z layer
        self.z_layer = torch.nn.ModuleList(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            for _ in range(self.k)
        )
        self.z_final_layer = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.z_elu = torch.nn.ELU()

        # p layer
        self.p_layer = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.p_final_layer = torch.nn.Linear(self.hidden_dim, self.hidden_dim)

        # c layer
        # uses layers of z.

        # s layer
        self.s_layer = torch.nn.ModuleList([
            torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            for _ in range(self.k)
        ])
        self.s_final_layer = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        # q layer

    def forward(self, batch: torch_geometric.data.Batch):
        return torch.stack([
            self.do_module(batch, ptree)
            for ptree in batch.ptree
        ])

    def do_module(self, batch, tree):
        # get node type
        node_type = next(iter(tree.keys()))

        # iterate over indices/trees and replace by data
        data_list = [
            self.fetch_or_produce(batch, subtree)
            for subtree in tree[node_type]
        ]

        # stack data
        data_list = torch.stack(data_list)

        # put through layer
        if node_type == "P":
            return self.p_layer_fn(data_list)
        elif node_type == "Z":
            return self.z_layer_fn(data_list)
        elif node_type == "C":
            return self.c_layer_fn(data_list)
        elif node_type == "S":
            return self.s_layer_fn(data_list)
        elif node_type == "Q":
            return self.q_layer_fn(data_list)
        else:
            raise NotImplementedError(f"permutation tree node type {node_type} is not implemented.")

    def p_layer_fn(self, data_list):
        # run through layer for each input
        after_p = self.p_layer(data_list)
        # sum up
        after_p = torch.sum(after_p, dim=0)
        # put through final linear layer
        after_p = self.p_final_layer(after_p)

        return after_p

    def z_layer_fn(self, data_list):
        # run through embedding layers
        after_z = [
            layer(data_list)
            for layer in self.z_layer
        ]
        # shift
        _idx = torch.arange(data_list.shape[0])
        after_z = [
            embedding[torch.roll(_idx, shifts=idx), :]
            for idx, embedding in enumerate(after_z)
        ]
        after_z = torch.stack(after_z)
        # sum
        after_z = torch.sum(after_z, dim=0)
        # apply ELU
        after_z = self.z_elu(after_z)
        # layer
        after_z = self.z_final_layer(after_z)
        # sum
        after_z = torch.sum(after_z, dim=0)
        return self.z_elu(after_z)

    def c_layer_fn(self, data_list):
        one_direction = self.z_layer_fn(data_list)
        other_direction = self.z_layer_fn(data_list[::-1])

        return self.p_layer_fn([one_direction, other_direction])

    def s_layer_fn(self, data_list):
        # run through embedding layers
        after_s = [
            layer(data_list)
            for layer in self.s_layer
        ]
        # shift
        _idx = torch.arange(data_list.shape[0])
        after_s = [
            embedding[idx:embedding.shape[-2]-self.k+idx+1, :]
            for idx, embedding in enumerate(after_s)
        ]
        after_s = torch.stack(after_s)
        # sum
        after_s = torch.sum(after_s, dim=0)
        # apply ELU
        after_s = self.s_elu(after_s)
        # layer
        after_s = self.s_final_layer(after_s)
        # sum
        after_s = torch.sum(after_s, dim=0)
        return self.s_elu(after_s)

    def q_layer_fn(self, data_list):
        one_direction = self.s_layer_fn(data_list)
        other_direction = self.s_layer_fn(data_list[::-1])

        return self.p_layer_fn([one_direction, other_direction])

    def fetch_or_produce(self, batch, subtree):
        if isinstance(subtree, int):
            return batch.x[subtree]
        else:
            return self.do_module(batch, subtree)
