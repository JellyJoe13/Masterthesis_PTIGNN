import torch
import torch_geometric

from ptgnn.transform.ptree_matrix import type_dict


class ComplexPtreeLayer(torch.nn.Module):
    def __init__(
            self,
            k: int,
            hidden_dim: int
    ):
        super(ComplexPtreeLayer, self).__init__()
        self.k = k
        self.hidden_dim = hidden_dim

        # s layer
        self.s_layer = torch.nn.ModuleList([
            torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            for _ in range(self.k)
        ])
        self.s_intermediate_layer = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.s_final_layer = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.s_elu = torch.nn.ELU()

        # z layer
        self.z_layer = torch.nn.ModuleList([
            torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            for _ in range(self.k)
        ])
        self.z_intermediate_layer = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.z_final_layer = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.z_elu = torch.nn.ELU()

        # p layer
        self.p_layer = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.p_elu = torch.nn.ELU()
        self.p_final_layer = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.final_elu = torch.nn.ELU()
        self.type_dict = type_dict

    def forward(
            self,
            batch
    ):
        # create initial data space
        data_array = batch.x[batch.initial_map]
        # todo: check if all -1 are minded - especially for s when whole order matrix col is -1

        # iterate over layers
        for layer_idx in range(batch.num_layer):

            # fetch instructions for this layer
            order_matrix = batch[f"layer{layer_idx}_order_matrix"] + 1
            current_layer_pooling = batch[f"layer{layer_idx}_pooling"]
            type_mask = batch[f"layer{layer_idx}_type_mask"]

            # order matrix contains -1. This is due to not all permutations requiring k elements to connect.
            # solve this with creating new index zero that is full of zeros and increasing order_matrix index
            data_array = torch.cat(
                [
                    torch.zeros(1, data_array.shape[-1], device=data_array.device),
                    data_array
                ],
                dim=0
            )

            # create order mask (required later to remove elements that were -1 -> 0 but after linear layer they
            # are no longer 0)
            mask_order_matrix = order_matrix == 0

            # change indices from order list to real data (technically this inflates the matrix, but all are references
            # so no actual space should be required.
            data_array = data_array[order_matrix]

            # do operation for each layer and put result into a temporary array
            # array initialized with the first column to cover the case of doing nothing (tree extension for uniform
            # layer count)
            # todo: replace with zero and add selectively to it
            temporary_data_array = data_array[0].clone()
            for node_type, node_value in self.type_dict.items():  # todo: where was this one dict again... features?
                # create mask for this type:
                mask = type_mask == node_value

                if not mask.any():
                    continue

                if node_type == "Z":
                    # iterate over k and get embeddings
                    temp = torch.stack([
                        emb(t)
                        for emb, t in zip(self.z_layer, data_array[:, mask])
                    ], dim=0)
                    # set entries that were previously only the -1's to zero
                    temp[mask_order_matrix[:, mask]] = 0.

                    # sum k (shifted [in order_matrix] embeddings)
                    temp = temp.sum(dim=0)

                    # run through final layer and put into temporary_array
                    temporary_data_array[mask] = self.z_intermediate_layer(temp)

                elif node_type == "S":
                    # iterate over k and get embeddings
                    temp = torch.stack([
                        emb(t)
                        for emb, t in zip(self.s_layer, data_array[:, mask])
                    ], dim=0)
                    # set entries that were previously only the -1's to zero
                    temp[mask_order_matrix[:, mask]] = 0.

                    # sum k (shifted [in order_matrix] embeddings)
                    temp = temp.sum(dim=0)

                    # run through final layer and put into temporary_array
                    temporary_data_array[mask] = self.s_intermediate_layer(temp)

                elif node_type == "P":
                    # no iterating over k layers
                    temp = self.p_layer(data_array[0, mask])

                    # put into temporary_array
                    temporary_data_array[mask] = temp
                else:
                    raise NotImplementedError("invalid node type")

            # in case of S type or errors, if order matrix contains only -1 entries in a column, set to 0
            temporary_data_array[mask_order_matrix.all(dim=0)] = 0.

            # global pooling
            # B = current_layer_pooling.max() + 1
            # data_array = torch.zeros(B, temporary_data_array.shape[-1], device=temporary_data_array.device)
            # for add_to, val in zip(current_layer_pooling, temporary_data_array):
            #     data_array[add_to] = data_array[add_to] + val
            data_array = torch_geometric.nn.global_add_pool(temporary_data_array, current_layer_pooling)

            # post aggregation embedding: ELU + linear
            # type_mask2 = type_mask[torch.unique(current_layer_pooling)]
            type_mask2 = torch_geometric.nn.global_max_pool(type_mask, current_layer_pooling)

            # apply final elu layer. why the masking? so that elements that are tunneled through don't get exposed to
            # elu layer before embedding
            mask = type_mask2 != 0
            data_array[mask] = self.final_elu(data_array[mask])

            # create mask for this type:
            # type p
            mask = type_mask2 == 1
            data_array[mask] = self.p_final_layer(data_array[mask])

            # type z
            # mask = type_mask2 == 2
            # data_array[mask] = self.z_final_layer(data_array[mask])

            # type s
            # mask = type_mask2 == 3
            # data_array[mask] = self.s_final_layer(data_array[mask])

        return data_array


