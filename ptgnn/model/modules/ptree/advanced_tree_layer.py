import torch.nn
import torch_geometric

from ptgnn.transform.ptree_matrix import type_dict


class AdvancedPermutationTreeLayer(torch.nn.Module):
    def __init__(
            self,
            k: int,
            hidden_dim: int,
            batch_norm: bool = False
    ):
        # set super
        super(AdvancedPermutationTreeLayer, self).__init__()

        # set own parameters
        self.k = k
        self.hidden_dim = hidden_dim

        # INITIALIZE LINEAR LAYERS
        # p_layer
        self.p_layer = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.p_final_layer = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        # z_layer
        self.z_layer = torch.nn.ModuleList([
            torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            for _ in range(self.k)
        ])
        self.z_final_layer = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        # s_layer
        self.s_layer = torch.nn.ModuleList([
            torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            for _ in range(self.k)
        ])
        self.s_final_layer = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        # init elu
        self.elu = torch.nn.ELU()
        # init node type dict
        self.type_dict = type_dict

        # batch norm
        self.do_batch_norm = batch_norm
        self.batch_norm = torch.nn.BatchNorm1d(self.hidden_dim)

    def forward(
            self,
            batch
    ):
        # create initial data space
        data_array = batch.x[batch.initial_map]

        # iterate over layers
        for layer_idx in range(batch.num_layer):

            # fetch instruction matrices for this layer
            order_matrix = batch[f"layer{layer_idx}_order_matrix"] + 1
            current_layer_pooling = batch[f"layer{layer_idx}_pooling"]
            type_mask = batch[f"layer{layer_idx}_type_mask"]

            # calc type mask after reduction
            type_mask_after = torch_geometric.nn.global_max_pool(type_mask, current_layer_pooling)

            # add zero layer to data_array
            data_array = torch.cat(
                [
                    torch.zeros(1, self.hidden_dim, device=data_array.device, requires_grad=False),
                    data_array
                ],
                dim=0
            )

            # init layer output
            layer_output = torch.zeros(
                current_layer_pooling.max() + 1,
                self.hidden_dim,
                device=data_array.device,
                requires_grad=False
            )

            # create mask for empty entries in order_matrix
            # why? remove element from null element after layers
            mask_order_matrix_empty = (order_matrix == 0)

            # extend mask
            mask_order_matrix_empty_extended = mask_order_matrix_empty.unsqueeze(-1).expand(-1, -1, self.hidden_dim)

            # select entries from data_array
            data_array = data_array[order_matrix]

            # iterate over different types and change layer_output accordingly
            for node_type, node_value in self.type_dict.items():

                # create mask for node_type
                node_type_mask = (type_mask == node_value)
                node_type_mask_after = (type_mask_after == node_value)

                # if the mask is empty skip this iteration
                if not node_type_mask.any():
                    continue

                if node_type == "P":
                    # send through first linear layer
                    after_p = self.p_layer(data_array[0, node_type_mask])
                    # sum up
                    after_p = torch_geometric.nn.global_add_pool(after_p, current_layer_pooling[node_type_mask])
                    # put through final layer
                    after_p = self.p_final_layer(after_p)

                    # do batch norm
                    if self.do_batch_norm:
                        after_p = self.batch_norm(after_p)

                    # put into layer_output
                    layer_output[node_type_mask_after] = layer_output[node_type_mask_after] + after_p[node_type_mask_after[:after_p.shape[0]]]

                elif node_type == "Z":
                    # shift stack
                    after_z = torch.stack([
                        self.z_layer[i](data_array[i, node_type_mask])
                        for i in range(self.k)
                    ], dim=0)

                    # masked cleaning empty entries
                    after_z = torch.masked_fill(
                        after_z,
                        mask_order_matrix_empty_extended[:, node_type_mask],
                        0.0
                    )

                    # sum
                    after_z = torch.sum(after_z, dim=0)
                    # apply elu
                    after_z = self.elu(after_z)
                    # linear layer
                    after_z = self.z_final_layer(after_z)

                    # masked cleaning empty entries
                    after_z = torch.masked_fill(
                        after_z,
                        mask_order_matrix_empty_extended.all(dim=0)[:after_z.shape[0]],
                        0.0
                    )

                    # global sum pooling
                    after_z = torch_geometric.nn.global_add_pool(after_z, current_layer_pooling[node_type_mask])
                    # apply elu
                    after_z = self.elu(after_z)

                    # do batch norm
                    if self.do_batch_norm:
                        after_z = self.batch_norm(after_z)

                    # put into layer output
                    layer_output[node_type_mask_after] = layer_output[node_type_mask_after] + after_z[node_type_mask_after[:after_z.shape[0]]]

                elif node_type == "S":
                    # shift stack
                    after_s = torch.stack([
                        self.s_layer[i](data_array[i, node_type_mask])
                        for i in range(self.k)
                    ], dim=0)

                    # masked cleaning empty entries
                    after_s = torch.masked_fill(
                        after_s,
                        mask_order_matrix_empty_extended[:, node_type_mask],
                        0.0
                    )

                    # sum
                    after_s = torch.sum(after_s, dim=0)
                    # apply elu
                    after_s = self.elu(after_s)
                    # apply layer
                    after_s = self.s_final_layer(after_s)

                    # masked cleaning empty entries
                    after_s = torch.masked_fill(
                        after_s,
                        mask_order_matrix_empty_extended.all(dim=0)[:after_s.shape[0]],
                        0.0
                    )

                    # global sum pooling
                    after_s = torch_geometric.nn.global_add_pool(after_s, current_layer_pooling[node_type_mask])
                    # apply elu
                    after_s = self.elu(after_s)

                    # do batch norm
                    if self.do_batch_norm:
                        after_s = self.batch_norm(after_s)

                    # put into layer output
                    layer_output[node_type_mask_after] = layer_output[node_type_mask_after] + after_s[node_type_mask_after[:after_s.shape[0]]]

            # ==========================================================================================================
            # notype tunneling to next layer
            # create mask for node_type
            node_type_mask = (type_mask == 0)
            node_type_mask_after = (type_mask_after == 0)

            layer_output[node_type_mask_after] = layer_output[node_type_mask_after] + data_array[0, node_type_mask]
            # ==========================================================================================================

            data_array = layer_output

            # todo: maybe transfer pooling and after pooling layers here?
            # apply global sum pooling
            # apply layers after sum pooling

        return data_array
