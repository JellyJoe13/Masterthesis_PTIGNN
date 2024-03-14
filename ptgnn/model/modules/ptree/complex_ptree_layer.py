import torch
import torch_geometric


class ComplexPtreeLayer(torch.nn.Module):
    def __init__(
            self,
            k: int,
            hidden_dim: int
    ):
        super(ComplexPtreeLayer, self).__init__()
        self.k = k
        self.hidden_dim = hidden_dim

        self.z_layer = torch.nn.ModuleList([
            torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            for _ in range(k)
        ])

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

    def forward(
            self,
            batch
    ):
        # make link to batch idx_matrix
        idx_matrix = batch.idx_matrix

        # load first index range of elements
        data_array = batch.x[idx_matrix[:, -1]]

        # get structure to orient to
        idx_structure = idx_matrix[:, :-1]

        for layer_idx in range(idx_matrix.shape[1]-2):
            # get indexes for graph pooling
            idx_structure, current_layer_pooling_counts = torch.unique(idx_structure[:, :-1], dim=0, return_counts=True)

            # get indexes for graph pooling
            current_layer_pooling = torch.repeat_interleave(current_layer_pooling_counts)

            # init circling
            # todo: rework for other types - treat everything as Z
            order_matrix = torch.zeros(self.k, len(current_layer_pooling), dtype=torch.int) - 1

            cur_pos = 0
            for i in current_layer_pooling_counts:
                current_k = min(self.k, i)
                r = torch.arange(cur_pos, cur_pos+i)
                for j in range(current_k):
                    order_matrix[:current_k, cur_pos+j] = torch.roll(r, shifts=-j)
                cur_pos += i

            # add zero padding to data list
            data_array = torch.cat([torch.zeros(1, data_array.shape[-1], device=data_array.device), data_array], dim=0)
            order_matrix += 1

            embedding = data_array[order_matrix]
            # mask_z = batch.type_matrix

            embedding = torch.stack([
                emb(t)
                for emb, t in zip(self.z_layer, embedding)
            ], dim=1)

            embedding[(order_matrix.T == 0).unsqueeze(-1).expand_as(embedding)] = 0.

            embedding = embedding.sum(dim=1)
            embedding = self.z_final_layer(embedding)

            # global pooling
            data_array = torch_geometric.nn.global_add_pool(embedding, current_layer_pooling)



