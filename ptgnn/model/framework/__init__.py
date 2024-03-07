import torch.nn


class GpsFrameworkModel(torch.nn.Module):
    """
    Adapted from https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/experiments/graphgps/network/gps_model.py#L55
    """
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            hidden_dim: int,
            local_gnn_type: str,
            global_model_type: str,
            layers_pre_mp: int,
            num_gt_layer: int
    ):
        # init encoder
        self.encoder = UniversalEncoder(...)

        # pre model MPs
        if layers_pre_mp > 0:
            self.pre_mp = PreMP(
                dim_in,
                gnn_dim,  # cfg.gnn.dim_inner
                layers_pre_mp
            )

        gt_layer = []

        # create gt layer
        for i in range(num_gt_layer):
            gt_layer.append(
                GPSLayer(

                )
            )
    # todo: rethink it. This uses transformers, positional embeddings and co to realize their model...
    #   does not aid in comparing ChiENN and our approach. maybe use this at very end to show that performance is
    #   competitive? Aka switch ChiENN with PTGNN?
