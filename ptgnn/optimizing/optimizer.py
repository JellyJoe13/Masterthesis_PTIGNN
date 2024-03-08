import torch
import typing


def adam_w_optimizer(
        parameter: typing.Iterator[torch.nn.Parameter],
        base_learning_rate: float,
        weight_decay: float,
        **kwargs
) -> torch.optim.AdamW:
    """
    Adapted from
    https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/experiments/graphgps/optimizer/extra_optimizers.py
    """
    return torch.optim.AdamW(
        params=parameter,
        lr=base_learning_rate,
        weight_decay=weight_decay
    )
