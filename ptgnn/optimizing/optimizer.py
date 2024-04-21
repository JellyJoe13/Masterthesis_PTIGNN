import torch
import typing


def adam_w_optimizer(
        parameter: typing.Iterator[torch.nn.Parameter],
        base_learning_rate: float,
        weight_decay: float,
        **kwargs
) -> torch.optim.AdamW:
    """
    Helper function that constructs the adam optimizer.

    Adapted from
    https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/experiments/graphgps/optimizer/extra_optimizers.py

    :param parameter: Parameters of model to optimize
    :type parameter: typing.Iterator[torch.nn.Parameter]
    :param base_learning_rate: initial learning rate
    :type base_learning_rate: float
    :param weight_decay: weight decay parameter
    :type weight_decay: float
    """
    return torch.optim.AdamW(
        params=parameter,
        lr=base_learning_rate,
        weight_decay=weight_decay
    )
