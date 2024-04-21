import math

import torch.optim


def cosine_with_warmup_scheduler(
        optimizer: torch.optim.Optimizer,
        num_warmup_epochs: int,
        max_epochs: int,
        last_epoch: int = -1,
        num_cycles: float = 0.5,
        **kwargs
):
    """
    Helper function that builds cosine with warmup scheduler.

    Adapted from
    https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/experiments/graphgps/optimizer/extra_optimizers.py

    :param optimizer: Optimizer for which the learning rate is to be controlled
    :type optimizer: torch.optim.Optimizer
    :param num_warmup_epochs: Number of warmup epochs
    :type num_warmup_epochs: int
    :param max_epochs: Maximal epochs (diminishes learning rate when nearing end
    :type max_epochs: int
    :param last_epoch: Specifies last epoch
    :type last_epoch: int
    :param num_cycles: number of cycles in cosine behavior
    :type num_cycles: float
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_epochs:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_epochs)))
        progress = float(current_step - num_warmup_epochs) / float(max(1, max_epochs - num_warmup_epochs))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)