"""
The contents of this file are adapted from
https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/chienn/model/utils.py
and serve the purpose of comparing this projects model to a pre-existing model.
"""
import torch
from torch import nn


def _build_single_embedding_layer(in_dim: int, out_dim: int, name: str):
    if name == 'linear':
        return nn.Linear(in_dim, out_dim, bias=False)
    elif name == 'identity':
        return nn.Identity()
    elif name == 'scalar':
        return nn.Linear(in_dim, 1, bias=True)
    elif name == 'self_concat':
        return lambda x: torch.cat([x, x], dim=-1)
    elif name == 'double':
        return lambda x: 2 * x
    elif hasattr(torch.nn, name):
        return getattr(torch.nn, name)()
    else:
        raise NotImplementedError(f'Layer name {name} is not implemented.')


def build_embedding_layer(in_dim: int, out_dim: int, name: str):
    """
    Function to build a pytorch embedding layer from a string that defines which type and if it is a composite type.

    :param in_dim: input dimension
    :type in_dim: int
    :param out_dim: output dimension
    :type out_dim: int
    :param name: name of the pytorch element(s)
    :type name: str
    :return: Embedding layer
    :rtype: torch.nn.Module
    """
    sub_names = name.split('+')
    if len(sub_names) == 1:
        return _build_single_embedding_layer(in_dim, out_dim, sub_names[0])
    else:
        return nn.Sequential(*[_build_single_embedding_layer(in_dim, out_dim, sub_name) for sub_name in sub_names])