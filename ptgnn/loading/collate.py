import json
from typing import Optional, Iterable, List

import torch
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData
from torch_geometric.loader.dataloader import Collater

from ptgnn.loading.chienn_collate import collate_with_circle_index


class UniversalCollater:
    """
    Inspiration/Extension of https://github.com/gmum/ChiENN/blob/master/experiments/graphgps/dataset/collate.py#L9
    """
    def __init__(
            self,
            follow_batch: Optional[Iterable[str]] = None,
            exclude_keys: Optional[Iterable[str]] = None,
            n_neighbors_in_circle: int = 3
    ):
        # set internal params
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.n_neighbors_in_circle = n_neighbors_in_circle

        # construct default collator
        self.default_collator = Collater(
            dataset=None,
            follow_batch=follow_batch,
            exclude_keys=exclude_keys
        )

    def __call__(
            self,
            batch: list
    ):
        # get sample element
        sample = batch[0]

        # logic for treating elements differently
        # check if it is a pyg data element (basedata)
        if isinstance(sample, BaseData):

            # if it has a circle index treat is as a chienn related data object
            # assumes that if not this field is deleted
            if hasattr(sample, 'ptree'):
                return permutation_tree_collation(batch)

            elif hasattr(sample, 'circle_index'):
                return collate_with_circle_index(
                    data_list=batch,
                    k_neighbors=self.n_neighbors_in_circle
                )

            # default treatment
            else:
                return self.default_collator(batch)


def permutation_tree_batching(data_list: List[BaseData]):
    # make container for transformed ptree strings
    ptree_batch_list = []

    # iterate over elements to be put into batch
    for data in data_list:

        # extract ptree string
        ptree_list = data.ptree
        # create dict from it
        ptree_list = [json.loads(ptree_string) for ptree_string in ptree_list]

        # elevate numbers in dict - will probably need to be recursive - check for int and if int then increase
        def _increase_int(d, increase):
            if increase == 0:
                return d
            if isinstance(d, dict):
                for key, value in d.items():
                    d[key] = _increase_int(value, increase)
                return d
            elif isinstance(d, list):
                return [
                    _increase_int(elem, increase)
                    for elem in d
                ]
            else:
                return d + increase
        ptree_list = [
            _increase_int(
                ptree_dict,
                increase=len(ptree_batch_list)
            )
            for ptree_dict in ptree_list
        ]
        # revert to string
        ptree_list = [
            json.dumps(ptree_dict)
            for ptree_dict in ptree_list
        ]
        # put into storage
        ptree_batch_list += ptree_list

    return ptree_batch_list


def permutation_tree_collation(
        data_list: List[BaseData]
):
    # create list of keys to exclude
    exclude_keys = ['ptree']
    if hasattr(data_list[0], 'num_layer'):
        num_layer = max([data.num_layer for data in data_list])

        exclude_keys += ['num_layer', 'initial_map']
        exclude_keys += [
            f"layer{layer_idx}_order_matrix"
            for layer_idx in range(num_layer)
        ]
        exclude_keys += [
            f"layer{layer_idx}_type_mask"
            for layer_idx in range(num_layer)
        ]
        exclude_keys += [
            f"layer{layer_idx}_pooling"
            for layer_idx in range(num_layer)
        ]

    batch = Batch.from_data_list(
        data_list=data_list,
        exclude_keys=exclude_keys
    )
    batch.ptree = permutation_tree_batching(data_list)

    if hasattr(data_list[0], 'num_layer'):
        batch.num_layer = num_layer

        # initial map
        pooling_max = []
        num_nodes = 0
        initial_map_matrix = []
        for data in data_list:
            initial_map_matrix.append(
                data.initial_map + num_nodes
            )
            pooling_max.append(data.initial_map.shape[0])
            num_nodes += data.x.shape[0]

        setattr(batch, "initial_map", torch.cat(initial_map_matrix))

        # get k parameter
        k = data_list[0].layer0_order_matrix[0][0].shape[0]

        # i know its stupid but having these wrapped in two lists was the best way to counter the auto-collation
        # function of pyg... without hardcoding the dimensions which

        for layer_idx in range(num_layer):

            # order matrix section and graph pooling
            order_matrix = []
            index_offset = 0

            for data, p_max in zip(data_list, pooling_max):
                o_m = data[f"layer{layer_idx}_order_matrix"][0][0]
                if o_m is None:
                    o_m = torch.stack([torch.arange(p_max)] + [torch.full((p_max,), -1) for _ in range(k-1)], dim=0)
                else:
                    o_m = o_m.clone()

                mask = (o_m == -1)
                o_m += index_offset
                o_m[mask] = -1

                index_offset += p_max
                order_matrix.append(o_m)

            batch[f"layer{layer_idx}_order_matrix"] = torch.cat(order_matrix, dim=1)

            # type mask section
            type_mask = [
                data[f"layer{layer_idx}_type_mask"][0][0]
                for data in data_list
            ]
            # iterate over type mask and create matrices for null situations
            for idx in range(len(type_mask)):
                if type_mask[idx] is None:
                    type_mask[idx] = torch.zeros(pooling_max[idx])
            batch[f"layer{layer_idx}_type_mask"] = torch.cat(type_mask, dim=0)

            # pooling
            old_pooling_max = pooling_max
            pooling_max = []
            index_offset = 0
            pool_matrix = []

            for data, pmax_old in zip(data_list, old_pooling_max):
                pool = data[f"layer{layer_idx}_pooling"][0][0]
                if pool is None:
                    pool = torch.arange(pmax_old)
                else:
                    pool = pool.clone()
                max_val = pool.max() + 1  # mind the 0

                pool += index_offset
                pool_matrix.append(pool)
                pooling_max.append(max_val)

                index_offset += max_val

            batch[f"layer{layer_idx}_pooling"] = torch.cat(pool_matrix, dim=0)

    return batch
