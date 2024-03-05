import json
from typing import Optional, Iterable, List

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
            if hasattr(sample, 'circle_index'):
                return collate_with_circle_index(
                    data_list=batch,
                    k_neighbors=self.n_neighbors_in_circle
                )

            elif hasattr(sample, 'ptree'):
                return permutation_tree_collation(batch)

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
    batch = Batch.from_data_list(
        data_list=data_list,
        exclude_keys=['ptree']
    )
    batch.ptree = permutation_tree_batching(data_list)
    return batch
