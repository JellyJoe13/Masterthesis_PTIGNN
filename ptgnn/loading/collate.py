from typing import Optional, Iterable, List

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


def permutation_tree_collation(
        batch: List[BaseData]
):
    # todo
    pass
